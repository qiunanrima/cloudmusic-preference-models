"""Resumeable 16k/EBU loudness extraction, train-only PCA256, full retraining."""
import os
from pathlib import Path
import json
import subprocess
import concurrent.futures
import tempfile
import time
import csv

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
STAMP='20261030_16k_lufs16_pca256_v1'
ID='msclap2023-16kmono-loudnorm-I16-TP1.5-LRA11-2pass-center7-resample44100-fp32-v1'
RAW=HERE/'cache/library_msclap_16k_lufs16.npz'
REDUCED=HERE/'cache/library_msclap_16k_lufs16_pca256.npz'
RUN=ROOT/'models_archive/preprocess_16k_256'/STAMP
os.environ['PYTHONIOENCODING']='utf-8'
os.environ['MSCLAP_PREPROCESS_ID']=ID
os.environ['MSCLAP_VECTOR_DIM']='1024'
os.environ['NUMBA_CACHE_DIR']=str(HERE/'cache/numba16k')

import numpy as np
import torch
import soundfile as sf
import librosa
from scipy.signal import resample_poly
import joblib
from sklearn.decomposition import PCA
from msclap_audio import AudioEmbedder,read_vectors,write_vectors,unit_rows


def ffmpeg(args):
    result=subprocess.run(['ffmpeg','-hide_banner','-nostdin','-nostats','-threads','1',*args],capture_output=True,check=True)
    text=result.stderr.decode('utf-8',errors='replace')
    start=text.rfind('{'); stop=text.find('}',start)
    return json.loads(text[start:stop+1])


def process(item):
    index,path,folder=item
    pre='aresample=16000,aformat=channel_layouts=mono'
    stats=ffmpeg(['-i',path,'-map','0:a:0','-vn','-af',pre+',loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json','-f','null','-'])
    keys=('input_i','input_tp','input_lra','input_thresh','target_offset')
    if not all(np.isfinite(float(stats[k])) for k in keys): raise ValueError('Non-finite integrated loudness')
    filt=(pre+',loudnorm=I=-16:TP=-1.5:LRA=11:linear=true:print_format=json:'+
          'measured_I='+stats['input_i']+':measured_TP='+stats['input_tp']+
          ':measured_LRA='+stats['input_lra']+':measured_thresh='+stats['input_thresh']+
          ':offset='+stats['target_offset']+',aresample=16000')
    destination=folder/f'{index:06d}.wav'
    output=ffmpeg(['-i',path,'-map','0:a:0','-vn','-af',filt,'-ar','16000','-ac','1','-c:a','pcm_f32le',str(destination)])
    info=sf.info(destination)
    if info.samplerate!=16000 or info.channels!=1: raise ValueError('Unexpected waveform format')
    size=7*16000
    wave,_=sf.read(destination,start=max(0,(info.frames-size)//2),frames=min(size,info.frames),dtype='float32')
    if not len(wave) or not np.isfinite(wave).all(): raise ValueError('Invalid waveform')
    if len(wave)<size: wave=np.tile(wave,int(np.ceil(size/len(wave))))[:size]
    wave=resample_poly(wave,441,160).astype(np.float32)
    destination.unlink()
    return path,wave,{'audio_path':path,'input':stats,'output':output}


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    log=(RUN/'pipeline.log').open('a',encoding='utf-8',buffering=1)
    def say(s): print(s,flush=True); log.write(s+'\n')
    with np.load(HERE/'cache/library_msclap_128k.npz') as data: paths=list(map(str,data['paths']))
    if len(paths)!=len(set(paths)) or not all(Path(p).is_file() for p in paths): raise ValueError('Invalid manifest')
    saved={}
    if RAW.exists():
        p,v=read_vectors(RAW); saved=dict(zip(p,v))
    pending=[(i,p) for i,p in enumerate(paths) if p not in saved]
    say(f'16k mono; two-pass -16 LUFS; TP -1.5 dBTP; LRA11; center7; pending={len(pending)}/{len(paths)}')
    if pending:
        say('Loading cached MS-CLAP audio tower...')
        encoder=AudioEmbedder()
        say('Encoder ready; beginning two-pass loudness processing')
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        start=time.monotonic()
        audit=(RUN/'loudness_audit.jsonl').open('a',encoding='utf-8',buffering=1)
        with tempfile.TemporaryDirectory(prefix='music_16k_',dir=HERE/'cache',ignore_cleanup_errors=True) as tmp, concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for offset in range(0,len(pending),32):
                jobs=[(i,p,Path(tmp)) for i,p in pending[offset:offset+32]]
                batch=list(pool.map(process,jobs))
                for j in range(0,len(batch),8):
                    group=batch[j:j+8]
                    waves=np.stack([r[1] for r in group])
                    with torch.inference_mode(): vec,_=encoder.model(torch.from_numpy(waves).cuda())
                    vec=unit_rows(vec.cpu().numpy())
                    for row,v in zip(group,vec):
                        saved[row[0]]=v; audit.write(json.dumps(row[2],ensure_ascii=False)+'\n')
                ordered=[p for p in paths if p in saved]
                write_vectors(RAW,ordered,np.stack([saved[p] for p in ordered]))
                say(f'cached={len(saved)}/{len(paths)}; elapsed_min={(time.monotonic()-start)/60:.1f}')
        audit.close(); del encoder
        import gc
        gc.collect(); torch.cuda.empty_cache()
    p,v=read_vectors(RAW)
    if set(p)!=set(paths): raise ValueError('Incomplete feature cache')
    split=json.loads((ROOT/'models_archive/ar_preference/20261030_128k_liked_files_v1/split.json').read_text(encoding='utf-8'))
    train_names=set(split['personal']['train'])|set(split['background']['background_train'])
    train=np.array([i for i,path in enumerate(p) if Path(path).name in train_names])
    if len(train)!=1734: raise ValueError('PCA training identity mismatch')
    pca=PCA(n_components=256,svd_solver='full')
    pca.fit(v[train]); projected=pca.transform(v).astype(np.float32)
    # No whitening: retain leading variance directions, normalize for cosine KNN.
    reduced=unit_rows(projected)
    reduced_id=ID+'-trainonly-PCA256-L2-v1'
    np.savez_compressed(REDUCED,paths=np.asarray(p),vectors=reduced,model_id='msclap:2023',preprocess_id=reduced_id)
    joblib.dump({'pca':pca,'normalization':'L2 after projection','train_files':[p[i] for i in train],
                 'raw_preprocess_id':ID,'preprocess_id':reduced_id},RUN/'projection256.joblib')
    write=(lambda path,obj:path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8'))
    write(RUN/'preprocessing.json',{'raw_cache':str(RAW),'cache_256':str(REDUCED),'preprocess_id':reduced_id,
         'dimension':256,'variance_retained':float(pca.explained_variance_ratio_.sum()),'pca_fit_train_only':len(train),
         'audio_effective_sr':16000,'encoder_interface_sr':44100,'loudness_I':-16,'loudness_TP':-1.5,'loudness_LRA':11,
         'source_manifest':str(HERE/'cache/library_msclap_128k.npz'),'source_audio':'original files; no new lossy MP3 encoding',
         'encoder_tf32':False,'window':'central7 seconds; short audio repeated','raw_dimension':1024})
    say(f'PCA256 complete; retained_variance={pca.explained_variance_ratio_.sum():.6f}')
    env=os.environ.copy(); env.update(MSCLAP_PREPROCESS_ID=reduced_id,MSCLAP_VECTOR_DIM='256',MSCLAP_CACHE=str(REDUCED),MUSIC_EXPERIMENT_NO_LATEST='1')
    dirs={n:ROOT/'models_archive'/n/STAMP for n in ('knn_preference','replay_net','random_forest_replay','diffusion_preference','conditional_diffusion','diffusion_background','ar_preference')}
    commands=[('knn_preference',['preference_knn.py','train','--cache',str(REDUCED)]),
              ('replay_net',['train_replay_model.py','--epochs','120','--batch-size','256']),
              ('random_forest_replay',['train_random_forest.py']),
              ('diffusion_preference',['train_diffusion_preference.py','--latent-dim','256','--epochs','120','--batch-size','256']),
              ('conditional_diffusion',['train_conditional_diffusion.py','--epochs','360','--patience','360','--batch-size','256']),
              ('diffusion_background',['evaluate_diffusion_background.py','--personal',str(dirs['conditional_diffusion']),'--epochs','360','--batch-size','256','--repeats','32']),
              ('ar_preference',['train_ar_preference.py','--split-experiment',str(dirs['diffusion_background']),'--dimension','256','--chunks','16','--batch-size','256'])]
    import sys
    for name,command in commands:
        directory=dirs[name]
        if (directory/'metrics.json').exists() and ((directory/'relative_ranking.csv').exists() or name in ('knn_preference','replay_net','random_forest_replay','diffusion_preference','conditional_diffusion')):
            say('REUSING completed '+name); continue
        if directory.exists(): raise RuntimeError(f'Incomplete model folder requires inspection: {directory}')
        if name in ('replay_net','random_forest_replay','diffusion_preference','conditional_diffusion'): command+=['--model',str(dirs['knn_preference'])]
        command+=['--out',str(directory)]
        say('TRAIN '+name)
        with (RUN/(name+'_console.log')).open('w',encoding='utf-8') as output:
            subprocess.run([sys.executable,*command],cwd=HERE,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
        say('FINISHED '+name)
    write(RUN/'models.json',{k:str(v) for k,v in dirs.items()})
    (RUN/'COMPLETE').write_text('All vectors and seven models completed',encoding='utf-8')
    say('COMPLETE: '+str(RUN))


if __name__=='__main__': main()
