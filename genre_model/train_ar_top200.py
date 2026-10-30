"""Top-200 behavior-weight favorite dataset; native1024 personal/background AR."""
from pathlib import Path
import os,sys,json,csv,subprocess,argparse
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
STAMP='20261031_16k_native1024_weight_top200_v2'
RUN=ROOT/'models_archive/ar_top200'/STAMP
MODEL=ROOT/'models_archive/ar_preference'/STAMP
SOURCE=ROOT/'models_archive/knn_preference/20261031_16k_lufs16_native1024_v1'
BASE=ROOT/'models_archive/ar_preference/20261031_16k_lufs16_native1024_v1'
CACHE=HERE/'cache/library_msclap_16k_lufs16.npz'
ID='msclap2023-16kmono-loudnorm-I16-TP1.5-LRA11-2pass-center7-resample44100-fp32-v1'

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    global RUN, MODEL, SOURCE, BASE, CACHE, ID
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--knn-model',type=Path,required=True)
    parser.add_argument('--background-ar',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    SOURCE=args.knn_model.resolve(); BASE=args.background_ar.resolve()
    CACHE=args.cache.resolve(); RUN=args.out.resolve(); MODEL=RUN/'model'
    config=read(SOURCE/'config.json')
    if config['dimension']!=1024: raise ValueError('Top200 native AR requires 1024 dimensions')
    ID=config['preprocess_id']
    RUN.mkdir(parents=True,exist_ok=False)
    source=read(SOURCE/'songs.json')
    liked={Path(s.strip().strip('"')).name for s in (ROOT/'ncm-playback-dl/tmp/liked-files.txt').read_text(encoding='utf-8-sig').splitlines() if s.strip()}
    assert {r['labels']['文件'] for r in source}==liked
    selected=sorted(source,key=lambda r:(-r['behavior']['weight'],r['labels']['文件'].casefold()))[:200]
    names=[r['labels']['文件'] for r in selected]
    groups_source=read(SOURCE/'splits.json')
    group_map=dict(zip(groups_source['files'],groups_source['groups']))
    groups=np.array([group_map[n] for n in names]); ids=np.arange(200)
    # Balance by counts only; no labels, vectors or model performance enter
    # split selection. Artist/recording groups remain intact.
    candidates=GroupShuffleSplit(n_splits=200,test_size=.2,random_state=42).split(ids,groups=groups)
    dev,test=min(candidates,key=lambda parts:abs(len(parts[1])-40))
    candidates=GroupShuffleSplit(n_splits=200,test_size=.2,random_state=43).split(dev,groups=groups[dev])
    tr,va=min(candidates,key=lambda parts:abs(len(parts[1])-32))
    train,val=dev[tr],dev[va]
    assert not(set(groups[train])&set(groups[val]) or set(groups[train])&set(groups[test]) or set(groups[val])&set(groups[test]))
    personal={key:[names[i] for i in part] for key,part in (('train',train),('validation',val),('test',test))}
    personal['other_favorites']=sorted(liked-set(names))
    split_dir=RUN/'data_split'; write(split_dir/'personal/split.json',personal)
    baseline=read(BASE/'split.json')['background']; rng=np.random.default_rng(2026)
    background={key:rng.permutation(baseline[key])[:count].tolist() for key,count in
                (('background_train',len(train)),('background_validation',len(val)))}
    background.update(personal_model=str(split_dir/'personal'),background_test=baseline['background_test'],
                      excluded_near_favorite=baseline.get('excluded_near_favorite',[]))
    assert not(set(background['background_train'])&liked)
    write(split_dir/'background/split.json',background)
    membership={n:part for part in ('train','validation','test') for n in personal[part]}
    with (RUN/'top200_weights.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f); w.writerow(['weight_rank','file','weight','play_count','play_count_known','in_liked_playlist','split'])
        for rank,row in enumerate(selected,1):
            b=row['behavior']; w.writerow([rank,row['labels']['文件'],b['weight'],b['play_count'],b['play_count_known'],b['in_liked_playlist'],membership[row['labels']['文件']]])
    write(RUN/'selection.json',{'source_model':str(SOURCE),'selected_count':200,'weight_formula':'0.5 + L + ln(1+m^2) if known; unknown keeps collection terms',
         'weight_usage':'Selection only; uniform training within selected train songs',
         'train':len(train),'validation':len(val),'test':len(test),'background_train':len(train),'background_validation':len(val),
         'weight_max':selected[0]['behavior']['weight'],'weight_min':selected[-1]['behavior']['weight'],'outside_favorites_excluded_from_background':len(liked)-200,
         'no_projection':True,'dimension':1024,'grouping':'original artist/duplicate groups; no group overlap across personal partitions'})
    env=os.environ.copy(); env.update(PYTHONIOENCODING='utf-8',MSCLAP_VECTOR_DIM='1024',MSCLAP_CACHE=str(CACHE),MSCLAP_PREPROCESS_ID=ID,MUSIC_EXPERIMENT_NO_LATEST='1')
    command=[sys.executable,'train_ar_preference.py','--split-experiment',str(split_dir/'background'),'--out',str(MODEL),
             '--no-projection','--dimension','1024','--chunks','64','--mixtures','4','--epochs','360','--patience','40','--batch-size','256','--lr','0.0003']
    print(f'Top200 selected: train={len(train)}, val={len(val)}, test={len(test)}; matched background sizes; epochs up to360',flush=True)
    with (RUN/'console.log').open('w',encoding='utf-8') as log:
        subprocess.run(command,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    metrics=read(MODEL/'metrics.json'); metrics['favorite_definition']='Top 200 of liked-files favorites by saved behavior weight'; metrics['selection_audit']=str(RUN/'selection.json'); write(MODEL/'metrics.json',metrics)
    report=['# 权重前200首相对AR模型','',f"个人数据集200首，按歌手/同录音分组：训练 {len(train)}，验证 {len(val)}，测试 {len(test)}。背景训练/验证与个人数量匹配，背景测试保持原独立留出集。其余1201首喜欢曲未用作背景或负例。",'',
            '采用16k响度对齐后的原生1024维，无PCA。64块×16维、2层Transformer、4头注意力、4分量GMM。权重仅用于选出前200名，入选后沿用均匀采样。初始学习率0.0003，batch上限256；最多360轮，patience40，保存最佳验证权重。','',
            f"留出Top200喜欢曲对未标记曲代理AUC：{metrics['proxy_auc_relative_ar']:.4f}。由于正样本集合改变，不能与全收藏模型的旧AUC直接比较。分数为Top200验证歌曲参考百分位，不是喜欢概率。",'',
            f'模型、完整榜和曲线目录：{MODEL}',f'选择清单：{RUN / "top200_weights.csv"}',
            '完整榜中 favorite_outside_top200 表示明确喜欢但不在前200的歌曲，不能视作不喜欢。']
    (RUN/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    write(RUN/'model.json',{'model_dir':str(MODEL)})
    (RUN/'COMPLETE').write_text('Top200 model trained',encoding='utf-8')
    print(json.dumps(metrics,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
