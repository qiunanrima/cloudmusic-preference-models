"""Controlled PCA64/128/256/512 relative AR comparison on aligned audio."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RUN=ROOT/'models_archive/ar_pca_sweep/20261031_16k_lufs16_v1'
BASE='20261030_16k_lufs16_pca256_v1'

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def save_table(path, records):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimensions',nargs='+',type=int,default=[64,128,256,512])
    args=parser.parse_args()
    if any(d<16 or d>512 or d%16 for d in args.dimensions):
        parser.error('Dimensions must be multiples of 16 between 16 and 512')
    RUN.mkdir(parents=True,exist_ok=True)
    with np.load(HERE/'cache/library_msclap_16k_lufs16.npz') as data:
        paths=data['paths']; raw=data['vectors'].astype(np.float32); raw_id=str(data['preprocess_id'])
    split=load(ROOT/'models_archive/ar_preference'/BASE/'split.json')
    train_names=set(split['personal']['train'])|set(split['background']['background_train'])
    train=np.array([i for i,p in enumerate(paths) if Path(p).name in train_names])
    assert len(train)==1734
    basis_path=RUN/'shared_pca512.joblib'
    if not basis_path.exists():
        basis=PCA(n_components=512,svd_solver='full').fit(raw[train])
        joblib.dump(basis,basis_path)
    basis=joblib.load(basis_path)
    projected=basis.transform(raw)
    models={}
    for dimension in args.dimensions:
        out=ROOT/'models_archive/ar_preference'/f'20261031_16k_lufs16_pca{dimension}_sweep_v1'
        if dimension==256:
            out=ROOT/'models_archive/ar_preference'/BASE
        models[dimension]=out
        reduced=projected[:,:dimension].astype(np.float32)
        reduced/=np.linalg.norm(reduced,axis=1,keepdims=True)
        if dimension==256:
            with np.load(HERE/'cache/library_msclap_16k_lufs16_pca256.npz') as old:
                assert np.array_equal(paths,old['paths'])
                assert np.min(np.sum(reduced*old['vectors'],axis=1)) > .99999
            print('PCA256 matches existing subspace (float32 SVD tolerance): reuse exact cache, projection, model and curves',flush=True)
            continue
        cache=RUN/f'library_pca{dimension}.npz'
        protocol=raw_id+f'-trainonly-PCA{dimension}-L2-sweep-v1'
        np.savez_compressed(cache,paths=paths,vectors=reduced,model_id='msclap:2023',preprocess_id=protocol)
        env=os.environ.copy()
        env.update(PYTHONIOENCODING='utf-8',MSCLAP_PREPROCESS_ID=protocol,MSCLAP_CACHE=str(cache),MSCLAP_VECTOR_DIM=str(dimension),MUSIC_EXPERIMENT_NO_LATEST='1')
        if (out/'metrics.json').exists():
            print(f'REUSE PCA{dimension}',flush=True); continue
        print(f'TRAIN PCA{dimension}: chunks={dimension//16}, batch=256, lr=0.0003',flush=True)
        with (RUN/f'console_pca{dimension}.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'train_ar_preference.py','--split-experiment',str(ROOT/'models_archive/diffusion_background'/BASE),'--out',str(out),'--dimension',str(dimension),'--chunks',str(dimension//16),'--epochs','160','--patience','25','--batch-size','256','--lr','0.0003','--seed','2026'],cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(f'FINISHED PCA{dimension}',flush=True)
    import torch
    from score_ar_preference import load_net
    from train_ar_preference import score
    torch.set_num_threads(4)
    with np.load(HERE/'cache/four_queries_16k_lufs16_native1024.npz') as data:
        query_names=[Path(p).name for p in data['paths']]
        query=basis.transform(data['vectors'])
    metrics=[]; songs=[]; all_rankings={}
    for dimension,out in models.items():
        history=rows(out/'loss.csv')
        features=joblib.load(out/'features.joblib')
        if dimension==256:
            original=joblib.load(ROOT/'models_archive/preprocess_16k_256'/BASE/'projection256.joblib')['pca']
            with np.load(HERE/'cache/four_queries_16k_lufs16_native1024.npz') as data:
                q=original.transform(data['vectors']).astype(np.float32)
        else:
            q=query[:,:dimension].astype(np.float32)
        q/=np.linalg.norm(q,axis=1,keepdims=True)
        z=(features['pca'].transform(q)/features['scale']).astype(np.float32)
        p=load_net(out/'model_personal.pt','cuda'); b=load_net(out/'model_background.pt','cuda')
        gain=score(p,z,'cuda')-score(b,z,'cuda')
        del p,b
        ranking=rows(out/'relative_ranking.csv'); all_rankings[dimension]=ranking
        ref=np.sort([float(r['ar_log_likelihood_ratio']) for r in ranking if r['favorite_split']=='validation'])
        for name,g in zip(query_names,gain):
            songs.append(dict(dimension=dimension,file=name,llr=float(g),percentile=float(np.searchsorted(ref,g,side='right')*100/len(ref))))
        m=load(out/'metrics.json')
        metrics.append(dict(dimension=dimension,chunks=dimension//16,variance_retained=float(basis.explained_variance_ratio_[:dimension].sum()),personal_auc=m['proxy_auc_personal_likelihood'],relative_ar_auc=m['proxy_auc_relative_ar'],validation_auc=None,personal_best_epoch=m['best']['personal']['best_epoch'],background_best_epoch=m['best']['background']['best_epoch'],model_dir=str(out)))
    identities=[{r['file']:(r['favorite_split'],r['background_split']) for r in rr} for rr in all_rankings.values()]
    assert all(identity==identities[0] for identity in identities)
    for metric in metrics:
        rr=all_rankings[metric['dimension']]
        val_pos=[r for r in rr if r['favorite_split']=='validation']
        val_bg=[r for r in rr if r['background_split']=='background_validation']
        metric['validation_auc']=float(roc_auc_score([1]*len(val_pos)+[0]*len(val_bg),[float(r['ar_log_likelihood_ratio']) for r in val_pos+val_bg]))
    save_table(RUN/'comparison.csv',metrics); save_table(RUN/'four_song_scores.csv',songs)
    (RUN/'results.json').write_text(json.dumps({'metrics':metrics,'songs':songs,'split_identical':True,'pca_train_count':len(train),'note':'Single seed. PCA256 reused. No confirmed dislikes; all scores are proxies.'},ensure_ascii=False,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(models),2,figsize=(13,3.5*len(models)),constrained_layout=True,squeeze=False)
    for (dimension,out),(a,b) in zip(models.items(),axes):
        h=rows(out/'loss.csv')
        for name in ('personal','background'):
            part=[r for r in h if r['model']==name]
            epochs=[int(r['epoch']) for r in part]
            for field in ('train_nll','validation_nll'):
                a.plot(epochs,[float(r[field]) for r in part],label=name+' '+field)
            b.step(epochs,[float(r['lr']) for r in part],label=name,where='post')
        a.set(title=f'PCA{dimension} loss',xlabel='Epoch',ylabel='NLL per chunk');a.legend(fontsize=8)
        b.set(title=f'PCA{dimension} LR',xlabel='Epoch',ylabel='Learning rate');b.legend()
    fig.savefig(RUN/'loss_lr.png',dpi=150);plt.close(fig)
    titles=['散花','Myth compiler','爱如火','苦巧克力']
    lines=['# 响度对齐 PCA 维度与相对 AR 对比','','固定同一划分、每块16维、2层Transformer、4高斯分量、batch256、初始LR=3e-4、最多160轮、早停25。PCA仅使用1734首训练歌曲拟合；256维复用完全一致的已有模型。全收藏训练，不是Top200。','', '| PCA维数 | 保留方差 | 验证AUC | 测试AUC | 散花 | Myth compiler | 爱如火 | 苦巧克力 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in metrics:
        s=[r['percentile'] for r in songs if r['dimension']==m['dimension']]
        lines.append(f"| {m['dimension']} | {m['variance_retained']:.3%} | {m['validation_auc']:.4f} | {m['relative_ar_auc']:.4f} | "+' | '.join(f'{v:.1f}' for v in s)+' |')
    lines+=['','AUC仅为收藏与未标记曲的检索代理；百分位不是喜欢概率。四曲含训练歌曲，不作为独立泛化评估。只有一个随机种子；每块维度固定使序列长度随PCA维数变化，差异不能完全归因于丢弃主成分。','','![Loss and LR](loss_lr.png)']
    (RUN/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    (RUN/'COMPLETE').write_text('Evaluated dimensions: '+','.join(map(str,models)),encoding='utf-8')
    print(json.dumps({'metrics':metrics,'songs':songs},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    main()
