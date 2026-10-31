"""Reconstruct per-dimension validation-selected KNN/KDE results and rankings."""
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from traditional_preference_experiment import HERE,ROOT,RUN,load,rows,table,unit,kernel_score

METHODS=['weighted_knn','relative_knn','weighted_kde','relative_kde']
DIMENSIONS=[64,128,192,256,512,1024]
OUT=RUN/'knn_kde_dimensions'

def main():
    OUT.mkdir(exist_ok=False)
    grid=rows(RUN/'validation_search.csv')
    split=load(RUN/'split.json')
    with np.load(HERE/'cache/library_msclap_16k_lufs16.npz') as d:
        paths=list(map(str,d['paths']));raw=unit(d['vectors'].astype(np.float32))
    names=[Path(p).name for p in paths];index={n:i for i,n in enumerate(names)};n=len(paths)
    with np.load(HERE/'cache/four_queries_16k_lufs16_native1024.npz') as d:
        query=unit(d['vectors'].astype(np.float32));qpaths=list(map(str,d['paths']))
    joined=np.vstack([raw,query])
    personal={s:np.array([index[n] for n in split['personal'][s]]) for s in ['train','validation','test']}
    bg={s:np.array([index[n] for n in split['background']['background_'+s]]) for s in ['train','validation','test']}
    fit=np.r_[personal['train'],bg['train']]
    val=np.r_[personal['validation'],bg['validation']];vy=np.r_[np.ones(len(personal['validation'])),np.zeros(len(bg['validation']))]
    test=np.r_[personal['test'],bg['test']];ty=np.r_[np.ones(len(personal['test'])),np.zeros(len(bg['test']))]
    behavior={r['labels']['文件']:r['behavior']['weight'] for r in load(ROOT/'models_archive/knn_preference/20261031_16k_lufs16_native1024_v1/songs.json')}
    wp=np.array([behavior[names[i]] for i in personal['train']])
    wb=np.ones(len(bg['train']))
    with np.load(ROOT/'models_archive/ar_tagged/20261031_16k_pca192_v2/conditions.npz') as d:
        assert list(map(str,d['paths']))==paths
        c=np.vstack([d['conditions'],d['query_conditions']])
    tag=unit(StandardScaler().fit(c[fit]).transform(c).astype(np.float32))
    basis=joblib.load(ROOT/'models_archive/ar_pca_sweep/20261031_16k_lufs16_v1/shared_pca512.joblib')
    projected=basis.transform(joined)
    mask_p=joined@raw[personal['train']].T>.995;mask_b=joined@raw[bg['train']].T>.995
    source=rows(ROOT/'models_archive/ar_preference/20261031_16k_lufs16_pca192_sweep_v1/relative_ranking.csv')
    membership={r['file']:r for r in source}
    results=[];four=[];fixed=[]
    for dimension in DIMENSIONS:
        audio=unit(projected[:,:dimension].astype(np.float32)) if dimension<1024 else joined
        matrices={}
        for view in ['audio','audio_tags']:
            z=audio if view=='audio' else unit(np.concatenate([np.sqrt(.75)*audio,np.sqrt(.25)*tag],axis=1))
            sp=z@z[personal['train']].T;sb=z@z[bg['train']].T;sp[mask_p]=-np.inf;sb[mask_b]=-np.inf
            matrices[view]=(sp,sb)
        for method in METHODS:
            candidates=[r for r in grid if r['method']==method and int(r['dimension'])==dimension]
            chosen=max(candidates,key=lambda r:float(r['validation_auc']))
            params=json.loads(chosen['parameters']);tau=params['tau'];k=params.get('k')
            sp,sb=matrices[chosen['view']]
            score=kernel_score(sp,wp,tau,k)
            if method.startswith('relative'):score-=kernel_score(sb,wb,tau,k)
            va=float(roc_auc_score(vy,score[val]));te=float(roc_auc_score(ty,score[test]))
            assert abs(va-float(chosen['validation_auc']))<1e-12
            record=dict(method=method,dimension=dimension,view=chosen['view'],parameters=chosen['parameters'],validation_auc=va,test_auc=te)
            results.append(record)
            ref=np.sort(score[personal['validation']]);pct=np.searchsorted(ref,score,side='right')*100/len(ref)
            for j,path in enumerate(qpaths):four.append(dict(method=method,dimension=dimension,file=Path(path).name,percentile=float(pct[n+j]),raw_score=float(score[n+j])))
            out=OUT/f'{method}_{dimension}';out.mkdir()
            ranking=[]
            for rank,i in enumerate(np.argsort(-score[:n],kind='stable'),1):
                m=membership[names[i]]
                ranking.append(dict(rank=rank,match_percentile=float(pct[i]),raw_score=float(score[i]),favorite_split=m['favorite_split'],background_split=m['background_split'],file=names[i],audio_path=paths[i]))
            assert len(ranking)==len({r['file'] for r in ranking})==4277
            table(out/'library_ranking.csv',ranking)
            candidates=[dict(r) for r in ranking if r['background_split']=='background_test']
            for rank,r in enumerate(candidates,1):r['rank']=rank
            assert len(candidates)==1845;table(out/'candidate_recommendations_heldout.csv',candidates)
        sp,sb=matrices['audio']
        for method,tau,k in [('relative_knn',.2,40),('relative_kde',.1,None)]:
            score=kernel_score(sp,wp,tau,k)-kernel_score(sb,wb,tau,k)
            fixed.append(dict(method=method,dimension=dimension,tau=tau,k=k,validation_auc=float(roc_auc_score(vy,score[val])),test_auc=float(roc_auc_score(ty,score[test]))))
    old=load(RUN/'results.json')['comparison']
    for r in old:
        if r['method'] in METHODS:
            found=next(q for q in results if q['method']==r['method'] and q['dimension']==r['dimension'])
            assert abs(found['test_auc']-r['test_auc'])<1e-12
    table(OUT/'comparison.csv',results);table(OUT/'four_song_scores.csv',four);table(OUT/'fixed_audio_parameters.csv',fixed)
    (OUT/'results.json').write_text(json.dumps(dict(comparison=results,four_songs=four,fixed_parameters=fixed),ensure_ascii=False,indent=2),encoding='utf-8')
    report=['# KNN/KDE：64至1024维','','每个方法、每个维数内，仅按验证集选音频/标签视图、k和tau；测试集完全相同，为373首收藏＋1845首未标记曲。64至512为训练集PCA；1024为原生坐标。表中差异包括超参数与视图变化。','','| 维数 | 加权KNN测试AUC | 相对KNN测试AUC | 加权KDE测试AUC | 相对KDE测试AUC |','|---|---:|---:|---:|---:|']
    for d in DIMENSIONS:
        report.append('| '+str(d)+' | '+' | '.join(f'{next(r["test_auc"] for r in results if r["dimension"]==d and r["method"]==m):.4f}' for m in METHODS)+' |')
    report+=['','## 相同参数、纯音频对照','','固定相对KNN k=40、tau=0.2；相对KDE tau=0.1。','','| 维数 | 相对KNN测试AUC | 相对KDE测试AUC |','|---|---:|---:|']
    for d in DIMENSIONS:report.append('| '+str(d)+' | '+' | '.join(f'{next(r["test_auc"] for r in fixed if r["dimension"]==d and r["method"]==m):.4f}' for m in ['relative_knn','relative_kde'])+' |')
    for method in METHODS:
        report+=['','## '+method+'：四曲参考百分位','','| 维数 | 散花 | Myth compiler | 爱如火 | 苦巧克力 |','|---|---:|---:|---:|---:|']
        for d in DIMENSIONS:report.append('| '+str(d)+' | '+' | '.join(f'{r["percentile"]:.1f}' for r in four if r['method']==method and r['dimension']==d)+' |')
    report+=['','完整参数与视图见comparison.csv；四曲包含训练歌曲，百分位不是喜欢概率。所有24组完整榜和候选榜均保留。']
    (OUT/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    for method in METHODS:ax.plot(DIMENSIONS,[next(r['test_auc'] for r in results if r['method']==method and r['dimension']==d) for d in DIMENSIONS],marker='o',label=method)
    ax.set(xlabel='Audio dimensions',ylabel='Held-out proxy AUC',title='KNN/KDE by dimension',xticks=DIMENSIONS);ax.legend();ax.grid(alpha=.2)
    fig.savefig(OUT/'auc_by_dimension.png',dpi=160);plt.close(fig)
    (OUT/'COMPLETE').write_text('24 configurations reconstructed, validation scores and previous winners verified',encoding='utf-8')
    print((OUT/'REPORT.md').read_text(encoding='utf-8'),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
