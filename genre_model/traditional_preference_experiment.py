"""Traditional positive/unlabeled preference experiments on frozen features.

No neural network is trained. Model/config selection uses validation proxy AUC
only. Random forest predicts replay intensity and is reported separately.
"""
import csv
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest,RandomForestRegressor
from sklearn.metrics import roc_auc_score,mean_absolute_error
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from threadpoolctl import threadpool_limits
from preference_data import write_json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RUN=ROOT/'models_archive/traditional_preference/20261031_16k_v2'

def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def table(p,r):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)

def kernel_score(sim,weights,tau,k=None):
    if k is not None:
        ids=np.argpartition(sim,-k,axis=1)[:,-k:]
        values=np.take_along_axis(sim,ids,1);w=weights[ids]
        # Preserve the original personal frequency support definition.
        return logsumexp((values-1)/tau+np.log(w),axis=1)-np.log(k)
    valid=np.isfinite(sim)
    return logsumexp((sim-1)/tau+np.log(weights)[None],axis=1)-np.log(np.maximum(valid@weights,1e-30))

def main():
    RUN.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();logfile=(RUN/'training.log').open('w',encoding='utf-8',buffering=1)
    def log(s):print(s,flush=True);logfile.write(s+'\n')
    with np.load(HERE/'cache/library_msclap_16k_lufs16.npz') as d:
        paths=list(map(str,d['paths']));raw=unit(d['vectors'].astype(np.float32))
    names=[Path(p).name for p in paths];by_name={n:i for i,n in enumerate(names)}
    with np.load(HERE/'cache/four_queries_16k_lufs16_native1024.npz') as d:
        query=unit(d['vectors'].astype(np.float32));query_paths=list(map(str,d['paths']))
    joined=np.vstack([raw,query]);n=len(raw)
    baseline=ROOT/'models_archive/ar_preference/20261031_16k_lufs16_pca192_sweep_v1'
    split=load(baseline/'split.json');membership={r['file']:r for r in rows(baseline/'relative_ranking.csv')}
    personal={s:np.array([by_name[name] for name in split['personal'][s]]) for s in ['train','validation','test']}
    bg={s:np.array([by_name[name] for name in split['background']['background_'+s]]) for s in ['train','validation','test']}
    fit=np.r_[personal['train'],bg['train']]
    assert len(set(fit))==1734
    favorites=load(ROOT/'models_archive/knn_preference/20261031_16k_lufs16_native1024_v1/songs.json')
    liked={Path(s.strip().strip('"')).name for s in (ROOT/'ncm-playback-dl/tmp/liked-files.txt').read_text(encoding='utf-8-sig').splitlines() if s.strip()}
    assert {r['labels']['文件'] for r in favorites}==liked==set(split['personal']['train']+split['personal']['validation']+split['personal']['test'])
    weights=np.ones(n,np.float64);counts=np.full(n,np.nan)
    for r in favorites:
        b=r['behavior'];i=by_name[r['labels']['文件']]
        weights[i]=.5+float(b['in_liked_playlist'])+(np.log1p(b['play_count']**2) if b['play_count_known'] else 0.)
        assert np.isclose(weights[i],b['weight'])
        if b['play_count_known']:counts[i]=b['play_count']
    condition_source=ROOT/'models_archive/ar_tagged/20261031_16k_pca192_v2'
    with np.load(condition_source/'conditions.npz') as d:
        assert list(map(str,d['paths']))==paths
        conditions=np.vstack([d['conditions'],d['query_conditions']])
    cs=StandardScaler().fit(conditions[fit]);tagvectors=unit(cs.transform(conditions).astype(np.float32))
    val=np.r_[personal['validation'],bg['validation']];vy=np.r_[np.ones(len(personal['validation'])),np.zeros(len(bg['validation']))]
    test=np.r_[personal['test'],bg['test']];ty=np.r_[np.ones(len(personal['test'])),np.zeros(len(bg['test']))]
    ref_p=personal['train'];ref_b=bg['train'];wp=weights[ref_p];wb=np.ones(len(ref_b))
    # Shared recording-window exclusions; score existing songs without exact self retrieval.
    exact_p=joined@raw[ref_p].T>.995;exact_b=joined@raw[ref_b].T>.995
    basis=joblib.load(ROOT/'models_archive/ar_pca_sweep/20261031_16k_lufs16_v1/shared_pca512.joblib')
    projected=basis.transform(joined)
    best={};search=[];replay_search=[];best_replay=None
    def candidate(method,d,view,params,scores,artifact):
        auc=float(roc_auc_score(vy,scores[val]))
        search.append(dict(method=method,dimension=d,view=view,parameters=json.dumps(params),validation_auc=auc))
        log(f'{method} dim={d} view={view} {params} validation_auc={auc:.4f}')
        if method not in best or auc>best[method]['validation_auc']:
            best[method]=dict(method=method,dimension=d,view=view,parameters=params,validation_auc=auc,scores=scores.copy(),artifact=artifact)
    for dimension in [64,128,192,256,512,1024]:
        audio=unit(projected[:,:dimension].astype(np.float32)) if dimension<1024 else joined
        for view in ['audio','audio_tags']:
            z=audio if view=='audio' else unit(np.concatenate([np.sqrt(.75)*audio,np.sqrt(.25)*tagvectors],axis=1))
            sp=z@z[ref_p].T;sb=z@z[ref_b].T;sp[exact_p]=-np.inf;sb[exact_b]=-np.inf
            for tau in [.05,.1,.2]:
                ps=kernel_score(sp,wp,tau);bs=kernel_score(sb,wb,tau)
                artifact=dict(reference_personal=z[ref_p],reference_background=z[ref_b],personal_weights=wp)
                candidate('weighted_kde',dimension,view,dict(tau=tau),ps,artifact)
                candidate('relative_kde',dimension,view,dict(tau=tau),ps-bs,artifact)
                for k in [5,20,40]:
                    pk=kernel_score(sp,wp,tau,k);bk=kernel_score(sb,wb,tau,k)
                    candidate('weighted_knn',dimension,view,dict(tau=tau,k=k),pk,artifact)
                    candidate('relative_knn',dimension,view,dict(tau=tau,k=k),pk-bk,artifact)
            for clusters in [3,6,12]:
                km=KMeans(n_clusters=clusters,n_init=3,max_iter=100,random_state=2026).fit(z[ref_p],sample_weight=wp)
                centers=unit(km.cluster_centers_);sim=z@centers.T
                mass=np.bincount(km.labels_,weights=wp,minlength=clusters);mass/=mass.sum()
                for mode in ['max','lse']:
                    score=sim.max(1) if mode=='max' else .1*logsumexp(sim/.1+np.log(mass)[None],axis=1)
                    candidate('multi_interest',dimension,view,dict(clusters=clusters,mode=mode,tau=.1),score,dict(kmeans=km,centers=centers,mass=mass))
            scaler=StandardScaler().fit(z[fit]);scaled=scaler.transform(z).astype(np.float32)
            # Direct sample weights for one-class models. None receives negative labels.
            for nu in [.05,.15]:
                svm=OneClassSVM(nu=nu,gamma='scale').fit(scaled[ref_p],sample_weight=wp/wp.mean())
                candidate('one_class_svm',dimension,view,dict(nu=nu,gamma='scale'),svm.score_samples(scaled),dict(estimator=svm,scaler=scaler))
            iso=IsolationForest(n_estimators=200,max_samples=min(256,len(ref_p)),random_state=2026,n_jobs=4).fit(scaled[ref_p],sample_weight=wp)
            candidate('isolation_forest',dimension,view,dict(trees=200),iso.score_samples(scaled),dict(estimator=iso,scaler=scaler))
            # Regularized diagonal mixtures; weighted bootstrap approximates sample weights.
            # Keep the GMM sweep in <=256 audio dimensions to limit small-sample freedom.
            if dimension<=256:
                sampled=np.random.default_rng(2026).choice(ref_p,size=4*len(ref_p),p=wp/wp.sum())
                for components in [1,4,8]:
                    gm=GaussianMixture(n_components=components,covariance_type='diag',reg_covar=.1,max_iter=100,n_init=1,random_state=2026).fit(scaled[sampled])
                    gb=GaussianMixture(n_components=components,covariance_type='diag',reg_covar=.1,max_iter=100,n_init=1,random_state=2026).fit(scaled[ref_b])
                    if not (gm.converged_ and gb.converged_):
                        log(f'SKIP unconverged GMM dimension={dimension} view={view} components={components}')
                        continue
                    candidate('relative_gmm',dimension,view,dict(components=components,covariance='diag',reg=.1,weighted_bootstrap_factor=4),gm.score_samples(scaled)-gb.score_samples(scaled),dict(personal=gm,background=gb,scaler=scaler,sampled_indices=sampled))
            if dimension in [192,1024]:
                tr=ref_p[np.isfinite(counts[ref_p])];va=personal['validation'][np.isfinite(counts[personal['validation']])]
                for leaf in [5,10]:
                    forest=RandomForestRegressor(n_estimators=250,min_samples_leaf=leaf,max_features=.3,max_depth=None,random_state=2026,n_jobs=4)
                    forest.fit(z[tr],np.log1p(counts[tr]))
                    pred=forest.predict(z);mae=float(mean_absolute_error(np.log1p(counts[va]),pred[va]))
                    replay_search.append(dict(dimension=dimension,view=view,min_samples_leaf=leaf,validation_mae=mae))
                    if best_replay is None or mae<best_replay['validation_mae']:
                        best_replay=dict(dimension=dimension,view=view,leaf=leaf,validation_mae=mae,predictions=pred,forest=forest,train=tr,validation=va)
            log(f'FINISHED dimension={dimension} view={view}; elapsed_min={(time.monotonic()-started)/60:.1f}')
    table(RUN/'validation_search.csv',search);table(RUN/'forest_validation.csv',replay_search)
    summary=[];four=[];top={}
    for method,chosen in best.items():
        out=RUN/method;out.mkdir()
        score=chosen['scores'];reference=np.sort(score[personal['validation']]);pct=np.searchsorted(reference,score,side='right')*100/len(reference)
        test_auc=float(roc_auc_score(ty,score[test]))
        record={k:chosen[k] for k in ['method','dimension','view','parameters','validation_auc']};record['test_auc']=test_auc
        summary.append(record);write_json(out/'metrics.json',record)
        joblib.dump(dict(**chosen['artifact'],configuration=record,condition_source=str(condition_source),tag_scaler=cs),out/'model.joblib',compress=3)
        ranked=[]
        for rank,i in enumerate(np.argsort(-score[:n],kind='stable'),1):
            m=membership[names[i]]
            ranked.append(dict(rank=rank,match_percentile=float(pct[i]),raw_score=float(score[i]),favorite_split=m['favorite_split'],background_split=m['background_split'],file=names[i],audio_path=paths[i]))
        assert len(ranked)==len({r['file'] for r in ranked})==4277
        table(out/'library_ranking.csv',ranked)
        candidates=[dict(r) for r in ranked if r['background_split']=='background_test']
        for rank,r in enumerate(candidates,1):r['rank']=rank
        assert len(candidates)==1845;table(out/'candidate_recommendations_heldout.csv',candidates)
        top[method]=candidates[:20]
        for j,path in enumerate(query_paths):four.append(dict(method=method,file=Path(path).name,percentile=float(pct[n+j]),raw_score=float(score[n+j])))
    rf=best_replay;te=personal['test'][np.isfinite(counts[personal['test']])]
    replay=dict(method='random_forest_replay',dimension=rf['dimension'],view=rf['view'],min_samples_leaf=rf['leaf'],validation_mae=rf['validation_mae'],test_mae=float(mean_absolute_error(np.log1p(counts[te]),rf['predictions'][te])),median_baseline_mae=float(mean_absolute_error(np.log1p(counts[te]),np.full(len(te),np.median(np.log1p(counts[rf['train']]))))),test_spearman=float(spearmanr(np.log1p(counts[te]),rf['predictions'][te]).statistic),test_count=len(te),target='log1p(playCount); known counts only, not taste probability')
    out=RUN/'random_forest_replay';out.mkdir();write_json(out/'metrics.json',replay);joblib.dump(rf['forest'],out/'model.joblib',compress=3)
    pred=rf['predictions'];reference=np.sort(pred[rf['validation']]);pct=np.searchsorted(reference,pred,side='right')*100/len(reference)
    ranked=[]
    for rank,i in enumerate(np.argsort(-pred[:n],kind='stable'),1):
        m=membership[names[i]]
        ranked.append(dict(rank=rank,replay_percentile=float(pct[i]),predicted_log1p_plays=float(pred[i]),favorite_split=m['favorite_split'],background_split=m['background_split'],file=names[i],audio_path=paths[i]))
    table(out/'library_ranking.csv',ranked);candidates=[dict(r) for r in ranked if r['background_split']=='background_test']
    for rank,r in enumerate(candidates,1):r['rank']=rank
    table(out/'candidate_recommendations_heldout.csv',candidates);top['random_forest_replay']=candidates[:20]
    for j,path in enumerate(query_paths):four.append(dict(method='random_forest_replay',file=Path(path).name,percentile=float(pct[n+j]),raw_score=float(pred[n+j])))
    for r in summary:r['parameters']=json.dumps(r['parameters'])
    table(RUN/'comparison.csv',summary);table(RUN/'four_song_scores.csv',four)
    write_json(RUN/'results.json',dict(comparison=summary,replay=replay,four_songs=four,candidate_top20=top))
    joblib.dump(dict(pca=basis,tag_scaler=cs,condition_source=str(condition_source),raw_cache=str(HERE/'cache/library_msclap_16k_lufs16.npz')),RUN/'feature_pipeline.joblib',compress=3)
    write_json(RUN/'split.json',split)
    report=['# 传统机器学习口味实验','','冻结现有16k响度对齐MS-CLAP特征，不训练神经网络。测试PCA64/128/192/256/512和原生1024维，以及音频、音频＋标签两种视图。标签视图使用已审计训练集近邻估计条件（25%余弦特征权重），不直接读取待测曲真实标签。每个方法只按验证集AUC选维数及超参数，然后报告同373首收藏＋1845首未标记曲的测试代理AUC。','', '| 方法 | 维数 | 特征 | 验证AUC | 测试AUC | 散花 | Myth | 爱如火 | 苦巧克力 |','|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(summary,key=lambda r:-r['validation_auc']):
        s=[q['percentile'] for q in four if q['method']==r['method']]
        report.append(f"| {r['method']} | {r['dimension']} | {r['view']} | {r['validation_auc']:.4f} | {r['test_auc']:.4f} | "+' | '.join(f'{v:.1f}' for v in s)+' |')
    report+=['',f"随机森林预测播放强度：测试MAE={replay['test_mae']:.4f}，中位数基线={replay['median_baseline_mae']:.4f}，Spearman={replay['test_spearman']:.4f}，样本={replay['test_count']}。与口味AUC任务不同。",'', '权重沿用0.5+L+ln(1+m²)，未知次数不加对数项。KNN/KDE与聚类直接使用权重；GMM使用4倍样本量的加权重采样近似权重；OneClassSVM和IsolationForest使用sample_weight。GMM仅扫描<=256维、对角协方差并加正则。随机森林以已知播放次数为目标，均匀训练。','', '全库排名含训练歌曲。未标记不等于不喜欢，百分位不是喜欢概率；各模型分数不宜直接平均。单随机种子，已多次查看的留出集只作实验对照。']
    (RUN/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
    labels=[r['method'] for r in summary];ix=np.arange(len(labels))
    ax.bar(ix-.18,[r['validation_auc'] for r in summary],.36,label='Validation')
    ax.bar(ix+.18,[r['test_auc'] for r in summary],.36,label='Held-out proxy')
    ax.set(xticks=ix,xticklabels=labels,ylabel='Proxy AUC',title='Traditional preference models');ax.tick_params(axis='x',rotation=25);ax.legend()
    fig.savefig(RUN/'model_comparison.png',dpi=160);plt.close(fig)
    (RUN/'COMPLETE').write_text('Traditional methods evaluated; all library rankings exported',encoding='utf-8')
    log((RUN/'REPORT.md').read_text(encoding='utf-8'));logfile.close()

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
