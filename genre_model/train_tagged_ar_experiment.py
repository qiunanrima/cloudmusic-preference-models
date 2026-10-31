"""Train-only semantic conditioning and behavior-weighted relative AR ablations.

Observed labels supervise a shared nearest-neighbor semantic estimator. Every
song receives estimated conditions, irrespective of favorite membership or label
availability. Training songs exclude themselves from condition estimation.
"""
import csv
import argparse
import json
import math
from pathlib import Path
import random

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
import torch
from torch import nn
from torch.nn import functional as F

from preference_data import read_json, write_json
from train_conditional_diffusion import prepare_conditions, TAG_FIELDS, NUMERIC_FIELDS
from train_ar_preference import ContinuousAR

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE=ROOT/'models_archive/ar_preference/20261031_16k_lufs16_pca192_sweep_v1'
RUN=ROOT/'models_archive/ar_tagged/20261031_16k_pca192_v2'
DIMENSION=192

def readrows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def table(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

class TaggedAR(ContinuousAR):
    def __init__(self,condition_dim,dimension=192):
        super().__init__(dimension,dimension//16,128,2,4)
        self.condition=nn.Sequential(nn.Linear(condition_dim,128),nn.GELU(),nn.Linear(128,128))

    def log_prob(self,vectors,condition):
        target=vectors.reshape(-1,self.chunks,self.chunk_dim)
        shifted=torch.cat([self.bos.expand(len(target),-1,-1),target[:,:-1]],dim=1)
        hidden=self.input(shifted)+self.position+self.condition(condition)[:,None]
        causal=torch.triu(torch.ones(self.chunks,self.chunks,device=vectors.device,dtype=torch.bool),diagonal=1)
        params=self.output(self.transformer(hidden,mask=causal)).reshape(-1,self.chunks,self.mixtures,1+2*self.chunk_dim)
        means=params[...,1:1+self.chunk_dim]
        logscale=params[...,1+self.chunk_dim:].clamp(-3,2)
        component=-.5*(((target[:,:,None]-means)/logscale.exp()).square()+2*logscale+math.log(2*math.pi)).sum(-1)
        return torch.logsumexp(F.log_softmax(params[...,0],dim=-1)+component,dim=-1).sum(-1)

def semantic_estimates(raw,train_indices,metadata,query,query_self):
    tags,numbers,mask,vocab,stats=prepare_conditions(metadata,train_indices)
    blocks=[]; valid=[]; field_sizes=[]
    for j,v in enumerate(vocab):
        block=np.zeros((len(raw),len(v)),dtype=np.float32)
        for i in train_indices:
            for t in tags[i,j]:
                if t: block[i,t-1]=1
        blocks.append(block);valid.append(np.any(block>0,axis=1).astype(np.float32));field_sizes.append(len(v))
    for j in range(len(NUMERIC_FIELDS)):
        blocks.append(numbers[:,j:j+1]);valid.append(mask[:,j]);field_sizes.append(1)
    nnindex=NearestNeighbors(n_neighbors=min(65,len(train_indices)),metric='cosine').fit(raw[train_indices])
    dist,nei=nnindex.kneighbors(query)
    chosen=train_indices[nei]
    kernels=np.exp(-dist/.1).astype(np.float32)
    kernels[chosen==query_self[:,None]]=0
    # Exclude self even for training rows; all candidates use training annotations only.
    output=[]
    for block,available in zip(blocks,valid):
        weights=kernels*available[chosen]
        denominator=weights.sum(1,keepdims=True)
        train_values=block[train_indices]*available[train_indices,None]
        own=np.isin(query_self,train_indices).astype(np.float32)[:,None]
        safe_self=np.clip(query_self,0,len(raw)-1)
        own_available=own*available[safe_self,None]
        fallback=(train_values.sum(0)[None]-block[safe_self]*own_available)/np.maximum(available[train_indices].sum()-own_available,1)
        estimated=(weights[:,:,None]*block[chosen]).sum(1)/np.maximum(denominator,1e-12)
        estimated=np.where(denominator>1e-12,estimated,fallback)
        # Reliability describes available training neighbors, never query metadata presence.
        reliability=denominator/np.maximum(kernels.sum(1,keepdims=True),1e-12)
        output.extend([estimated,reliability])
    condition=np.concatenate(output,axis=1).astype(np.float32)
    assert np.isfinite(condition).all()
    state=dict(train_indices=train_indices,vocabularies=vocab,numeric_stats=stats,field_sizes=field_sizes,
               metadata_train=[metadata[i] for i in train_indices],raw_training=raw[train_indices],
               method='Cosine-kNN65, temperature0.1, self excluded; only train labels; all library/query rows inferred identically',
               fields=TAG_FIELDS+NUMERIC_FIELDS)
    return condition,state

@torch.no_grad()
def predict(net,x,c,enabled=True):
    net.eval();output=[]
    for start in range(0,len(x),256):
        v=torch.from_numpy(x[start:start+256]).cuda()
        context=torch.from_numpy(c[start:start+256]).cuda()
        output.append((net.log_prob(v,context) if enabled else net.log_prob(v)).cpu().numpy())
    return np.concatenate(output).astype(np.float64)

def train_one(name,x,c,train,val,weights,enabled,out,log):
    net=(TaggedAR(c.shape[1],DIMENSION) if enabled else ContinuousAR(DIMENSION,DIMENSION//16,128,2,4)).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=8,min_lr=1e-5)
    best=float('inf');stale=0;history=[];best_epoch=0
    rng=np.random.default_rng(2026+(name=='background'))
    for epoch in range(1,161):
        net.train();order=rng.permutation(train);total=0.;weight_total=0.
        for start in range(0,len(order),256):
            ids=order[start:start+256]
            v=torch.from_numpy(x[ids]).cuda();context=torch.from_numpy(c[ids]).cuda()
            if enabled:
                # Whole-condition dropout also teaches a valid null branch.
                context=context*(torch.rand(len(ids),1,device='cuda')>=.2)
                losses=-net.log_prob(v,context)/net.chunks
            else: losses=-net.log_prob(v)/net.chunks
            w=torch.from_numpy(weights[ids]).cuda()
            loss=(losses*w).sum()/w.sum()
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step();total+=float((losses*w).sum().detach());weight_total+=float(w.sum())
        val_nll=-predict(net,x[val],c[val],enabled)/net.chunks
        # Checkpoint criterion matches the personal objective when weighted.
        validation=float(np.average(val_nll,weights=weights[val]))
        h=dict(model=name,epoch=epoch,train_nll=total/weight_total,validation_nll=validation,lr=opt.param_groups[0]['lr'])
        history.append(h)
        log(f"{name} epoch={epoch:03d} train_nll={h['train_nll']:.6f} validation_nll={validation:.6f} lr={h['lr']:.8f}")
        if validation<best-1e-4:
            best=validation;best_epoch=epoch;stale=0
            torch.save(dict(state_dict=net.state_dict(),condition_dim=c.shape[1],conditioned=enabled,dimension=DIMENSION,chunks=net.chunks,width=128,layers=2,mixtures=4),out/f'model_{name}.pt')
        else:stale+=1
        scheduler.step(validation)
        if stale>=25:break
    net.load_state_dict(torch.load(out/f'model_{name}.pt',map_location='cuda',weights_only=True)['state_dict'])
    return net,history,dict(best_epoch=best_epoch,best_validation_nll=best,trained_epochs=len(history))

def plot_history(out,history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    for model in ('personal','background'):
        h=[r for r in history if r['model']==model];e=[r['epoch'] for r in h]
        for field in ('train_nll','validation_nll'):axes[0].plot(e,[r[field] for r in h],label=model+' '+field)
        axes[1].step(e,[r['lr'] for r in h],label=model,where='post')
    axes[0].set(xlabel='Epoch',ylabel='NLL per chunk',title='Training and validation loss')
    axes[1].set(xlabel='Epoch',ylabel='Learning rate',title='Learning rate')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'loss_lr.png',dpi=150);plt.close(fig)

def main():
    global DIMENSION,BASE,RUN
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimension',type=int,choices=[64,128,192,256,512],default=192)
    args=parser.parse_args()
    DIMENSION=args.dimension
    BASE=ROOT/'models_archive/ar_preference'/('20261030_16k_lufs16_pca256_v1' if DIMENSION==256 else f'20261031_16k_lufs16_pca{DIMENSION}_sweep_v1')
    RUN=ROOT/'models_archive/ar_tagged'/f'20261031_16k_pca{DIMENSION}_v2'
    RUN.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    with np.load(HERE/'cache/library_msclap_16k_lufs16.npz') as d:
        paths=list(map(str,d['paths']));raw=d['vectors'].astype(np.float32)
    raw/=np.linalg.norm(raw,axis=1,keepdims=True)
    names=[Path(p).name for p in paths];by_name={n:i for i,n in enumerate(names)}
    baseline_rows=readrows(BASE/'relative_ranking.csv');saved={r['file']:r for r in baseline_rows}
    split=read_json(BASE/'split.json')
    personal={k:np.array([by_name[n] for n in split['personal'][k]]) for k in ('train','validation','test')}
    bg={k:np.array([by_name[n] for n in split['background']['background_'+k]]) for k in ('train','validation','test')}
    train=np.r_[personal['train'],bg['train']]
    assert len(train)==len(set(train))==1734
    known={r['文件']:r for r in read_json(ROOT/'dataset_v3/labels_v5_clean.json')}
    songs=read_json(ROOT/'models_archive/knn_preference/20261031_16k_lufs16_native1024_v1/songs.json')
    for r in songs:known[r['labels']['文件']]=r['labels']
    liked={Path(n.strip().strip('"')).name for n in (ROOT/'ncm-playback-dl/tmp/liked-files.txt').read_text(encoding='utf-8-sig').splitlines() if n.strip()}
    assert {r['labels']['文件'] for r in songs}==liked==set(split['personal']['train']+split['personal']['validation']+split['personal']['test'])
    metadata=[known.get(n,{'文件':n}) for n in names]
    behavior={r['labels']['文件']:r['behavior'] for r in songs}
    weights=np.ones(len(names),np.float32)
    for n,b in behavior.items():
        expected=.5+float(b['in_liked_playlist'])+(float(np.log1p(b['play_count']**2)) if b['play_count_known'] else 0.)
        assert np.isclose(expected,b['weight'])
        weights[by_name[n]]=expected
    with np.load(HERE/'cache/four_queries_16k_lufs16_native1024.npz') as d:
        queries=d['vectors'].astype(np.float32);query_paths=list(map(str,d['paths']))
    queries/=np.linalg.norm(queries,axis=1,keepdims=True)
    joined=np.vstack([raw,queries]);self_ids=np.r_[np.arange(len(raw)),[by_name.get(Path(p).name,-1) for p in query_paths]]
    conditions,state=semantic_estimates(raw,train,metadata,joined,self_ids)
    joblib.dump(state,RUN/'semantic_estimator.joblib',compress=3)
    np.savez_compressed(RUN/'conditions.npz',paths=paths,conditions=conditions[:len(raw)],query_paths=query_paths,query_conditions=conditions[len(raw):])
    basis=joblib.load(ROOT/'models_archive/ar_pca_sweep/20261031_16k_lufs16_v1/shared_pca512.joblib')
    if DIMENSION==256:
        basis=joblib.load(ROOT/'models_archive/preprocess_16k_256/20261030_16k_lufs16_pca256_v1/projection256.joblib')['pca']
    reduced=basis.transform(joined)[:,:DIMENSION].astype(np.float32);reduced/=np.linalg.norm(reduced,axis=1,keepdims=True)
    features=joblib.load(BASE/'features.joblib')
    x=(features['pca'].transform(reduced)/features['scale']).astype(np.float32)
    library=x[:len(raw)];condition=conditions[:len(raw)]
    test=np.r_[personal['test'],bg['test']];y=np.r_[np.ones(len(personal['test'])),np.zeros(len(bg['test']))]
    validation=np.r_[personal['validation'],bg['validation']];vy=np.r_[np.ones(len(personal['validation'])),np.zeros(len(bg['validation']))]
    results=[];four=[]
    baseline_gain=np.array([float(saved[n]['ar_log_likelihood_ratio']) for n in names])
    results.append(dict(variant='audio_uniform',validation_auc=float(roc_auc_score(vy,baseline_gain[validation])),test_auc=float(roc_auc_score(y,baseline_gain[test])),condition_shuffle_test_auc=None,condition_null_test_auc=None,model_dir=str(BASE)))
    old=read_json(ROOT/'models_archive/ar_pca_sweep/20261031_16k_lufs16_v1/results.json')
    for r in old['songs']:
        if r['dimension']==DIMENSION:four.append(dict(variant='audio_uniform',file=r['file'],percentile=r['percentile'],llr=r['llr']))
    for variant,enabled,weighted in [('audio_weighted',False,True),('tags_uniform',True,False),('tags_weighted',True,True)]:
        out=RUN/variant
        if variant=='audio_weighted' and DIMENSION==192:
            existing=ROOT/'models_archive/ar_tagged/20261031_16k_pca192_v1/audio_weighted'
            if (existing/'metrics.json').exists():out=existing
        if (out/'metrics.json').exists():
            results.append(read_json(out/'metrics.json')['comparison']);four.extend(read_json(out/'four_song_scores.json'));continue
        out.mkdir(exist_ok=False)
        random.seed(2026);np.random.seed(2026);torch.manual_seed(2026)
        histories=[];bests={};nets={}
        with (out/'training.log').open('w',encoding='utf-8',buffering=1) as stream:
            def log(s):print(variant+' '+s,flush=True);stream.write(s+'\n')
            for name,sub in [('personal',personal),('background',bg)]:
                w=weights if name=='personal' and weighted else np.ones(len(names),np.float32)
                nets[name],history,bests[name]=train_one(name,library,condition,sub['train'],sub['validation'],w,enabled,out,log)
                histories+=history
        table(out/'loss.csv',histories);plot_history(out,histories)
        p=predict(nets['personal'],x,conditions,enabled);b=predict(nets['background'],x,conditions,enabled);gain=p-b
        ref=np.sort(gain[personal['validation']]);pct=np.searchsorted(ref,gain,side='right')*100/len(ref)
        record=dict(variant=variant,validation_auc=float(roc_auc_score(vy,gain[validation])),test_auc=float(roc_auc_score(y,gain[test])),condition_shuffle_test_auc=None,condition_null_test_auc=None,model_dir=str(out))
        if enabled:
            perm=np.random.default_rng(9026).permutation(test)
            for label,context in [('shuffle',condition[perm]),('null',np.zeros_like(condition[test]))]:
                g=predict(nets['personal'],library[test],context)-predict(nets['background'],library[test],context)
                record['condition_'+label+'_test_auc']=float(roc_auc_score(y,g))
        ranked=[]
        for rank,i in enumerate(np.argsort(-gain[:len(raw)],kind='stable'),1):
            rr=saved[names[i]]
            ranked.append(dict(rank=rank,favorite_reference_percentile=float(pct[i]),ar_log_likelihood_ratio=float(gain[i]),personal_log_likelihood=float(p[i]),background_log_likelihood=float(b[i]),favorite_split=rr['favorite_split'],background_split=rr['background_split'],file=names[i],audio_path=paths[i]))
        assert len(ranked)==len(set(r['file'] for r in ranked))==4277
        table(out/'relative_ranking.csv',ranked)
        candidates=[dict(r) for r in ranked if r['background_split']=='background_test']
        for rank,r in enumerate(candidates,1):r['rank']=rank
        table(out/'candidate_recommendations_heldout.csv',candidates)
        qrows=[dict(variant=variant,file=Path(path).name,percentile=float(pct[len(raw)+j]),llr=float(gain[len(raw)+j])) for j,path in enumerate(query_paths)]
        write_json(out/'four_song_scores.json',qrows);four+=qrows
        joblib.dump(features,out/'features.joblib',compress=3)
        write_json(out/'split.json',split)
        write_json(out/'metrics.json',dict(comparison=record,best=bests,dimension=DIMENSION,chunks=DIMENSION//16,conditioned=enabled,condition_dim=conditions.shape[1],behavior_weighted=weighted,weight_formula='0.5 + L + ln(1+m^2); unknown omits log term',weight_usage='weighted batch NLL / sum(weights); no weighted resampling',batch_size=256,initial_lr=.0003,seed=2026,condition_dropout=.2,condition_method=state['method'],condition_fields=state['fields'],warning='Scores are proxies, not P(like); all song conditions are estimated from train labels to avoid favorite-membership leakage'))
        results.append(record)
        del nets
        torch.cuda.empty_cache()
    table(RUN/'comparison.csv',results);table(RUN/'four_song_scores.csv',four)
    write_json(RUN/'results.json',dict(comparison=results,four_songs=four,condition_fields=state['fields'],condition_dim=conditions.shape[1],train_only_condition_source_count=len(train),favorite_test=len(personal['test']),unlabeled_test=len(bg['test'])))
    lines=[f'# 标签与行为权重：{DIMENSION}维相对AR消融','',f'固定原{DIMENSION}维音频、PCA、分组划分和主干架构。标签条件来自只用训练标签建立的共享近邻估计器；全库和四曲统一估计，不直接把标签是否缺失输入模型。未将未标记曲当成已确认不喜欢。','', '| 方案 | 验证AUC | 测试AUC | 打乱条件测试AUC | 空条件测试AUC | 散花 | Myth | 爱如火 | 苦巧克力 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        values=[q['percentile'] for q in four if q['variant']==r['variant']]
        details=[r['condition_shuffle_test_auc'],r['condition_null_test_auc']]
        lines.append('| '+r['variant']+f" | {r['validation_auc']:.4f} | {r['test_auc']:.4f} | "+' | '.join('-' if v is None else f'{v:.4f}' for v in details)+' | '+' | '.join(f'{v:.1f}' for v in values)+' |')
    lines+=['','行为权重严格沿用用户公式。仅个人训练与验证NLL加权，背景均匀；标签网络20%整条件dropout，推断使用实际估计条件。标签与原音频有关，不能预设它提供独立新信息。单随机种子，四曲含训练歌曲，百分位不是概率。条件模型增加了编码器参数，需结合打乱/空条件诊断，不能将指标差全部归因于标签语义。']
    (RUN/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    (RUN/'COMPLETE').write_text('Three ablations trained and ranked',encoding='utf-8')
    print(json.dumps(dict(comparison=results,four_songs=four),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
