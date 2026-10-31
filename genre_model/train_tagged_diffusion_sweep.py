"""Paired diffusion ablations on train-only inferred tags and behavior weights.

Matched architecture, splits, conditioning estimator and feature transforms.
Density proxy = background noise-prediction MSE - personal MSE.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn

from preference_data import read_json,write_json
from train_conditional_diffusion import FiLMBlock,noise_schedule

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RUN=ROOT/'models_archive/diffusion_tagged/20261031_pca_sweep_v1'
VARIANTS=[('audio_uniform',False,False),('audio_weighted',False,True),('tags_uniform',True,False),('tags_weighted',True,True)]

def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def table(path,data):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

class DenseConditionDenoiser(nn.Module):
    def __init__(self,dimension,condition_dim):
        super().__init__()
        self.dimension=dimension;self.condition_dim=condition_dim
        self.conditions=nn.Sequential(nn.Linear(condition_dim,256),nn.GELU(),nn.Linear(256,256))
        self.time=nn.Sequential(nn.Linear(64,256),nn.GELU(),nn.Linear(256,256))
        self.input=nn.Linear(dimension,256)
        self.blocks=nn.ModuleList([FiLMBlock(256) for _ in range(3)])
        self.output=nn.Sequential(nn.LayerNorm(256),nn.GELU(),nn.Linear(256,dimension))

    def forward(self,noisy,t,c):
        f=torch.exp(torch.linspace(0,-math.log(10000),32,device=noisy.device))
        angles=t.float()[:,None]*f[None]
        context=self.time(torch.cat([angles.sin(),angles.cos()],1))+self.conditions(c)
        hidden=self.input(noisy)
        for block in self.blocks:hidden=block(hidden,context)
        return self.output(hidden)

def noise_loss(net,x,t,eps,alpha,c):
    a=alpha[t,None]
    noisy=a.sqrt()*x+(1-a).sqrt()*eps
    return (net(noisy,t,c)-eps).square().mean(1)

@torch.no_grad()
def score(net,x,c,enabled,repeats=32):
    net.eval();alpha=noise_schedule(100,'cuda');output=np.zeros(len(x),np.float64)
    for t in (5,15,30,50):
        for repeat in range(repeats):
            eps=np.random.default_rng(np.random.SeedSequence([2026,t,repeat])).standard_normal(x.shape[1],dtype=np.float32)
            eps=torch.from_numpy(eps).cuda()
            for start in range(0,len(x),256):
                clean=torch.from_numpy(x[start:start+256]).cuda()
                context=torch.from_numpy(c[start:start+256]).cuda() if enabled else torch.zeros(len(clean),net.condition_dim,device='cuda')
                ts=torch.full((len(clean),),t,dtype=torch.long,device='cuda')
                output[start:start+256]+=noise_loss(net,clean,ts,eps.expand_as(clean),alpha,context).cpu().numpy()
    return output/(4*repeats)

def load_net(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    net=DenseConditionDenoiser(checkpoint['dimension'],checkpoint['condition_dim']).cuda()
    net.load_state_dict(checkpoint['state_dict']);return net

def train(branch,dimension,x,c,train_idx,val_idx,weights,enabled,out,max_epochs):
    if (out/'metrics.json').exists():return load_net(out/'model.pt'),rows(out/'loss.csv'),read_json(out/'metrics.json')
    out.mkdir(parents=True,exist_ok=False)
    seed=2026+(branch=='background');torch.manual_seed(seed);np.random.seed(seed)
    net=DenseConditionDenoiser(dimension,c.shape[1]).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=5e-4,weight_decay=.01)
    schedule=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=8,min_lr=1e-5)
    alpha=noise_schedule(100,'cuda');rng=np.random.default_rng(seed)
    val_x=torch.from_numpy(x[val_idx]).cuda()
    val_c=torch.from_numpy(c[val_idx]).cuda() if enabled else torch.zeros(len(val_idx),c.shape[1],device='cuda')
    # Fixed, per-song validation noise; never regenerated between epochs.
    vrng=np.random.default_rng(3026)
    val_t=torch.from_numpy(vrng.integers(0,100,len(val_idx),dtype=np.int64)).cuda()
    val_eps=torch.from_numpy(vrng.standard_normal(val_x.shape,dtype=np.float32)).cuda()
    val_w=torch.from_numpy(weights[val_idx]).cuda()
    history=[];best=float('inf');best_epoch=0;stale=0
    with (out/'training.log').open('w',encoding='utf-8',buffering=1) as log:
        log.write(f'dimension={dimension} branch={branch} tags={enabled} batch=256 initial_lr=0.0005 max_epochs={max_epochs} weighted_loss=True\n')
        for epoch in range(1,max_epochs+1):
            net.train();total=0.;count=0.
            for start in range(0,len(train_idx),256):
                if start==0:order=rng.permutation(train_idx)
                ids=order[start:start+256]
                clean=torch.from_numpy(x[ids]).cuda()
                context=torch.from_numpy(c[ids]).cuda() if enabled else torch.zeros(len(ids),c.shape[1],device='cuda')
                # Consume the same dropout RNG in every ablation for comparability.
                context=context*(torch.rand(len(ids),1,device='cuda')>=.2)
                t=torch.randint(100,(len(ids),),device='cuda');eps=torch.randn_like(clean)
                losses=noise_loss(net,clean,t,eps,alpha,context)
                w=torch.from_numpy(weights[ids]).cuda();loss=(losses*w).sum()/w.sum()
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.)
                opt.step();total+=float((losses*w).sum().detach());count+=float(w.sum())
            net.eval()
            with torch.no_grad():
                conditional=noise_loss(net,val_x,val_t,val_eps,alpha,val_c)
                null=noise_loss(net,val_x,val_t,val_eps,alpha,torch.zeros_like(val_c)) if enabled else conditional
                vc=float((conditional*val_w).sum()/val_w.sum());vn=float((null*val_w).sum()/val_w.sum())
                value=.8*vc+.2*vn if enabled else vc
            h=dict(model=branch,epoch=epoch,train_loss=total/count,validation_loss=value,validation_conditioned_loss=vc,validation_null_loss=vn,lr=opt.param_groups[0]['lr'])
            history.append(h)
            message=f"epoch={epoch:03d} train_loss={h['train_loss']:.6f} validation_loss={value:.6f} lr={h['lr']:.8f}"
            log.write(message+'\n')
            if epoch%20==0:print(f'{dimension}D {out.name} {branch} '+message,flush=True)
            if value<best-1e-5:
                best=value;best_epoch=epoch;stale=0
                torch.save(dict(state_dict=net.state_dict(),dimension=dimension,condition_dim=c.shape[1],conditioned=enabled,width=256,blocks=3,steps=100),out/'model.pt')
            else:stale+=1
            schedule.step(value)
            if stale>=40:break
    metrics=dict(best_epoch=best_epoch,best_validation_loss=best,trained_epochs=len(history),dimension=dimension,conditioned=enabled,batch_size=256,initial_lr=.0005,seed=seed)
    table(out/'loss.csv',history);write_json(out/'metrics.json',metrics)
    net.load_state_dict(torch.load(out/'model.pt',map_location='cuda',weights_only=True)['state_dict'])
    return net,history,metrics

def plot(out,h):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for branch in ['personal','background']:
        part=[r for r in h if r['model']==branch];e=[int(r['epoch']) for r in part]
        for key in ['train_loss','validation_loss']:axes[0].plot(e,[float(r[key]) for r in part],label=branch+' '+key)
        axes[1].step(e,[float(r['lr']) for r in part],where='post',label=branch)
    axes[0].set(xlabel='Epoch',ylabel='Noise MSE',title='Loss')
    axes[1].set(xlabel='Epoch',ylabel='Learning rate',title='LR')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'loss_lr.png',dpi=150);plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimensions',nargs='+',type=int,default=[64,128,192,256,512])
    parser.add_argument('--epochs',type=int,default=360)
    args=parser.parse_args()
    if any(d not in [64,128,192,256,512] for d in args.dimensions):parser.error('Unsupported dimension')
    RUN.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    with np.load(HERE/'cache/library_msclap_16k_lufs16.npz') as d:
        paths=list(map(str,d['paths']));raw=d['vectors'].astype(np.float32)
    raw/=np.linalg.norm(raw,axis=1,keepdims=True)
    names=[Path(p).name for p in paths];by_name={n:i for i,n in enumerate(names)}
    with np.load(HERE/'cache/four_queries_16k_lufs16_native1024.npz') as d:
        q=d['vectors'].astype(np.float32);query_paths=list(map(str,d['paths']))
    q/=np.linalg.norm(q,axis=1,keepdims=True);joined=np.vstack([raw,q])
    condition_source=ROOT/'models_archive/ar_tagged/20261031_16k_pca192_v2'
    with np.load(condition_source/'conditions.npz') as d:
        assert paths==list(map(str,d['paths'])) and query_paths==list(map(str,d['query_paths']))
        conditions=np.vstack([d['conditions'],d['query_conditions']]).astype(np.float32)
    split=read_json(ROOT/'models_archive/ar_preference/20261031_16k_lufs16_pca192_sweep_v1/split.json')
    personal={s:np.array([by_name[n] for n in split['personal'][s]]) for s in ['train','validation','test']}
    bg={s:np.array([by_name[n] for n in split['background']['background_'+s]]) for s in ['train','validation','test']}
    train_idx=np.r_[personal['train'],bg['train']]
    assert len(set(train_idx))==1734
    membership={r['file']:r for r in rows(ROOT/'models_archive/ar_preference/20261031_16k_lufs16_pca192_sweep_v1/relative_ranking.csv')}
    weights=np.ones(len(raw),np.float32)
    source=read_json(ROOT/'models_archive/knn_preference/20261031_16k_lufs16_native1024_v1/songs.json')
    assert {r['labels']['文件'] for r in source}==set(split['personal']['train']+split['personal']['validation']+split['personal']['test'])
    for r in source:
        b=r['behavior'];expected=.5+float(b['in_liked_playlist'])+(float(np.log1p(b['play_count']**2)) if b['play_count_known'] else 0.)
        assert np.isclose(expected,b['weight']);weights[by_name[r['labels']['文件']]]=expected
    test=np.r_[personal['test'],bg['test']];y=np.r_[np.ones(len(personal['test'])),np.zeros(len(bg['test']))]
    val=np.r_[personal['validation'],bg['validation']];vy=np.r_[np.ones(len(personal['validation'])),np.zeros(len(bg['validation']))]
    all_results=[];four=[]
    for dimension in args.dimensions:
        dimdir=RUN/f'pca{dimension}';dimdir.mkdir(exist_ok=True)
        basis=joblib.load(ROOT/'models_archive/ar_pca_sweep/20261031_16k_lufs16_v1/shared_pca512.joblib')
        if dimension==256:basis=joblib.load(ROOT/'models_archive/preprocess_16k_256/20261030_16k_lufs16_pca256_v1/projection256.joblib')['pca']
        reduced=basis.transform(joined)[:,:dimension].astype(np.float32);reduced/=np.linalg.norm(reduced,axis=1,keepdims=True)
        mean=reduced[train_idx].mean(0);std=reduced[train_idx].std(0);std=np.maximum(std,.5*np.median(std))
        x=((reduced-mean)/std).astype(np.float32)
        joblib.dump(dict(pca=basis,dimension=dimension,normalization='L2 after truncated PCA',mean=mean,std=std,condition_source=str(condition_source)),dimdir/'features.joblib',compress=3)
        write_json(dimdir/'split.json',split)
        bg_cache={}
        for variant,enabled,weighted in VARIANTS:
            out=dimdir/variant
            if (out/'metrics.json').exists():
                all_results.append(read_json(out/'metrics.json')['comparison']);four.extend(read_json(out/'four_song_scores.json'));continue
            print(f'TRAIN {dimension}D {variant}',flush=True)
            if enabled not in bg_cache:
                bgdir=dimdir/('background_tags' if enabled else 'background_audio')
                bgnet,bgh,bgm=train('background',dimension,x[:len(raw)],conditions[:len(raw)],bg['train'],bg['validation'],np.ones(len(raw),np.float32),enabled,bgdir,args.epochs)
                bgscores=score(bgnet,x,conditions,enabled)
                bg_cache[enabled]=(bgnet,bgh,bgm,bgscores,bgdir)
            bgnet,bgh,bgm,bgscores,bgdir=bg_cache[enabled]
            pdir=dimdir/('personal_'+variant)
            pnet,ph,pm=train('personal',dimension,x[:len(raw)],conditions[:len(raw)],personal['train'],personal['validation'],weights if weighted else np.ones(len(raw),np.float32),enabled,pdir,args.epochs)
            pscores=score(pnet,x,conditions,enabled);gain=bgscores-pscores
            ref=np.sort(gain[personal['validation']]);pct=np.searchsorted(ref,gain,side='right')*100/len(ref)
            record=dict(dimension=dimension,variant=variant,validation_auc=float(roc_auc_score(vy,gain[val])),test_auc=float(roc_auc_score(y,gain[test])),condition_shuffle_test_auc=None,condition_null_test_auc=None,model_dir=str(out))
            if enabled:
                perm=np.random.default_rng(9026).permutation(test)
                for mode,c in [('shuffle',conditions[perm]),('null',np.zeros_like(conditions[test]))]:
                    diagnostic=score(bgnet,x[test],c,True)-score(pnet,x[test],c,True)
                    record['condition_'+mode+'_test_auc']=float(roc_auc_score(y,diagnostic))
            out.mkdir(exist_ok=False)
            ranked=[]
            for rank,i in enumerate(np.argsort(-gain[:len(raw)],kind='stable'),1):
                m=membership[names[i]]
                ranked.append(dict(rank=rank,favorite_reference_percentile=float(pct[i]),relative_denoising_gain=float(gain[i]),personal_mse=float(pscores[i]),background_mse=float(bgscores[i]),favorite_split=m['favorite_split'],background_split=m['background_split'],file=names[i],audio_path=paths[i]))
            assert len(ranked)==len(set(r['file'] for r in ranked))==4277
            table(out/'relative_ranking.csv',ranked)
            candidates=[dict(r) for r in ranked if r['background_split']=='background_test']
            for rank,r in enumerate(candidates,1):r['rank']=rank
            assert len(candidates)==1845
            table(out/'candidate_recommendations_heldout.csv',candidates)
            qrows=[dict(dimension=dimension,variant=variant,file=Path(path).name,percentile=float(pct[len(raw)+j]),gain=float(gain[len(raw)+j])) for j,path in enumerate(query_paths)]
            four+=qrows;write_json(out/'four_song_scores.json',qrows)
            table(out/'loss.csv',ph+bgh);plot(out,ph+bgh)
            write_json(out/'metrics.json',dict(comparison=record,personal_model=str(pdir/'model.pt'),background_model=str(bgdir/'model.pt'),best_personal=pm,best_background=bgm,condition_source=str(condition_source),condition_dropout=.2,weighted=weighted,weight_formula='0.5+L+ln(1+m^2); unknown omits log term',score='paired background MSE minus personal MSE; common Gaussian noise across songs/models, seed2026, times5/15/30/50, repeats32',architecture='width256, 3 FiLM residual blocks; identical in all ablations',warning='Percentiles are not P(like); tags are estimated using training annotations only'))
            all_results.append(record)
            print(f'FINISHED {dimension}D {variant} validation_auc={record["validation_auc"]:.4f} test_auc={record["test_auc"]:.4f}',flush=True)
            del pnet
        del bg_cache;torch.cuda.empty_cache()
    table(RUN/'comparison.csv',all_results);table(RUN/'four_song_scores.csv',four)
    write_json(RUN/'results.json',dict(comparison=all_results,four_songs=four,favorite_test=373,unlabeled_test=1845))
    report=['# 相对扩散：标签与播放权重消融','','16k响度对齐，固定训练/验证/测试划分，PCA仅用训练集拟合。条件复用已审计的AR标签估计器；全库及四曲使用估计条件。各分支宽256、3层FiLM残差，batch256、初始LR5e-4、最多360轮、早停40；每种背景分支在相同维数下共享，个人分支独立训练。新架构的纯音频分支是匹配基线，不能直接与旧512宽扩散归因比较。','', '| PCA维数 | 纯音频 | 只加权重 | 只加标签 | 标签＋权重 |','|---|---:|---:|---:|---:|']
    for d in args.dimensions:
        m={r['variant']:r for r in all_results if r['dimension']==d}
        report.append('| '+str(d)+' | '+' | '.join(f'{m[v]["test_auc"]:.4f}' for v,_,_ in VARIANTS)+' |')
    report+=['','## 标签＋权重四曲参考百分位','','| 维数 | 散花 | Myth compiler | 爱如火 | 苦巧克力 |','|---|---:|---:|---:|---:|']
    for d in args.dimensions:
        report.append('| '+str(d)+' | '+' | '.join(f'{r["percentile"]:.1f}' for r in four if r['dimension']==d and r['variant']=='tags_weighted')+' |')
    report+=['','仅单随机种子；未标记不等于不喜欢，AUC为检索代理；百分位不是喜欢概率。四曲包含训练歌曲，不能作为独立泛化测试。所有日志、模型、loss.csv、loss_lr.png和4277首排名均保留于维数/方案目录。']
    (RUN/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    for variant,_,_ in VARIANTS:ax.plot(args.dimensions,[next(r['test_auc'] for r in all_results if r['dimension']==d and r['variant']==variant) for d in args.dimensions],marker='o',label=variant)
    ax.set(xlabel='PCA dimensions',ylabel='Held-out proxy AUC',title='Relative diffusion ablations',xticks=args.dimensions);ax.legend();ax.grid(alpha=.2)
    fig.savefig(RUN/'auc_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(len(args.dimensions),2,figsize=(13,3.5*len(args.dimensions)),constrained_layout=True,squeeze=False)
    for d,(a,b) in zip(args.dimensions,axes):
        h=rows(RUN/f'pca{d}/tags_weighted/loss.csv')
        for branch in ['personal','background']:
            part=[r for r in h if r['model']==branch];e=[int(r['epoch']) for r in part]
            for key in ['train_loss','validation_loss']:a.plot(e,[float(r[key]) for r in part],label=branch+' '+key)
            b.step(e,[float(r['lr']) for r in part],where='post',label=branch)
        a.set(title=f'{d}D tags+weights loss',xlabel='Epoch',ylabel='Noise MSE');a.legend(fontsize=7)
        b.set(title=f'{d}D LR',xlabel='Epoch',ylabel='Learning rate');b.legend(fontsize=8)
    fig.savefig(RUN/'loss_lr_all_dimensions.png',dpi=150);plt.close(fig)
    (RUN/'COMPLETE').write_text('All diffusion ablations completed',encoding='utf-8')
    print((RUN/'REPORT.md').read_text(encoding='utf-8'),flush=True)

if __name__=='__main__':main()
