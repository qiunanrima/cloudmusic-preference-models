"""Serial GPU execution and verification of all conditional AR dimensions."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RUN=ROOT/'models_archive/ar_tagged/dimension_comparison_20261031'

def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def table(path,data):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def main():
    RUN.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
    results=[];songs=[];reference=None
    for dimension in [64,128,192,256,512]:
        out=ROOT/'models_archive/ar_tagged'/f'20261031_16k_pca{dimension}_v2'
        if not (out/'COMPLETE').exists():
            print(f'TRAIN {dimension}: weighted-only, tags-only, tags+weights; serial GPU',flush=True)
            with (RUN/f'console_{dimension}.log').open('w',encoding='utf-8') as f:
                subprocess.run([sys.executable,'train_tagged_ar_experiment.py','--dimension',str(dimension)],cwd=HERE,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        current=load(out/'results.json')
        assert (current['favorite_test'],current['unlabeled_test'])==(373,1845)
        for record in current['comparison']:
            p=Path(record['model_dir']);ranking=rows(p/'relative_ranking.csv')
            identity={r['file']:(r['favorite_split'],r['background_split']) for r in ranking}
            if reference is None:reference=identity
            assert identity==reference and len(ranking)==len(identity)==4277
            assert all(float(a['ar_log_likelihood_ratio'])>=float(b['ar_log_likelihood_ratio']) for a,b in zip(ranking,ranking[1:]))
            results.append(dict(dimension=dimension,**record))
        songs.extend([dict(dimension=dimension,**r) for r in current['four_songs']])
        print(f'FINISHED {dimension}: all rankings verified',flush=True)
    table(RUN/'comparison.csv',results);table(RUN/'four_song_scores.csv',songs)
    (RUN/'results.json').write_text(json.dumps(dict(comparison=results,four_songs=songs),ensure_ascii=False,indent=2),encoding='utf-8')
    variants=['audio_uniform','audio_weighted','tags_uniform','tags_weighted']
    report=['# 全维数标签与权重消融','','共同373首收藏与1845首未标记留出曲。每块16维，batch256，LR3e-4，单随机种子。条件统一由训练集标签近邻估计，不直接使用待评分歌曲的真实标签；条件缺失与收藏身份无直接关联。','','| 维数 | 原音频 | 只加权重 | 只加标签 | 标签＋权重 |','|---|---:|---:|---:|---:|']
    for d in [64,128,192,256,512]:
        scores={r['variant']:r['test_auc'] for r in results if r['dimension']==d}
        report.append('| '+str(d)+' | '+' | '.join(f'{scores[v]:.4f}' for v in variants)+' |')
    report+=['','## 标签＋权重：四首歌曲','','| 维数 | 散花 | Myth compiler | 爱如火 | 苦巧克力 |','|---|---:|---:|---:|---:|']
    for d in [64,128,192,256,512]:
        s=[r['percentile'] for r in songs if r['dimension']==d and r['variant']=='tags_weighted']
        report.append('| '+str(d)+' | '+' | '.join(f'{v:.1f}' for v in s)+' |')
    report+=['','分数不是喜欢概率。不同维数的序列长度也不同。所有模型与全库排名均保留于相应维数目录；192维复用已完成实验。加条件也增加参数量，不应将差异全解释为标签语义。']
    (RUN/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    dimensions=[64,128,192,256,512]
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    for v in variants:
        ax.plot(dimensions,[next(r['test_auc'] for r in results if r['dimension']==d and r['variant']==v) for d in dimensions],marker='o',label=v)
    ax.set(xlabel='PCA dimensions',ylabel='Held-out proxy AUC',title='Relative AR: labels and behavior weight ablations',xticks=dimensions)
    ax.legend();ax.grid(alpha=.2);fig.savefig(RUN/'auc_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(5,2,figsize=(13,17),constrained_layout=True)
    for d,(a,b) in zip(dimensions,axes):
        h=rows(ROOT/'models_archive/ar_tagged'/f'20261031_16k_pca{d}_v2/tags_weighted/loss.csv')
        for branch in ['personal','background']:
            part=[r for r in h if r['model']==branch];e=[int(r['epoch']) for r in part]
            for key in ['train_nll','validation_nll']:a.plot(e,[float(r[key]) for r in part],label=branch+' '+key)
            b.step(e,[float(r['lr']) for r in part],label=branch,where='post')
        a.set(title=f'{d}D tags+weights loss',xlabel='Epoch',ylabel='NLL/chunk');a.legend(fontsize=7)
        b.set(title=f'{d}D learning rate',xlabel='Epoch',ylabel='LR');b.legend(fontsize=8)
    fig.savefig(RUN/'loss_lr_all_dimensions.png',dpi=150);plt.close(fig)
    (RUN/'COMPLETE').write_text('All dimensions and ablations verified',encoding='utf-8')
    print((RUN/'REPORT.md').read_text(encoding='utf-8'),flush=True)

if __name__=='__main__':main()
