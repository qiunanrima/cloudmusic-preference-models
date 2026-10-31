"""Verify exported splits, proxy metrics, complete rankings and training traces."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from train_tagged_diffusion_sweep import RUN,ROOT,rows,write_json

def main():
    data=json.loads((RUN/'results.json').read_text(encoding='utf-8'))
    assert len(data['comparison'])==20 and len(data['four_songs'])==80
    expected={p.name for p in ROOT.glob('*.mp3')}
    reference={r['file']:(r['favorite_split'],r['background_split']) for r in rows(ROOT/'models_archive/ar_preference/20261031_16k_lufs16_pca192_sweep_v1/relative_ranking.csv')}
    for item in data['comparison']:
        out=Path(item['model_dir']);r=rows(out/'relative_ranking.csv')
        assert len(r)==len({v['file'] for v in r})==4277
        assert {v['file'] for v in r}==expected
        assert {v['file']:(v['favorite_split'],v['background_split']) for v in r}==reference
        gains=np.array([float(v['relative_denoising_gain']) for v in r]);assert np.isfinite(gains).all()
        assert np.all(gains[:-1]>=gains[1:])
        test=[v for v in r if v['favorite_split']=='test' or v['background_split']=='background_test']
        targets=[int(v['favorite_split']=='test') for v in test]
        assert sum(targets)==373 and len(targets)-sum(targets)==1845
        auc=roc_auc_score(targets,[float(v['relative_denoising_gain']) for v in test])
        assert abs(auc-item['test_auc'])<1e-12
        assert len(rows(out/'candidate_recommendations_heldout.csv'))==1845
        history=rows(out/'loss.csv')
        assert np.isfinite(np.array([[float(v[k]) for k in ['train_loss','validation_loss','lr']] for v in history])).all()
        assert all(float(v['lr'])>0 for v in history)
        assert (out/'loss_lr.png').is_file()
    result=dict(status='passed',models=20,library_per_model=4277,heldout_favorites=373,heldout_unlabeled=1845,four_song_rows=80,checks=['unique/full library coverage','same artist/recording splits','descending finite scores','AUC recomputed from exports','heldout candidate counts','finite training losses/LR','plot artifacts'])
    write_json(RUN/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
