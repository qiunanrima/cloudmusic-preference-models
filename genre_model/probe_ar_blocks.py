"""Audit PCA chunk contributions; choose variance weighting on validation only."""
from pathlib import Path
import argparse
import csv
import json

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from msclap_audio import default_vector_cache, read_vectors, unit_rows
from preference_data import read_json, write_json
from score_ar_preference import load_net


@torch.no_grad()
def chunk_scores(net, vectors, device):
    net.eval()
    return np.concatenate([net.log_prob_chunks(torch.from_numpy(vectors[start:start+256]).to(device)).cpu().numpy()
                           for start in range(0, len(vectors), 256)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = joblib.load(args.model / "features.joblib")
    split = read_json(args.model / "split.json")
    paths, raw = read_vectors(default_vector_cache())
    by_name = {Path(p).name: i for i, p in enumerate(paths)}
    x = (features["pca"].transform(unit_rows(raw))/features["scale"]).astype(np.float32)
    personal = load_net(args.model / "model_personal.pt", device)
    background = load_net(args.model / "model_background.pt", device)
    delta = chunk_scores(personal, x, device)-chunk_scores(background, x, device)
    subsets = {}
    for subset in ("validation", "test"):
        pos = split["personal"][subset]
        neg = split["background"]["background_"+subset]
        subsets[subset] = (np.array([by_name[n] for n in pos+neg]),
                           np.r_[np.ones(len(pos)), np.zeros(len(neg))])
    ev = features["pca"].explained_variance_ratio_
    variance = np.array([a.sum() for a in np.split(ev, personal.chunks)])
    candidates = []
    val_idx, val_y = subsets["validation"]
    for tau in (0., .3, .5, 1.):
        weight = variance**tau
        weight /= weight.sum()
        candidates.append({"tau": tau, "weights": weight.tolist(),
                           "validation_auc": float(roc_auc_score(val_y, delta[val_idx]@weight))})
    best = max(candidates, key=lambda r: r["validation_auc"])
    test_idx, test_y = subsets["test"]
    blocks = []
    for k in range(personal.chunks):
        blocks.append({"block": k+1, "pca_variance_fraction": float(variance[k]),
                       "validation_auc": float(roc_auc_score(val_y, delta[val_idx,k])),
                       "test_auc_descriptive": float(roc_auc_score(test_y, delta[test_idx,k])),
                       "validation_llr_std": float(delta[val_idx,k].std())})
    total = delta.sum(axis=1)
    with (args.model / "relative_ranking.csv").open(encoding="utf-8-sig", newline="") as stream:
        saved_rows = {r["file"]: r for r in csv.DictReader(stream)}
    saved = {name: float(r["ar_log_likelihood_ratio"]) for name, r in saved_rows.items()}
    discrepancy = max(abs(float(value)-saved[Path(p).name]) for p, value in zip(paths, total))
    if discrepancy > 1e-3:
        raise ValueError(f"Chunk decomposition differs from saved model scores: {discrepancy}")
    weighted = delta@np.array(best["weights"])
    reference = np.sort(weighted[val_idx[val_y == 1]])
    ranking = []
    for rank, index in enumerate(np.argsort(-weighted, kind="stable"), 1):
        row = saved_rows[Path(paths[index]).name]
        ranking.append({"rank": rank,
                        "favorite_reference_percentile": float(np.searchsorted(reference, weighted[index], side="right")*100/len(reference)),
                        "weighted_ar_score": float(weighted[index]), "original_ar_log_likelihood_ratio": float(total[index]),
                        "favorite_split": row["favorite_split"], "background_split": row["background_split"],
                        "file": row["file"], "audio_path": paths[index]})
    for suffix, records in (("ranking", ranking),
                            ("heldout", [r for r in ranking if r["background_split"] == "background_test"])):
        with (args.model / f"variance_weighted_{suffix}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(ranking[0]))
            writer.writeheader()
            writer.writerows(records)
    rng = np.random.default_rng(426)
    pos, neg = np.flatnonzero(test_y == 1), np.flatnonzero(test_y == 0)
    improvements = []
    for _ in range(1000):
        picked = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        improvements.append(roc_auc_score(test_y[picked], weighted[test_idx[picked]])-
                            roc_auc_score(test_y[picked], total[test_idx[picked]]))
    result = {"model": str(args.model.resolve()), "selection": "tau chosen using validation AUC only; test blocks descriptive",
              "chunks": blocks, "variance_weight_candidates": candidates,
              "selected_tau": best["tau"],
              "baseline_validation_auc": float(roc_auc_score(val_y, total[val_idx])),
              "baseline_test_auc": float(roc_auc_score(test_y, total[test_idx])),
              "selected_weighted_test_auc": float(roc_auc_score(test_y, delta[test_idx]@np.array(best["weights"]))),
              "weighted_minus_baseline_bootstrap_95ci": np.quantile(improvements, [.025, .975]).tolist(),
              "max_saved_score_discrepancy": discrepancy,
              "warning": "No checkpoint retrained or default ranking changed; weighted chunk sum is an experimental score. Unlabeled songs may be liked."}
    write_json(args.model / "block_probe.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
