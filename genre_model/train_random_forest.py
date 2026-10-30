"""Train and evaluate a random forest for favorite-song replay intensity."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit

from msclap_audio import DIM, PREPROCESS_ID, default_vector_cache, read_vectors
from preference_data import ROOT, write_json
from preference_knn import PreferenceModel, artist_groups, model_directory


def metrics(target, prediction):
    rho = spearmanr(target, prediction).statistic if np.std(prediction) > 1e-9 else np.nan
    return {"n": int(len(target)), "mae_log1p": float(np.mean(np.abs(target - prediction))),
            "spearman": float(rho) if np.isfinite(rho) else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    knn = PreferenceModel(model_directory(args.model))
    x = np.asarray(knn.vectors, dtype=np.float32)
    y = np.log1p(np.asarray(knn.counts, dtype=np.float64))
    groups = artist_groups(knn.rows, knn.recordings)
    indices = np.arange(len(x))
    development, test = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                                                random_state=args.seed).split(indices, groups=groups))
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                                     random_state=args.seed + 1).split(
                                         development, groups=groups[development]))
    train, validation = development[tr], development[va]
    train, validation, test = [part[np.isfinite(y[part])] for part in (train, validation, test)]
    out = args.out or ROOT / "models_archive/random_forest_replay" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    log_stream = (out / "training.log").open("w", encoding="utf-8", buffering=1)
    def log(message, **kwargs):
        print(message, flush=True)
        log_stream.write(message + "\n")
    log(f"train={len(train)} validation={len(validation)} test={len(test)}; "
          f"features={x.shape[1]}, jobs={args.jobs}", flush=True)
    candidates = [
        {"max_depth": 8, "min_samples_leaf": 5, "max_features": .1},
        {"max_depth": 12, "min_samples_leaf": 5, "max_features": .3},
        {"max_depth": 16, "min_samples_leaf": 3, "max_features": .1},
        {"max_depth": None, "min_samples_leaf": 8, "max_features": .3},
    ]
    validation_results = []
    for params in candidates:
        forest = RandomForestRegressor(n_estimators=120, bootstrap=True,
                                       max_samples=.8, n_jobs=args.jobs,
                                       random_state=args.seed, **params)
        forest.fit(x[train], y[train])
        val = metrics(y[validation], forest.predict(x[validation]))
        result = {**params, **val}
        validation_results.append(result)
        log(f"params={params} val_mae={val['mae_log1p']:.6f} "
              f"val_spearman={val['spearman']}", flush=True)
    best = min(validation_results, key=lambda row: row["mae_log1p"])
    params = {key: best[key] for key in ("max_depth", "min_samples_leaf", "max_features")}
    final_train = np.sort(np.concatenate([train, validation]))
    forest = RandomForestRegressor(n_estimators=400, bootstrap=True, max_samples=.8,
                                   oob_score=True, n_jobs=args.jobs,
                                   random_state=args.seed, **params)
    forest.fit(x[final_train], y[final_train])
    test_pred = forest.predict(x[test])
    baseline = np.full(len(test), np.median(y[final_train]))
    result = {"model": f"RandomForestRegressor on {DIM}-dimensional frozen cache vectors",
              "input_dimension": DIM,
              "preprocess_id": PREPROCESS_ID, "vector_cache": str(default_vector_cache().resolve()),
              "target": "log1p(playCount), known counts only; not P(like)",
              "knn_model_dir": str(knn.directory), "seed": args.seed,
              "split": "artist and duplicate-recording groups; same seed and partitions as replay net",
              "train_count": len(train), "validation_count": len(validation),
              "final_train_count": len(final_train), "test_count": len(test),
              "candidates": validation_results, "best_params": params,
              "final_estimators": forest.n_estimators,
              "heldout": metrics(y[test], test_pred),
              "heldout_median_baseline": metrics(y[test], baseline),
              "oob": metrics(y[final_train], forest.oob_prediction_),
              "limitations": "No dislike labels; predicts replay intensity among favorites, not binary liking"}
    joblib.dump(forest, out / "model.joblib", compress=3)
    write_json(out / "metrics.json", result)
    write_json(out / "split.json", {name: [knn.rows[int(i)]["文件"] for i in part]
                                    for name, part in (("train", train), ("validation", validation),
                                                       ("test", test))})
    with (out / "validation.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(validation_results[0]))
        writer.writeheader()
        writer.writerows(validation_results)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    labels = [f"depth={r['max_depth']}, leaf={r['min_samples_leaf']}, features={r['max_features']}"
              for r in validation_results]
    ax.barh(labels, [r["mae_log1p"] for r in validation_results])
    ax.set(xlabel="Validation MAE of log1p play count (lower is better)",
           title="Random forest hyperparameter comparison")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "validation.png")
    plt.close(fig)

    paths, vectors = read_vectors(default_vector_cache())
    predictions = forest.predict(vectors)
    reference = np.sort(forest.oob_prediction_)
    scores = (np.searchsorted(reference, predictions, side="left") +
              np.searchsorted(reference, predictions, side="right")) * 50 / len(reference)
    rows = sorted(zip(paths, predictions, scores), key=lambda row: (-row[2], Path(row[0]).name))
    with (out / "library_replay_ranking.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "replay_percentile", "predicted_log1p_playcount", "file", "audio_path"])
        for rank, (path, prediction, score) in enumerate(rows, 1):
            writer.writerow([rank, float(score), float(prediction), Path(path).name, path])
    write_json(ROOT / "models_archive/random_forest_replay/latest.json", {"model_dir": str(out.resolve())})
    log(f"heldout={result['heldout']}; baseline={result['heldout_median_baseline']}", flush=True)
    log(f"saved model, validation plot and experimental ranking: {out}", flush=True)
    log_stream.close()


if __name__ == "__main__":
    main()
