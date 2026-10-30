"""Train a real neural replay-intensity regressor on favorite MS-CLAP vectors.

Only known playback counts provide targets. Uncollected songs are unlabeled,
not negative examples. The neural leaderboard is experimental and separate
from the existing positive-only KNN leaderboard.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from msclap_audio import DIM, PREPROCESS_ID, default_vector_cache, read_vectors
from preference_knn import PreferenceModel, artist_groups, model_directory
from preference_data import ROOT, write_json


class ReplayNet(nn.Module):
    def __init__(self, dropout=.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def metrics(target, prediction):
    rho = spearmanr(target, prediction).statistic if np.std(prediction) > 1e-9 else np.nan
    return {"n": len(target), "mae_log1p": float(np.mean(np.abs(target - prediction))),
            "spearman": float(rho) if np.isfinite(rho) else None}


@torch.no_grad()
def infer(net, x, device, batch_size=1024):
    net.eval()
    result = []
    for start in range(0, len(x), batch_size):
        result.append(net(torch.from_numpy(x[start:start + batch_size]).to(device)).cpu().numpy())
    return np.concatenate(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.lr <= 0:
        raise ValueError("batch size and learning rate must be positive")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    knn = PreferenceModel(model_directory(args.model))
    x = np.asarray(knn.vectors, dtype=np.float32)
    y = np.log1p(np.asarray(knn.counts, dtype=np.float64))
    groups = artist_groups(knn.rows, knn.recordings)
    indices = np.arange(len(x))
    development, test = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                                                random_state=args.seed).split(indices, groups=groups))
    train, validation = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                                                random_state=args.seed + 1).split(
                                                    development, groups=groups[development]))
    train, validation = development[train], development[validation]
    train, validation, test = [part[np.isfinite(y[part])] for part in (train, validation, test)]
    if min(len(train), len(validation), len(test)) < 10:
        raise ValueError("Insufficient known play counts in one of the splits")
    mean, std = float(np.mean(y[train])), float(np.std(y[train]))
    if std < 1e-6:
        raise ValueError("Playback target has no variation")
    train_set = TensorDataset(torch.from_numpy(x[train]),
                              torch.from_numpy(((y[train] - mean) / std).astype(np.float32)))
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    net = ReplayNet().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=7, min_lr=1e-5)
    loss_fn = nn.SmoothL1Loss(beta=.5)
    out = args.out or ROOT / "models_archive/replay_net" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    log_stream = (out / "training.log").open("w", encoding="utf-8", buffering=1)
    def log(message, **kwargs):
        print(message, flush=True)
        log_stream.write(message + "\n")
    log(f"Device={device}; train={len(train)}, validation={len(validation)}, test={len(test)}; "
          f"batch_size={args.batch_size}, initial_lr={args.lr}", flush=True)
    history, best_loss, best_epoch, stagnant = [], float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        net.train()
        total = 0.
        for features, target in loader:
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(net(features), target)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(features)
        train_loss = total / len(train)
        val_normalized = infer(net, x[validation], device)
        val_loss = float(np.mean(torch.nn.functional.smooth_l1_loss(
            torch.from_numpy(val_normalized),
            torch.from_numpy(((y[validation] - mean) / std).astype(np.float32)),
            beta=.5, reduction="none").numpy()))
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "lr": lr})
        log(f"epoch {epoch:03d} train_loss={train_loss:.6f} "
              f"val_loss={val_loss:.6f} lr={lr:.8f}", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, stagnant = val_loss, epoch, 0
            torch.save({"state_dict": net.state_dict(), "target_mean": mean,
                        "target_std": std, "seed": args.seed}, out / "model.pt")
        else:
            stagnant += 1
        scheduler.step(val_loss)
        if stagnant >= args.patience:
            log(f"Early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
            break
    checkpoint = torch.load(out / "model.pt", map_location=device, weights_only=True)
    net.load_state_dict(checkpoint["state_dict"])
    test_pred = infer(net, x[test], device) * std + mean
    baseline = np.full(len(test), np.median(y[train]))
    result = {"model": f"MS-CLAP PCA {DIM} -> MLP 256 -> 64 -> replay intensity",
              "input_dimension": DIM,
              "preprocess_id": PREPROCESS_ID, "vector_cache": str(default_vector_cache().resolve()),
              "target": "log1p(playCount), known counts only; not P(like)",
              "knn_model_dir": str(knn.directory), "seed": args.seed,
              "split": "artist and duplicate-recording groups; 20% test then 20% of development validation",
              "train_count": len(train), "validation_count": len(validation), "test_count": len(test),
              "batch_size": args.batch_size, "initial_lr": args.lr,
              "optimizer": "AdamW(weight_decay=0.01)", "loss": "SmoothL1(beta=0.5) on train-standardized log1p playCount",
              "scheduler": "ReduceLROnPlateau(factor=0.5,patience=7,min_lr=1e-5)",
              "best_epoch": best_epoch, "best_val_loss": best_loss,
              "heldout": metrics(y[test], test_pred), "heldout_median_baseline": metrics(y[test], baseline),
              "limitations": "No dislike labels; predicts replay intensity among favorites, not binary liking"}
    with (out / "loss.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    ax.plot([v["epoch"] for v in history], [v["train_loss"] for v in history], label="Train loss")
    ax.plot([v["epoch"] for v in history], [v["val_loss"] for v in history], label="Validation loss")
    ax.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"Best epoch {best_epoch}")
    ax.set(xlabel="Epoch", ylabel="SmoothL1 loss (standardized target)", title="Replay model learning curve")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    ax.step([v["epoch"] for v in history], [v["lr"] for v in history], where="post")
    ax.set(xlabel="Epoch", ylabel="Learning rate", title="Replay model learning rate")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "learning_rate.png")
    plt.close(fig)
    write_json(out / "metrics.json", result)
    write_json(out / "split.json", {name: [knn.rows[int(i)]["文件"] for i in part]
                                    for name, part in (("train", train), ("validation", validation),
                                                       ("test", test))})

    paths, library_vectors = read_vectors(default_vector_cache())
    predictions = infer(net, library_vectors.astype(np.float32), device) * std + mean
    reference = np.sort(infer(net, x, device) * std + mean)
    scores = (np.searchsorted(reference, predictions, side="left") +
              np.searchsorted(reference, predictions, side="right")) * 50 / len(reference)
    records = sorted(zip(paths, predictions, scores), key=lambda item: -item[2])
    with (out / "library_replay_ranking.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "replay_percentile", "predicted_log1p_playcount", "file", "audio_path"])
        for rank, (path, prediction, score) in enumerate(records, 1):
            writer.writerow([rank, float(score), float(prediction), Path(path).name, path])
    write_json(ROOT / "models_archive/replay_net/latest.json", {"model_dir": str(out.resolve())})
    log(f"Held-out={result['heldout']}; baseline={result['heldout_median_baseline']}", flush=True)
    log(f"Saved model, loss curve and experimental leaderboard: {out}", flush=True)
    log_stream.close()


if __name__ == "__main__":
    main()
