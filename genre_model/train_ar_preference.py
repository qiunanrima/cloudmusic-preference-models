"""Train continuous autoregressive density models for favorite/background contrast.

No unlabeled song is treated as a confirmed dislike. All splits are reused
from the earlier diffusion experiment. Scores are density-ratio proxies, not
calibrated probabilities of liking a song.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import random

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F

from msclap_audio import PREPROCESS_ID, default_vector_cache, read_vectors, unit_rows
from preference_data import ROOT, read_json, write_json


class ContinuousAR(nn.Module):
    def __init__(self, dimension=128, chunks=8, width=128, layers=2, mixtures=4):
        super().__init__()
        if dimension % chunks:
            raise ValueError("dimension must be divisible by chunks")
        self.dimension = dimension
        self.chunks = chunks
        self.chunk_dim = dimension // chunks
        self.mixtures = mixtures
        self.bos = nn.Parameter(torch.zeros(1, 1, self.chunk_dim))
        self.input = nn.Linear(self.chunk_dim, width)
        self.position = nn.Parameter(torch.zeros(1, chunks, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=width * 2,
            dropout=.1, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers,
                                                 enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(width),
                                    nn.Linear(width, mixtures * (1 + 2 * self.chunk_dim)))

    def log_prob(self, vectors):
        return self.log_prob_chunks(vectors).sum(dim=-1)

    def log_prob_chunks(self, vectors):
        """Conditional log density for each chunk; sum recovers log_prob."""
        target = vectors.reshape(-1, self.chunks, self.chunk_dim)
        shifted = torch.cat([self.bos.expand(len(target), -1, -1), target[:, :-1]], dim=1)
        hidden = self.input(shifted) + self.position
        causal = torch.triu(torch.ones(self.chunks, self.chunks, device=vectors.device,
                                        dtype=torch.bool), diagonal=1)
        hidden = self.transformer(hidden, mask=causal)
        params = self.output(hidden).reshape(
            -1, self.chunks, self.mixtures, 1 + 2 * self.chunk_dim)
        mixture_logits = params[..., 0]
        means = params[..., 1:1 + self.chunk_dim]
        log_scale = params[..., 1 + self.chunk_dim:].clamp(-3., 2.)
        component_logp = -.5 * (((target[:, :, None] - means) /
                                  log_scale.exp()).square() + 2 * log_scale +
                                 math.log(2 * math.pi)).sum(dim=-1)
        chunk_logp = torch.logsumexp(
            F.log_softmax(mixture_logits, dim=-1) + component_logp, dim=-1)
        return chunk_logp


def train_one(name, net, vectors, train_indices, val_indices, args, device, out, log):
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=8, min_lr=1e-5)
    rng = np.random.default_rng(args.seed + (0 if name == "personal" else 1))
    val_x = torch.from_numpy(vectors[val_indices]).to(device)
    best = float("inf")
    best_epoch = 0
    stagnant = 0
    history = []
    checkpoint_path = out / f"model_{name}.pt"
    for epoch in range(1, args.epochs + 1):
        net.train()
        picks = rng.permutation(train_indices)
        total = 0.
        for start in range(0, len(picks), args.batch_size):
            chosen = picks[start:start + args.batch_size]
            x = torch.from_numpy(vectors[chosen]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = -net.log_prob(x).mean() / net.chunks
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            optimizer.step()
            total += float(loss.item()) * len(chosen)
        net.eval()
        with torch.no_grad():
            val_loss = float((-net.log_prob(val_x).mean() / net.chunks).item())
        train_loss = total / len(train_indices)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"model": name, "epoch": epoch, "train_nll": train_loss,
                        "validation_nll": val_loss, "lr": lr})
        log(f"{name} epoch {epoch:03d} train_nll={train_loss:.6f} "
            f"val_nll={val_loss:.6f} lr={lr:.8f}")
        if val_loss < best - 1e-4:
            best, best_epoch, stagnant = val_loss, epoch, 0
            torch.save({"state_dict": net.state_dict(), "dimension": net.dimension,
                        "chunks": net.chunks, "mixtures": net.mixtures,
                        "width": args.width, "layers": args.layers}, checkpoint_path)
        else:
            stagnant += 1
        scheduler.step(val_loss)
        if stagnant >= args.patience:
            log(f"{name} early stopping: epoch={epoch}; best_epoch={best_epoch}")
            break
    net.load_state_dict(torch.load(checkpoint_path, map_location=device,
                                   weights_only=True)["state_dict"])
    return history, best_epoch, best


@torch.no_grad()
def score(net, vectors, device, batch_size=256):
    net.eval()
    result = np.empty(len(vectors), dtype=np.float64)
    for start in range(0, len(vectors), batch_size):
        stop = min(start + batch_size, len(vectors))
        x = torch.from_numpy(vectors[start:stop]).to(device)
        result[start:stop] = net.log_prob(x).cpu().numpy()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-experiment", type=Path,
                        default=Path(read_json(ROOT / "models_archive/diffusion_background/latest.json")["model_dir"]))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--no-projection", action="store_true")
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--mixtures", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.dimension % args.chunks:
        parser.error("dimension must be divisible by chunks")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = read_json(args.split_experiment / "split.json")
    personal_dir = Path(split["personal_model"])
    personal_split = read_json(personal_dir / "split.json")
    paths, raw_vectors = read_vectors(default_vector_cache())
    vectors = unit_rows(raw_vectors)
    names = np.array([Path(p).name for p in paths])
    by_name = {name: i for i, name in enumerate(names)}
    def indices(values):
        return np.array([by_name[name] for name in values], dtype=np.int64)
    personal_train = indices(personal_split["train"])
    personal_val = indices(personal_split["validation"])
    personal_test = indices(personal_split["test"])
    bg_train = indices(split["background_train"])
    bg_val = indices(split["background_validation"])
    bg_test = indices(split["background_test"])
    if (set(personal_train) | set(personal_val) | set(personal_test)) & (set(bg_train) | set(bg_val) | set(bg_test)):
        raise ValueError("Favorite/background splits overlap")
    out = args.out or ROOT / "models_archive/ar_preference" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    log_stream = (out / "training.log").open("w", encoding="utf-8", buffering=1)
    def log(message):
        print(message, flush=True)
        log_stream.write(message + "\n")
    fit_indices = np.r_[personal_train, bg_train]
    if args.no_projection and args.dimension != vectors.shape[1]:
        raise ValueError('No-projection AR dimension must equal the input dimension')
    pca = (StandardScaler(with_mean=True, with_std=False) if args.no_projection else
           PCA(n_components=args.dimension, svd_solver="randomized", random_state=args.seed))
    projected = pca.fit_transform(vectors[fit_indices])
    scale = np.maximum(projected.std(axis=0), .5 * np.median(projected.std(axis=0)))
    transformed = (pca.transform(vectors) / scale).astype(np.float32)
    joblib.dump({"pca": pca, "scale": scale}, out / "features.joblib", compress=3)
    write_json(out / "split.json", {"personal": personal_split,
                                    "background": split, "source_experiment": str(args.split_experiment)})
    log(f"device={device}; favorite_train={len(personal_train)}; background_train={len(bg_train)}; "
        f"favorite_test={len(personal_test)}; background_test={len(bg_test)}; "
        f"feature_dim={args.dimension}; no_projection={args.no_projection}; "
        f"batch_size={args.batch_size}; lr={args.lr}; sampling=uniform")
    histories = []
    bests = {}
    nets = {}
    for name, train_idx, val_idx in (("personal", personal_train, personal_val),
                                     ("background", bg_train, bg_val)):
        net = ContinuousAR(args.dimension, args.chunks, args.width,
                           args.layers, args.mixtures).to(device)
        log(f"{name} parameters={sum(p.numel() for p in net.parameters())}")
        history, best_epoch, best_loss = train_one(name, net, transformed,
                                                  train_idx, val_idx, args,
                                                  device, out, log)
        histories.extend(history)
        bests[name] = {"best_epoch": best_epoch, "best_validation_nll": best_loss,
                       "trained_epochs": len(history)}
        nets[name] = net
    with (out / "loss.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(histories[0]))
        writer.writeheader()
        writer.writerows(histories)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    for name in nets:
        current = [row for row in histories if row["model"] == name]
        ax.plot([row["epoch"] for row in current], [row["train_nll"] for row in current],
                label=f"{name} train")
        ax.plot([row["epoch"] for row in current], [row["validation_nll"] for row in current],
                label=f"{name} validation")
    ax.set(xlabel="Epoch", ylabel="NLL per chunk", title="Continuous AR training")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), dpi=160)
    for name in nets:
        current = [row for row in histories if row["model"] == name]
        ax.step([row["epoch"] for row in current], [row["lr"] for row in current],
                where="post", label=name)
    ax.set(xlabel="Epoch", ylabel="Learning rate", title="Continuous AR learning rate")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "learning_rate.png")
    plt.close(fig)
    personal_ll = score(nets["personal"], transformed, device)
    background_ll = score(nets["background"], transformed, device)
    gain = personal_ll - background_ll
    test_indices = np.r_[personal_test, bg_test]
    targets = np.r_[np.ones(len(personal_test)), np.zeros(len(bg_test))]
    raw_auc = float(roc_auc_score(targets, personal_ll[test_indices]))
    relative_auc = float(roc_auc_score(targets, gain[test_indices]))
    reference = np.sort(gain[personal_val])
    percentiles = np.searchsorted(reference, gain, side="right") * 100 / len(reference)
    centroid = vectors[bg_train].mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    centroid_cos = vectors @ centroid
    favorite_split = np.full(len(names), "unlabeled", dtype=object)
    for label, idx in (("train", personal_train), ("validation", personal_val), ("test", personal_test)):
        favorite_split[idx] = label
    outside = personal_split.get('other_favorites', [])
    if outside:
        favorite_split[indices(outside)] = 'favorite_outside_top200'
    background_split = np.full(len(names), "background_excluded", dtype=object)
    background_split[np.r_[personal_train, personal_val, personal_test]] = "favorite"
    if outside:
        background_split[indices(outside)] = 'favorite_outside_top200'
    excluded = split.get("excluded_near_favorite", [])
    if excluded:
        background_split[indices(excluded)] = "duplicate_of_favorite_excluded"
    for label, idx in (("background_train", bg_train), ("background_validation", bg_val),
                       ("background_test", bg_test)):
        background_split[idx] = label
    order = np.argsort(-gain, kind="stable")
    with (out / "relative_ranking.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["rank", "favorite_reference_percentile", "ar_log_likelihood_ratio",
                         "personal_log_likelihood", "background_log_likelihood",
                         "favorite_split", "background_split", "file", "audio_path"])
        for rank, i in enumerate(order, 1):
            writer.writerow([rank, float(percentiles[i]), float(gain[i]),
                             float(personal_ll[i]), float(background_ll[i]),
                             favorite_split[i], background_split[i], names[i], paths[i]])
    with (out / "candidate_recommendations_heldout.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["rank", "favorite_reference_percentile", "ar_log_likelihood_ratio",
                         "personal_log_likelihood", "background_log_likelihood", "file", "audio_path"])
        for rank, i in enumerate([i for i in order if background_split[i] == "background_test"], 1):
            writer.writerow([rank, float(percentiles[i]), float(gain[i]),
                             float(personal_ll[i]), float(background_ll[i]), names[i], paths[i]])
    results = {"model": ("Native-coordinate continuous autoregressive Transformer Gaussian mixture; no projection" if args.no_projection else "PCA plus chunked continuous autoregressive Transformer Gaussian mixture"),
               "preprocess_id": PREPROCESS_ID, "vector_cache": str(default_vector_cache().resolve()),
               "sampling": "uniform favorite and uniform background; no unlabeled-as-negative training",
               "score": "personal log likelihood minus background log likelihood in train-standardized native coordinates" if args.no_projection else "personal log likelihood minus background log likelihood in PCA space",
               "warning": "Density-ratio ranking, not a calibrated P(like). Unlabeled songs may be liked.",
               "feature_dimension": args.dimension,
               "pca_explained_variance": None if args.no_projection else float(pca.explained_variance_ratio_.sum()),
               "no_projection": args.no_projection,
               "chunks": args.chunks, "mixtures": args.mixtures, "width": args.width,
               "layers": args.layers, "batch_size": args.batch_size, "initial_lr": args.lr,
               "favorite_train": len(personal_train), "background_train": len(bg_train),
               "favorite_test": len(personal_test), "background_test": len(bg_test),
               "proxy_auc_personal_likelihood": raw_auc,
               "proxy_auc_relative_ar": relative_auc,
               "spearman_centroid_personal_likelihood": float(spearmanr(centroid_cos, personal_ll).statistic),
               "spearman_centroid_relative_ar": float(spearmanr(centroid_cos, gain).statistic),
               "best": bests, "seed": args.seed, "split_experiment": str(args.split_experiment)}
    write_json(out / "metrics.json", results)
    write_json(ROOT / "models_archive/ar_preference/latest.json", {"model_dir": str(out.resolve())})
    log(f"Heldout proxy AUC: personal_ll={raw_auc:.4f}; relative_ar={relative_auc:.4f}")
    log(f"Saved AR model and ranking: {out}")
    log_stream.close()


if __name__ == "__main__":
    main()
