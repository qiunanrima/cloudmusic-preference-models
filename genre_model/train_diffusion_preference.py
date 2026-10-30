"""Weighted latent diffusion experiment for positive-only music preference.

The normalized favorite weights are sampling probabilities, not probabilities
of liking an arbitrary song. Candidate scores are held-out favorite percentiles
of a fixed-noise denoising error, not calibrated P(like).
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
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch import nn

from msclap_audio import PREPROCESS_ID, default_vector_cache, read_vectors, unit_rows
from preference_data import ROOT, write_json
from preference_knn import PreferenceModel, artist_groups, model_directory


class Denoiser(nn.Module):
    def __init__(self, dim=128, width=384, dropout=.1):
        super().__init__()
        self.dim = dim
        self.input = nn.Sequential(nn.Linear(dim + 64, width), nn.LayerNorm(width), nn.GELU())
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(width, width), nn.LayerNorm(width))
            for _ in range(2)
        ])
        self.output = nn.Linear(width, dim)

    def forward(self, noisy, t):
        frequencies = torch.exp(torch.linspace(0, -math.log(10000), 32, device=noisy.device))
        angles = t.float()[:, None] * frequencies[None]
        time_embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        hidden = self.input(torch.cat([noisy, time_embedding], dim=1))
        for block in self.blocks:
            hidden = torch.nn.functional.gelu(hidden + block(hidden))
        return self.output(hidden)


def alpha_schedule(steps, device):
    betas = torch.linspace(1e-4, .02, steps, device=device)
    return torch.cumprod(1 - betas, dim=0)


def denoise_loss(net, clean, t, noise, alpha_bar):
    alpha = alpha_bar[t][:, None]
    noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
    predicted = net(noisy, t)
    return (predicted - noise).square().mean(dim=1)


@torch.no_grad()
def score_latents(net, latents, alpha_bar, device, batch_size=512,
                  timesteps=(5, 15, 30, 50), repeats=32, seed=2026):
    """Average fixed-seed denoising MSE; lower means closer to learned density."""
    net.eval()
    scores = np.zeros(len(latents), dtype=np.float64)
    for t_value in timesteps:
        for repeat in range(repeats):
            noise = np.random.default_rng(np.random.SeedSequence(
                [seed, int(t_value), repeat])).standard_normal(latents.shape[1], dtype=np.float32)
            for start in range(0, len(latents), batch_size):
                stop = min(start + batch_size, len(latents))
                clean = torch.from_numpy(latents[start:stop]).to(device)
                eps = torch.from_numpy(noise).to(device).expand(stop - start, -1)
                t = torch.full((stop - start,), t_value, device=device, dtype=torch.long)
                scores[start:stop] += denoise_loss(net, clean, t, eps, alpha_bar).cpu().numpy()
    return scores / (len(timesteps) * repeats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--no-projection", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.lr <= 0 or args.latent_dim < 2:
        raise ValueError("Invalid training parameters")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    knn = PreferenceModel(model_directory(args.model))
    x = unit_rows(knn.vectors)
    groups = artist_groups(knn.rows, knn.recordings)
    indices = np.arange(len(x))
    development, test = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                                                random_state=args.seed).split(indices, groups=groups))
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                                     random_state=args.seed + 1).split(
                                         development, groups=groups[development]))
    train, validation = development[tr], development[va]
    out = args.out or ROOT / "models_archive/diffusion_preference" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    log_stream = (out / "training.log").open("w", encoding="utf-8", buffering=1)
    def log(message, **kwargs):
        print(message, flush=True)
        log_stream.write(message + "\n")

    # Fit the feature map on training favorites only. Whiten the retained
    # components so diffusion noise has comparable scale in each direction.
    if args.no_projection:
        args.latent_dim = x.shape[1]
        pca = StandardScaler(with_mean=True, with_std=False).fit(x[train])
        raw_std = x[train].std(axis=0)
        scale = np.maximum(raw_std, .5*np.median(raw_std)).astype(np.float32)
    else:
        pca = PCA(n_components=args.latent_dim, svd_solver="randomized", random_state=args.seed)
        pca.fit(x[train])
        scale = np.sqrt(np.maximum(pca.explained_variance_, 1e-8)).astype(np.float32)
    z = (pca.transform(x) / scale).astype(np.float32)
    joblib.dump({"pca": pca, "scale": scale}, out / "projection.joblib", compress=3)
    sampling_prob = knn.weights[train] / knn.weights[train].sum()
    effective_samples = float(1 / np.sum(sampling_prob**2))
    net = Denoiser(args.latent_dim).to(device)
    alpha_bar = alpha_schedule(100, device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=7, min_lr=1e-5)
    rng = np.random.default_rng(args.seed)
    val_clean = torch.from_numpy(z[validation]).to(device)
    val_rng = np.random.default_rng(args.seed + 1000)
    val_t = torch.from_numpy(val_rng.integers(0, len(alpha_bar), len(validation),
                                               dtype=np.int64)).to(device)
    val_eps = torch.from_numpy(val_rng.standard_normal(val_clean.shape, dtype=np.float32)).to(device)
    history, best_loss, best_epoch, stagnant = [], float("inf"), 0, 0
    log(f"device={device}; favorites train={len(train)}, validation={len(validation)}, test={len(test)}; "
          f"latent={args.latent_dim}; batch_size={args.batch_size}; lr={args.lr}; "
          f"weighted_sampling_effective_n={effective_samples:.1f}", flush=True)
    for epoch in range(1, args.epochs + 1):
        net.train()
        picks = rng.choice(len(train), size=len(train), replace=True, p=sampling_prob)
        total = 0.
        for start in range(0, len(picks), args.batch_size):
            chosen = train[picks[start:start + args.batch_size]]
            clean = torch.from_numpy(z[chosen]).to(device)
            t = torch.randint(len(alpha_bar), (len(clean),), device=device)
            eps = torch.randn_like(clean)
            optimizer.zero_grad(set_to_none=True)
            loss = denoise_loss(net, clean, t, eps, alpha_bar).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(clean)
        train_loss = total / len(train)
        net.eval()
        with torch.no_grad():
            val_loss = float(denoise_loss(net, val_clean, val_t, val_eps, alpha_bar).mean().item())
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "validation_loss": val_loss, "lr": lr})
        log(f"epoch {epoch:03d} train_loss={train_loss:.6f} "
              f"val_loss={val_loss:.6f} lr={lr:.8f}", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, stagnant = val_loss, epoch, 0
            torch.save({"state_dict": net.state_dict(), "latent_dim": args.latent_dim,
                        "steps": len(alpha_bar), "seed": args.seed}, out / "model.pt")
        else:
            stagnant += 1
        scheduler.step(val_loss)
        if stagnant >= args.patience:
            log(f"early stopping at epoch {epoch}; best={best_epoch}", flush=True)
            break
    checkpoint = torch.load(out / "model.pt", map_location=device, weights_only=True)
    net.load_state_dict(checkpoint["state_dict"])
    with (out / "loss.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="Train")
    ax.plot([h["epoch"] for h in history], [h["validation_loss"] for h in history], label="Validation")
    ax.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"Best epoch {best_epoch}")
    ax.set(xlabel="Epoch", ylabel="Noise-prediction MSE", title="Weighted latent diffusion training")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    ax.step([h["epoch"] for h in history], [h["lr"] for h in history], where="post")
    ax.set(xlabel="Epoch", ylabel="Learning rate", title="Latent diffusion learning rate")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "learning_rate.png")
    plt.close(fig)

    paths, library_vectors = read_vectors(default_vector_cache())
    library_z = (pca.transform(library_vectors) / scale).astype(np.float32)
    losses = score_latents(net, library_z, alpha_bar, device)
    loss_by_name = {Path(path).name: float(loss) for path, loss in zip(paths, losses)}
    test_loss = np.array([loss_by_name[knn.rows[int(i)]["文件"]] for i in test])
    val_loss = np.array([loss_by_name[knn.rows[int(i)]["文件"]] for i in validation])
    favorite_names = {row["文件"] for row in knn.rows}
    unlabeled_loss = np.array([loss for path, loss in zip(paths, losses)
                               if Path(path).name not in favorite_names])
    # These are retrieval diagnostics with unlabeled songs as a comparison
    # group, not a valid dislike-classification test.
    proxy_auc = float(roc_auc_score(
        np.r_[np.ones(len(test_loss)), np.zeros(len(unlabeled_loss))],
        -np.r_[test_loss, unlabeled_loss]))
    training_vectors = x[train]
    test_cos = (x[test] @ training_vectors.T).max(axis=1)
    unlabeled_vectors = np.array([v for p, v in zip(paths, library_vectors)
                                  if Path(p).name not in favorite_names], dtype=np.float32)
    unlabel_cos = (unlabeled_vectors @ training_vectors.T).max(axis=1)
    cosine_proxy_auc = float(roc_auc_score(
        np.r_[np.ones(len(test_cos)), np.zeros(len(unlabel_cos))],
        np.r_[test_cos, unlabel_cos]))
    # Higher score = lower loss than more held-out favorite references.
    sorted_reference = np.sort(val_loss)
    scores = (len(sorted_reference) - np.searchsorted(sorted_reference, losses, side="left") +
              len(sorted_reference) - np.searchsorted(sorted_reference, losses, side="right")) * 50 / len(sorted_reference)
    rows = sorted(zip(paths, losses, scores), key=lambda row: (row[1], Path(row[0]).name))
    split_by_file = {knn.rows[int(i)]["文件"]: subset for subset, part in
                     (("train", train), ("validation", validation), ("test", test)) for i in part}
    with (out / "library_diffusion_ranking.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "favorite_reference_percentile", "denoising_mse",
                         "favorite_split", "file", "audio_path"])
        for rank, (path, loss, score) in enumerate(rows, 1):
            writer.writerow([rank, float(score), float(loss),
                             split_by_file.get(Path(path).name, "unlabeled"), Path(path).name, path])
    metrics = {"model": ("weighted DDPM on native coordinates with train-only per-axis standardization" if args.no_projection else "weighted latent DDPM epsilon-prediction on PCA-whitened MS-CLAP embeddings"),
               "no_projection": args.no_projection,
               "preprocess_id": PREPROCESS_ID, "vector_cache": str(default_vector_cache().resolve()),
               "scoring_noise_protocol": "shared Gaussian across songs; 32 repeats at t=5/15/30/50; seed=2026",
               "favorite_definition": "All labeled songs are positive; uncollected songs are unlabeled",
               "weight_usage": "p_i = w_i / sum_train(w), weighted sampling with replacement",
               "probability_warning": "p_i is training sampling probability, not P(like|song); output is percentile, not probability",
               "train_favorites": len(train), "validation_favorites": len(validation),
               "test_favorites": len(test), "unlabeled_library": len(unlabeled_loss),
               "batch_size": args.batch_size, "initial_lr": args.lr,
               "latent_dim": args.latent_dim, "diffusion_steps": len(alpha_bar),
               "best_epoch": best_epoch, "best_validation_loss": best_loss,
               "effective_sampling_n": effective_samples,
               "heldout_favorite_denoising_mse_median": float(np.median(test_loss)),
               "unlabeled_denoising_mse_median": float(np.median(unlabeled_loss)),
               "proxy_auc_diffusion": proxy_auc, "proxy_auc_nearest_cosine": cosine_proxy_auc,
               "proxy_auc_warning": "Unlabeled library songs can be liked; AUC is only a retrieval proxy, not dislike accuracy",
               "knn_model_dir": str(knn.directory), "seed": args.seed}
    write_json(out / "metrics.json", metrics)
    write_json(out / "split.json", {name: [knn.rows[int(i)]["文件"] for i in part]
                                    for name, part in (("train", train), ("validation", validation),
                                                       ("test", test))})
    write_json(ROOT / "models_archive/diffusion_preference/latest.json", {"model_dir": str(out.resolve())})
    log(f"Held-out favorite vs unlabeled proxy AUC: diffusion={proxy_auc:.4f}; "
          f"nearest cosine={cosine_proxy_auc:.4f}", flush=True)
    log(f"Saved diffusion model and candidate ranking: {out}", flush=True)
    log_stream.close()


if __name__ == "__main__":
    main()
