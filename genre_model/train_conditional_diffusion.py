"""Direct-1024 MS-CLAP FiLM diffusion with optional metadata conditioning.

Favorite weights define training sampling probabilities only. Inference uses
the unconditional branch for every library candidate so scores are comparable.
The resulting percentile is not a calibrated probability of liking a song.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import random
import shutil

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch import nn
from torch.nn import functional as F

from msclap_audio import (DIM, PREPROCESS_ID, default_vector_cache,
                          read_vectors, unit_rows)
from preference_data import ROOT, read_json, write_json
from preference_knn import (PreferenceModel, TAG_WEIGHTS, NUMERIC_WEIGHTS,
                            artist_groups, confidence, model_directory, tag_values)


TAG_FIELDS = list(TAG_WEIGHTS)
NUMERIC_FIELDS = list(NUMERIC_WEIGHTS)
MAX_TAG_VALUES = 4


def prepare_conditions(rows, training_indices, vocabularies=None, numeric_stats=None):
    if vocabularies is None:
        vocabularies = []
        for field in TAG_FIELDS:
            values = sorted({v for i in training_indices
                             for v in tag_values(rows[int(i)], field)
                             if confidence(rows[int(i)], field) > 0})
            vocabularies.append({v: j + 1 for j, v in enumerate(values)})
    tags = np.zeros((len(rows), len(TAG_FIELDS), MAX_TAG_VALUES), dtype=np.int64)
    for i, row in enumerate(rows):
        for j, (field, vocab) in enumerate(zip(TAG_FIELDS, vocabularies)):
            if confidence(row, field) <= 0:
                continue
            values = [vocab[v] for v in sorted(tag_values(row, field)) if v in vocab]
            tags[i, j, :min(len(values), MAX_TAG_VALUES)] = values[:MAX_TAG_VALUES]
    raw = np.zeros((len(rows), len(NUMERIC_FIELDS)), dtype=np.float32)
    mask = np.zeros_like(raw)
    for i, row in enumerate(rows):
        for j, field in enumerate(NUMERIC_FIELDS):
            value = row.get(field)
            if isinstance(value, (int, float)) and np.isfinite(value) and value > 0:
                raw[i, j] = np.log(value) if field != NUMERIC_FIELDS[1] else value
                mask[i, j] = 1
    if numeric_stats is None:
        means, stds = [], []
        for j in range(len(NUMERIC_FIELDS)):
            selected = raw[training_indices, j][mask[training_indices, j] > 0]
            means.append(float(selected.mean()) if len(selected) else 0.)
            stds.append(max(float(selected.std()), 1e-3) if len(selected) else 1.)
        numeric_stats = {"mean": means, "std": stds}
    numbers = ((raw - np.asarray(numeric_stats["mean"])) /
               np.asarray(numeric_stats["std"])).astype(np.float32) * mask
    return tags, numbers, mask, vocabularies, numeric_stats


class ConditionEncoder(nn.Module):
    def __init__(self, vocabulary_sizes, width=256):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(size + 1, 32)
                                         for size in vocabulary_sizes])
        self.project = nn.Sequential(nn.Linear(len(vocabulary_sizes) * 32 + 6, width),
                                     nn.GELU(), nn.Linear(width, width))

    def forward(self, tags, numeric, mask):
        pieces = []
        for field, embedding in enumerate(self.embeddings):
            indices = tags[:, field]
            valid = indices != 0
            embedded = embedding(indices)
            counts = valid.sum(dim=1, keepdim=True)
            pooled = (embedded * valid[..., None]).sum(dim=1) / counts.clamp(min=1)
            pooled = torch.where(counts > 0, pooled, embedding.weight[0][None])
            pieces.append(pooled)
        return self.project(torch.cat([*pieces, numeric, mask], dim=1))


class FiLMBlock(nn.Module):
    def __init__(self, width=512):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.film = nn.Linear(256, width * 2)
        self.layers = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                    nn.Dropout(.1), nn.Linear(width, width))

    def forward(self, hidden, context):
        scale, shift = self.film(context).chunk(2, dim=1)
        transformed = self.norm(hidden) * (1 + scale) + shift
        return hidden + .2 * self.layers(transformed)


class ConditionalDenoiser(nn.Module):
    def __init__(self, vocabulary_sizes, dimension=DIM):
        super().__init__()
        self.conditions = ConditionEncoder(vocabulary_sizes)
        self.time = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 256))
        self.input = nn.Linear(dimension, 512)
        self.blocks = nn.ModuleList([FiLMBlock() for _ in range(3)])
        self.output = nn.Sequential(nn.LayerNorm(512), nn.GELU(), nn.Linear(512, dimension))

    def forward(self, noisy, t, tags, numeric, mask):
        frequencies = torch.exp(torch.linspace(0, -math.log(10000), 32, device=noisy.device))
        angles = t.float()[:, None] * frequencies[None]
        time_embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        context = self.time(time_embedding) + self.conditions(tags, numeric, mask)
        hidden = self.input(noisy)
        for block in self.blocks:
            hidden = block(hidden, context)
        return self.output(hidden)


def drop_conditions(tags, numeric, mask, field_probability=.15, unconditional_probability=.15):
    tags, numeric, mask = tags.clone(), numeric.clone(), mask.clone()
    field_drop = torch.rand(tags.shape[:2], device=tags.device) < field_probability
    tags[field_drop] = 0
    number_drop = torch.rand(mask.shape, device=mask.device) < field_probability
    mask[number_drop] = 0
    numeric[number_drop] = 0
    all_drop = torch.rand(len(tags), device=tags.device) < unconditional_probability
    tags[all_drop] = 0
    numeric[all_drop] = 0
    mask[all_drop] = 0
    return tags, numeric, mask


def noise_schedule(steps, device):
    betas = torch.linspace(1e-4, .02, steps, device=device)
    return torch.cumprod(1 - betas, dim=0)


def losses(net, clean, timesteps, noise, alpha_bar, tags, numeric, mask):
    alpha = alpha_bar[timesteps][:, None]
    noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
    prediction = net(noisy, timesteps, tags, numeric, mask)
    return (prediction - noise).square().mean(dim=1)


@torch.no_grad()
def score_vectors(net, vectors, alpha_bar, device, batch_size=256,
                  timesteps=(5, 15, 30, 50), repeats=3, seed=2026):
    """Unconditional score with shared noise across songs and paired models.

    Common random numbers keep the finite-sample score smooth in the input
    vector, so alternate encodings do not receive unrelated Monte Carlo noise.
    The result is independent of query order and scoring batch size.
    """
    net.eval()
    result = np.zeros(len(vectors), dtype=np.float64)
    for t_value in timesteps:
        for repeat in range(repeats):
            noise = np.random.default_rng(np.random.SeedSequence(
                [seed, int(t_value), repeat])).standard_normal(
                    vectors.shape[1], dtype=np.float32)
            for start in range(0, len(vectors), batch_size):
                stop = min(start + batch_size, len(vectors))
                clean = torch.from_numpy(vectors[start:stop]).to(device)
                eps = torch.from_numpy(noise).to(device).expand(stop - start, -1)
                t = torch.full((stop - start,), t_value, device=device, dtype=torch.long)
                tags = torch.zeros((stop - start, len(TAG_FIELDS), MAX_TAG_VALUES),
                                   device=device, dtype=torch.long)
                numbers = torch.zeros((stop - start, len(NUMERIC_FIELDS)), device=device)
                mask = torch.zeros_like(numbers)
                result[start:stop] += losses(net, clean, t, eps, alpha_bar,
                                              tags, numbers, mask).cpu().numpy()
    return result / (len(timesteps) * repeats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", type=Path, help="Continue from a prior conditional diffusion model directory")
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    knn = PreferenceModel(model_directory(args.model))
    vectors = unit_rows(knn.vectors)
    groups = artist_groups(knn.rows, knn.recordings)
    indices = np.arange(len(vectors))
    development, test = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                                                random_state=args.seed).split(indices, groups=groups))
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                                     random_state=args.seed + 1).split(
                                         development, groups=groups[development]))
    train, validation = development[tr], development[va]
    resume = args.resume.resolve() if args.resume else None
    if resume:
        previous_split = read_json(resume / "split.json")
        for subset, part in (("train", train), ("validation", validation), ("test", test)):
            expected = [knn.rows[int(i)]["文件"] for i in part]
            if previous_split[subset] != expected:
                raise ValueError(f"Resume split mismatch: {subset}")
        previous_metrics = read_json(resume / "metrics.json")
        if previous_metrics["knn_model_dir"] != str(knn.directory):
            raise ValueError("Resume KNN model mismatch")
        with (resume / "loss.csv").open(encoding="utf-8-sig", newline="") as f:
            history = [{key: int(value) if key == "epoch" else float(value)
                        for key, value in row.items()} for row in csv.DictReader(f)]
        start_epoch = history[-1]["epoch"] + 1
    else:
        history, start_epoch = [], 1
    out = args.out or ROOT / "models_archive/conditional_diffusion" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    log_stream = (out / "training.log").open("w", encoding="utf-8", buffering=1)
    if resume:
        log_stream.write((resume / "training.log").read_text(encoding="utf-8"))
    def log(message):
        print(message, flush=True)
        log_stream.write(message + "\n")

    if resume:
        features = joblib.load(resume / "features.joblib")
        mean, std = features["mean"], features["std"]
        vocabularies, numeric_stats = features["vocabularies"], features["numeric_stats"]
        shutil.copyfile(resume / "features.joblib", out / "features.joblib")
        tags, numbers, masks, _, _ = prepare_conditions(
            knn.rows, train, vocabularies, numeric_stats)
    else:
        mean = vectors[train].mean(axis=0).astype(np.float32)
        raw_std = vectors[train].std(axis=0)
        std = np.maximum(raw_std, .5 * np.median(raw_std)).astype(np.float32)
        tags, numbers, masks, vocabularies, numeric_stats = prepare_conditions(knn.rows, train)
        joblib.dump({"mean": mean, "std": std, "vocabularies": vocabularies,
                     "numeric_stats": numeric_stats}, out / "features.joblib", compress=3)
    normalized = ((vectors - mean) / std).astype(np.float32)
    probabilities = knn.weights[train] / knn.weights[train].sum()
    effective_n = float(1 / np.sum(probabilities**2))
    net = ConditionalDenoiser([len(v) for v in vocabularies]).to(device)
    alpha_bar = noise_schedule(100, device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=7, min_lr=1e-5)
    if resume:
        previous_checkpoint = torch.load(resume / "model.pt", map_location=device, weights_only=True)
        net.load_state_dict(previous_checkpoint["state_dict"])
        if "optimizer" in previous_checkpoint:
            optimizer.load_state_dict(previous_checkpoint["optimizer"])
            scheduler.load_state_dict(previous_checkpoint["scheduler"])
            optimizer_restored = True
        else:
            optimizer_restored = False
        shutil.copyfile(resume / "model.pt", out / "model.pt")
        torch.manual_seed(args.seed + start_epoch)
    else:
        optimizer_restored = False
    rng = np.random.default_rng(args.seed if not resume else args.seed + start_epoch)
    val_rng = np.random.default_rng(args.seed + 1000)
    val_x = torch.from_numpy(normalized[validation]).to(device)
    val_t = torch.from_numpy(val_rng.integers(0, len(alpha_bar), len(validation),
                                               dtype=np.int64)).to(device)
    val_eps = torch.from_numpy(val_rng.standard_normal(val_x.shape, dtype=np.float32)).to(device)
    val_tags = torch.from_numpy(tags[validation]).to(device)
    val_numbers = torch.from_numpy(numbers[validation]).to(device)
    val_masks = torch.from_numpy(masks[validation]).to(device)
    best_row = min(history, key=lambda row: row["validation_mean_loss"]) if history else None
    best_loss = best_row["validation_mean_loss"] if best_row else float("inf")
    best_epoch = best_row["epoch"] if best_row else 0
    stagnant = 0
    if resume:
        log(f"resuming_from={resume}; additional_epochs={args.epochs}; "
            f"optimizer_state={'restored' if optimizer_restored else 'reset (not saved in prior checkpoint)'}")
    log(f"device={device}; train={len(train)}, validation={len(validation)}, test={len(test)}; "
        f"direct_dim={DIM}; batch_size={args.batch_size}; lr={args.lr}; "
        f"parameters={sum(p.numel() for p in net.parameters())}; "
            f"weighted_sampling_effective_n={effective_n:.1f}; "
            f"preprocess_id={PREPROCESS_ID}; vector_cache={default_vector_cache()}")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        net.train()
        picks = rng.choice(len(train), size=len(train), replace=True, p=probabilities)
        total = 0.
        for start in range(0, len(picks), args.batch_size):
            selected = train[picks[start:start + args.batch_size]]
            clean = torch.from_numpy(normalized[selected]).to(device)
            t = torch.randint(len(alpha_bar), (len(clean),), device=device)
            eps = torch.randn_like(clean)
            batch_tags = torch.from_numpy(tags[selected]).to(device)
            batch_numbers = torch.from_numpy(numbers[selected]).to(device)
            batch_masks = torch.from_numpy(masks[selected]).to(device)
            batch_tags, batch_numbers, batch_masks = drop_conditions(
                batch_tags, batch_numbers, batch_masks)
            optimizer.zero_grad(set_to_none=True)
            loss = losses(net, clean, t, eps, alpha_bar,
                          batch_tags, batch_numbers, batch_masks).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(clean)
        train_loss = total / len(train)
        net.eval()
        with torch.no_grad():
            conditional = losses(net, val_x, val_t, val_eps, alpha_bar,
                                  val_tags, val_numbers, val_masks).mean()
            unconditional = losses(net, val_x, val_t, val_eps, alpha_bar,
                                    torch.zeros_like(val_tags), torch.zeros_like(val_numbers),
                                    torch.zeros_like(val_masks)).mean()
            val_cond, val_uncond = float(conditional.item()), float(unconditional.item())
        val_loss = (val_cond + val_uncond) / 2
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "validation_conditional_loss": val_cond,
                        "validation_unconditional_loss": val_uncond,
                        "validation_mean_loss": val_loss, "lr": lr})
        log(f"epoch {epoch:03d} train={train_loss:.6f} "
            f"val_cond={val_cond:.6f} val_uncond={val_uncond:.6f} lr={lr:.8f}")
        improved = val_loss < best_loss - 1e-5
        if improved:
            best_loss, best_epoch, stagnant = val_loss, epoch, 0
        else:
            stagnant += 1
        scheduler.step(val_loss)
        if improved:
            torch.save({"state_dict": net.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "vocabulary_sizes": [len(v) for v in vocabularies],
                        "steps": len(alpha_bar), "seed": args.seed,
                        "preprocess_id": PREPROCESS_ID}, out / "model.pt")
        if stagnant >= args.patience:
            log(f"early stopping at epoch {epoch}; best={best_epoch}")
            break
    checkpoint = torch.load(out / "model.pt", map_location=device, weights_only=True)
    net.load_state_dict(checkpoint["state_dict"])
    with (out / "loss.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    for key, label in (("train_loss", "Train"),
                       ("validation_conditional_loss", "Val conditioned"),
                       ("validation_unconditional_loss", "Val unconditioned")):
        ax.plot([h["epoch"] for h in history], [h[key] for h in history], label=label)
    if resume:
        ax.axvline(start_epoch - 1, color="purple", ls=":", lw=1,
                   label=f"Resume after epoch {start_epoch - 1}")
    ax.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"Best epoch {best_epoch}")
    ax.set(xlabel="Epoch", ylabel="Noise-prediction MSE", title="Conditional FiLM diffusion")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=160)
    ax.step([h["epoch"] for h in history], [h["lr"] for h in history], where="post")
    if resume:
        ax.axvline(start_epoch - 1, color="purple", ls=":", lw=1,
                   label=f"Optimizer reset after epoch {start_epoch - 1}" if not optimizer_restored
                   else f"Resume after epoch {start_epoch - 1}")
        ax.legend()
    ax.set(xlabel="Epoch", ylabel="Learning rate", title="Conditional diffusion learning rate")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "learning_rate.png")
    plt.close(fig)

    paths, library_vectors = read_vectors(default_vector_cache())
    library_x = ((unit_rows(library_vectors) - mean) / std).astype(np.float32)
    denoise = score_vectors(net, library_x, alpha_bar, device)
    by_name = {Path(path).name: float(value) for path, value in zip(paths, denoise)}
    test_loss = np.array([by_name[knn.rows[int(i)]["文件"]] for i in test])
    val_loss = np.array([by_name[knn.rows[int(i)]["文件"]] for i in validation])
    favorite_names = {row["文件"] for row in knn.rows}
    unlabeled = np.array([value for path, value in zip(paths, denoise)
                          if Path(path).name not in favorite_names])
    proxy_auc = float(roc_auc_score(
        np.r_[np.ones(len(test_loss)), np.zeros(len(unlabeled))],
        -np.r_[test_loss, unlabeled]))
    train_vectors = vectors[train]
    test_cos = (vectors[test] @ train_vectors.T).max(axis=1)
    unlabel_vectors = np.array([v for path, v in zip(paths, library_vectors)
                                if Path(path).name not in favorite_names], dtype=np.float32)
    unlabel_cos = (unlabel_vectors @ train_vectors.T).max(axis=1)
    cosine_auc = float(roc_auc_score(
        np.r_[np.ones(len(test_cos)), np.zeros(len(unlabel_cos))],
        np.r_[test_cos, unlabel_cos]))
    reference = np.sort(val_loss)
    scores = (len(reference) - np.searchsorted(reference, denoise, side="left") +
              len(reference) - np.searchsorted(reference, denoise, side="right")) * 50 / len(reference)
    # The 210-song reference ECDF has many ties (including many 100s); rank
    # by the continuous denoising error, not filename within a tied percentile.
    rows = sorted(zip(paths, denoise, scores), key=lambda item: (item[1], Path(item[0]).name))
    split_by_file = {knn.rows[int(i)]["文件"]: subset for subset, part in
                     (("train", train), ("validation", validation), ("test", test)) for i in part}
    with (out / "library_diffusion_ranking.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "favorite_reference_percentile", "denoising_mse",
                         "favorite_split", "file", "audio_path"])
        for rank, (path, loss, score) in enumerate(rows, 1):
            writer.writerow([rank, float(score), float(loss),
                             split_by_file.get(Path(path).name, "unlabeled"), Path(path).name, path])
    with (out / "candidate_recommendations.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "favorite_reference_percentile", "denoising_mse", "file", "audio_path"])
        candidates = [(path, loss, score) for path, loss, score in rows
                      if Path(path).name not in favorite_names]
        for rank, (path, loss, score) in enumerate(candidates, 1):
            writer.writerow([rank, float(score), float(loss), Path(path).name, path])
    result = {"model": f"Direct {DIM}-dimensional FiLM residual MLP diffusion with learned tag embeddings",
              "input_dimension": DIM,
              "preprocess_id": PREPROCESS_ID,
              "vector_cache": str(default_vector_cache().resolve()),
              "feature_transform": "Per-dimension standardization on training favorites; no PCA",
              "conditioning": "8 categorical fields + 3 numeric fields; null embeddings; 15% field dropout and 15% full-condition dropout",
              "weight_usage": "p_i = w_i / sum_train(w), weighted sampling with replacement",
              "probability_warning": "Sampling probabilities and denoising percentiles are not P(like|song)",
              "ranking_mode": "Unconditional denoising for all candidates; held-out favorite ECDF",
              "train_favorites": len(train), "validation_favorites": len(validation),
              "test_favorites": len(test), "unlabeled_library": len(unlabeled),
              "batch_size": args.batch_size, "initial_lr": args.lr,
              "parameters": sum(p.numel() for p in net.parameters()),
              "best_epoch": best_epoch, "best_validation_loss": best_loss,
              "resumed_from": str(resume) if resume else None,
              "resume_optimizer_state": "restored" if optimizer_restored else "reset" if resume else "new",
              "effective_sampling_n": effective_n,
              "heldout_favorite_denoising_mse_median": float(np.median(test_loss)),
              "unlabeled_denoising_mse_median": float(np.median(unlabeled)),
              "proxy_auc_diffusion": proxy_auc, "proxy_auc_nearest_cosine": cosine_auc,
              "proxy_auc_warning": "Unlabeled songs may be liked; this is a retrieval proxy, not dislike accuracy",
              "knn_model_dir": str(knn.directory), "seed": args.seed}
    write_json(out / "metrics.json", result)
    write_json(out / "split.json", {name: [knn.rows[int(i)]["文件"] for i in part]
                                    for name, part in (("train", train), ("validation", validation),
                                                       ("test", test))})
    write_json(ROOT / "models_archive/conditional_diffusion/latest.json", {"model_dir": str(out.resolve())})
    log(f"Held-out favorite vs unlabeled proxy AUC: conditional FiLM={proxy_auc:.4f}; "
        f"nearest cosine={cosine_auc:.4f}")
    log(f"Saved conditional diffusion model and ranking: {out}")
    log_stream.close()


if __name__ == "__main__":
    main()
