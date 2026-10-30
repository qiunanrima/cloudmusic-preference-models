"""Compare personal and background denoisers using paired noise.

This is an experimental ranking proxy, not a calibrated likelihood ratio or
probability of liking. The background is trained on unlabeled songs, which may
include future favorites. All original models and rankings are left intact.
"""
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
from sklearn.metrics import roc_auc_score
import torch

from msclap_audio import PREPROCESS_ID, default_vector_cache, read_vectors, unit_rows
from preference_data import ROOT, read_json, write_json
from train_conditional_diffusion import (ConditionalDenoiser, MAX_TAG_VALUES,
                                         NUMERIC_FIELDS, TAG_FIELDS, losses,
                                         noise_schedule, score_vectors)
from preference_knn import UnionFind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--personal", type=Path,
                        default=Path(read_json(ROOT / "models_archive/conditional_diffusion/latest.json")["model_dir"]))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--epochs", type=int, default=360)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    personal = args.personal.resolve()
    out = args.out or ROOT / "models_archive/diffusion_background" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    log_file = (out / "training.log").open("w", encoding="utf-8", buffering=1)

    def log(message):
        print(message, flush=True)
        log_file.write(message + "\n")

    features = joblib.load(personal / "features.joblib")
    paths, raw = read_vectors(default_vector_cache())
    vectors = unit_rows(raw)
    normalized = ((vectors - features["mean"]) / features["std"]).astype(np.float32)
    names = np.array([Path(path).name for path in paths])
    split = read_json(personal / "split.json")
    personal_train = set(split["train"])
    personal_validation = set(split["validation"])
    personal_test = set(split["test"])
    favorites = personal_train | personal_validation | personal_test
    name_to_index = {name: i for i, name in enumerate(names)}
    if not personal_test.issubset(name_to_index) or not personal_validation.issubset(name_to_index):
        raise ValueError("Personal split does not match the vector cache")
    positive_idx = np.array([name_to_index[name] for name in favorites], dtype=np.int64)
    unlabeled_all = np.flatnonzero(~np.isin(names, list(favorites)))
    # Exclude alternate encodings of any known favorite from background
    # training and AUC negatives. They remain scored in the full ranking.
    near_favorite = np.zeros(len(unlabeled_all), dtype=bool)
    for start in range(0, len(unlabeled_all), 128):
        block = unlabeled_all[start:start+128]
        near_favorite[start:start+len(block)] = (
            vectors[block] @ vectors[positive_idx].T).max(axis=1) > .995
    unlabeled = unlabeled_all[~near_favorite]
    # Group near-identical recordings before splitting so alternate encodings
    # cannot land in both background train and held-out evaluation.
    eligible_vectors = vectors[unlabeled]
    background_union = UnionFind(len(unlabeled))
    for start in range(0, len(unlabeled), 128):
        stop = min(start + 128, len(unlabeled))
        local_sim = eligible_vectors[start:stop] @ eligible_vectors.T
        rr, cc = np.where(local_sim > .995)
        for r, c in zip(rr.tolist(), cc.tolist()):
            left, right = start + r, c
            if left < right:
                background_union.join(left, right)
    components = {}
    for local_idx in range(len(unlabeled)):
        components.setdefault(background_union.root(local_idx), []).append(int(unlabeled[local_idx]))
    rng = np.random.default_rng(args.seed)
    component_keys = list(components)
    rng.shuffle(component_keys)
    background_train, background_validation, background_test = [], [], []
    for key in component_keys:
        group = components[key]
        if len(background_train) < len(personal_train):
            background_train.extend(group)
        elif len(background_validation) < len(personal_validation):
            background_validation.extend(group)
        else:
            background_test.extend(group)
    if len(background_train) < len(personal_train) or len(background_validation) < len(personal_validation):
        raise ValueError("Insufficient eligible background songs for grouped matched-size split")
    background_train = np.asarray(background_train, dtype=np.int64)
    background_validation = np.asarray(background_validation, dtype=np.int64)
    background_test = np.asarray(background_test, dtype=np.int64)
    from msclap_audio import cache_protocol
    fixed = cache_protocol().get('fixed_background_split')
    if fixed:
        original = read_json(fixed)
        eligible = set(unlabeled.tolist())
        def fixed_indices(key):
            return np.array([name_to_index[n] for n in original[key]
                             if name_to_index[n] in eligible], dtype=np.int64)
        background_train = fixed_indices('background_train')
        background_validation = fixed_indices('background_validation')
        background_test = fixed_indices('background_test')
        # New near duplicates crossing the fixed boundary are omitted from
        # held-out evaluation, never silently moved into a training set.
        def disjoint_audio(candidates, seen):
            keep = (vectors[candidates] @ vectors[seen].T).max(axis=1) <= .995
            return candidates[keep]
        background_validation = disjoint_audio(background_validation, background_train)
        background_test = disjoint_audio(background_test, np.r_[background_train, background_validation])
        if min(len(background_train), len(background_validation), len(background_test)) == 0:
            raise ValueError('Fixed background split became empty')
    write_json(out / "split.json", {
        "personal_model": str(personal),
        "background_train": names[background_train].tolist(),
        "background_validation": names[background_validation].tolist(),
        "background_test": names[background_test].tolist(),
        "favorite_test": sorted(personal_test),
        "excluded_near_favorite": names[unlabeled_all[near_favorite]].tolist(),
        "background_recording_groups": len(components),
        "duplicate_similarity_threshold": .995})

    checkpoint = torch.load(personal / "model.pt", map_location="cpu", weights_only=True)
    sizes = checkpoint["vocabulary_sizes"]
    background = ConditionalDenoiser(sizes).to(device)
    personal_net = ConditionalDenoiser(sizes).to(device)
    personal_net.load_state_dict(checkpoint["state_dict"])
    personal_net.eval()
    alpha_bar = noise_schedule(checkpoint["steps"], device)
    optimizer = torch.optim.AdamW(background.parameters(), lr=args.lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=7, min_lr=1e-5)
    val_rng = np.random.default_rng(args.seed + 1000)
    val_x = torch.from_numpy(normalized[background_validation]).to(device)
    val_t = torch.from_numpy(val_rng.integers(0, len(alpha_bar), len(val_x), dtype=np.int64)).to(device)
    val_eps = torch.from_numpy(val_rng.standard_normal(val_x.shape, dtype=np.float32)).to(device)
    zeros_tags = torch.zeros((len(val_x), len(TAG_FIELDS), MAX_TAG_VALUES), device=device, dtype=torch.long)
    zeros_numbers = torch.zeros((len(val_x), len(NUMERIC_FIELDS)), device=device)
    zeros_mask = torch.zeros_like(zeros_numbers)
    best_loss = float("inf")
    best_epoch = 0
    history = []
    log(f"device={device}; train={len(background_train)}; validation={len(background_validation)}; "
        f"test={len(background_test)}; batch_size={args.batch_size}; lr={args.lr}; "
        f"epochs={args.epochs}; parameters={sum(p.numel() for p in background.parameters())}")
    for epoch in range(1, args.epochs + 1):
        background.train()
        indices = rng.permutation(background_train)
        total = 0.
        for start in range(0, len(indices), args.batch_size):
            selection = indices[start:start + args.batch_size]
            clean = torch.from_numpy(normalized[selection]).to(device)
            t = torch.randint(len(alpha_bar), (len(clean),), device=device)
            eps = torch.randn_like(clean)
            tags = torch.zeros((len(clean), len(TAG_FIELDS), MAX_TAG_VALUES), device=device, dtype=torch.long)
            numbers = torch.zeros((len(clean), len(NUMERIC_FIELDS)), device=device)
            mask = torch.zeros_like(numbers)
            optimizer.zero_grad(set_to_none=True)
            loss = losses(background, clean, t, eps, alpha_bar, tags, numbers, mask).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(selection)
        background.eval()
        with torch.no_grad():
            val_loss = float(losses(background, val_x, val_t, val_eps, alpha_bar,
                                    zeros_tags, zeros_numbers, zeros_mask).mean().item())
        lr = optimizer.param_groups[0]["lr"]
        train_loss = total / len(background_train)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "validation_loss": val_loss, "lr": lr})
        log(f"epoch {epoch:03d} train={train_loss:.6f} val={val_loss:.6f} lr={lr:.8f}")
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch = val_loss, epoch
            torch.save({"state_dict": background.state_dict(), "vocabulary_sizes": sizes,
                        "steps": len(alpha_bar), "seed": args.seed}, out / "model.pt")
        scheduler.step(val_loss)

    with (out / "loss.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="Train")
    ax.plot([h["epoch"] for h in history], [h["validation_loss"] for h in history], label="Validation")
    ax.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"Best epoch {best_epoch}")
    ax.set(xlabel="Epoch", ylabel="Noise-prediction MSE", title="Unconditional background diffusion")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=160)
    ax.step([h["epoch"] for h in history], [h["lr"] for h in history], where="post")
    ax.set(xlabel="Epoch", ylabel="Learning rate", title="Background diffusion learning rate")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "learning_rate.png")
    plt.close(fig)

    best = torch.load(out / "model.pt", map_location=device, weights_only=True)
    background.load_state_dict(best["state_dict"])
    personal_loss = score_vectors(personal_net, normalized, alpha_bar, device,
                                  repeats=args.repeats)
    background_loss = score_vectors(background, normalized, alpha_bar, device,
                                    repeats=args.repeats)
    gain = background_loss - personal_loss
    positive = np.array([name_to_index[name] for name in sorted(personal_test)])
    evaluation = np.r_[positive, background_test]
    targets = np.r_[np.ones(len(positive)), np.zeros(len(background_test))]
    auc_raw = float(roc_auc_score(targets, -personal_loss[evaluation]))
    auc_relative = float(roc_auc_score(targets, gain[evaluation]))
    centroid = vectors[background_train].mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    centrality = vectors @ centroid
    corr_raw = float(spearmanr(centrality, -personal_loss).statistic)
    corr_relative = float(spearmanr(centrality, gain).statistic)
    ref = np.sort(gain[np.array([name_to_index[name] for name in personal_validation])])
    percentile = np.searchsorted(ref, gain, side="right") * 100 / len(ref)
    background_split = np.full(len(paths), "favorite", dtype=object)
    background_split[unlabeled_all[near_favorite]] = "duplicate_of_favorite_excluded"
    background_split[unlabeled] = "background_excluded_fixed_boundary" if fixed else "background_test"
    background_split[background_train] = "background_train"
    background_split[background_validation] = "background_validation"
    background_split[background_test] = "background_test"
    favorite_split = np.full(len(paths), "unlabeled", dtype=object)
    for label, members in split.items():
        favorite_split[[name_to_index[name] for name in members]] = label
    order = np.argsort(-gain, kind="stable")
    with (out / "relative_ranking.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["rank", "favorite_reference_percentile", "relative_denoising_gain",
                         "personal_mse", "background_mse", "favorite_split", "background_split",
                         "file", "audio_path"])
        for rank, index in enumerate(order, 1):
            writer.writerow([rank, float(percentile[index]), float(gain[index]),
                             float(personal_loss[index]), float(background_loss[index]),
                             favorite_split[index], background_split[index], names[index], paths[index]])
    with (out / "candidate_recommendations_heldout.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["rank", "favorite_reference_percentile", "relative_denoising_gain",
                         "personal_mse", "background_mse", "file", "audio_path"])
        for rank, index in enumerate([i for i in order if background_split[i] == "background_test"], 1):
            writer.writerow([rank, float(percentile[index]), float(gain[index]),
                             float(personal_loss[index]), float(background_loss[index]), names[index], paths[index]])
    result = {
        "personal_model": str(personal), "background_model": str(out / "model.pt"),
        "preprocess_id": PREPROCESS_ID,
        "vector_cache": str(default_vector_cache().resolve()),
        "background_train": len(background_train), "background_validation": len(background_validation),
        "background_test": len(background_test), "favorite_test": len(positive),
        "background_recording_groups": len(components),
        "background_excluded_near_favorite": int(near_favorite.sum()),
        "duplicate_similarity_threshold": .995,
        "epochs": args.epochs, "batch_size": args.batch_size, "initial_lr": args.lr,
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "paired_noise_repeats": args.repeats, "timesteps": [5, 15, 30, 50],
        "scoring_noise_protocol": "shared standard Gaussian per timestep/repeat across all songs and both models; seed=2026",
        "proxy_auc_raw_personal_mse": auc_raw, "proxy_auc_relative_gain": auc_relative,
        "spearman_centroid_raw_score": corr_raw, "spearman_centroid_relative_gain": corr_relative,
        "warning": "Unlabeled songs are not confirmed dislikes; gain is not a calibrated likelihood ratio or P(like).",
    }
    write_json(out / "metrics.json", result)
    write_json(ROOT / "models_archive/diffusion_background/latest.json", {"model_dir": str(out.resolve())})
    log(f"Heldout proxy AUC: raw={auc_raw:.4f}, relative={auc_relative:.4f}; "
        f"centroid correlation: raw={corr_raw:.4f}, relative={corr_relative:.4f}")
    log(f"Saved experiment: {out}")
    log_file.close()


if __name__ == "__main__":
    main()
