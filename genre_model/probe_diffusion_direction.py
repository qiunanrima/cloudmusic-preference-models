"""Compare single-step directional fidelity with denoising gain on the same holdout."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional as F

from msclap_audio import default_vector_cache, read_vectors, unit_rows
from preference_data import read_json, write_json
from train_conditional_diffusion import (ConditionalDenoiser, MAX_TAG_VALUES,
                                         NUMERIC_FIELDS, TAG_FIELDS, noise_schedule)


def probe(net, vectors, alpha_bar, device, *, timestep=25, repeats=8, seed=2026):
    net.eval()
    mse = np.zeros(len(vectors), dtype=np.float64)
    cosine = np.zeros(len(vectors), dtype=np.float64)
    alpha = alpha_bar[timestep]
    with torch.no_grad():
        for repeat in range(repeats):
            eps_all = np.random.default_rng(np.random.SeedSequence(
                [seed, timestep, repeat])).standard_normal(vectors.shape[1], dtype=np.float32)
            for start in range(0, len(vectors), 256):
                end = min(start + 256, len(vectors))
                clean = torch.from_numpy(vectors[start:end]).to(device)
                eps = torch.from_numpy(eps_all).to(device).expand(end - start, -1)
                noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * eps
                t = torch.full((len(clean),), timestep, device=device, dtype=torch.long)
                tags = torch.zeros((len(clean), len(TAG_FIELDS), MAX_TAG_VALUES), device=device, dtype=torch.long)
                numbers = torch.zeros((len(clean), len(NUMERIC_FIELDS)), device=device)
                prediction = net(noisy, t, tags, numbers, numbers)
                reconstructed = (noisy - (1 - alpha).sqrt() * prediction) / alpha.sqrt()
                mse[start:end] += (prediction - eps).square().mean(dim=1).cpu().numpy()
                cosine[start:end] += F.cosine_similarity(clean, reconstructed).cpu().numpy()
    return mse / repeats, cosine / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    settings = read_json(args.experiment / "metrics.json")
    personal_dir = Path(settings["personal_model"])
    personal_checkpoint = torch.load(personal_dir / "model.pt", map_location="cpu", weights_only=True)
    background_checkpoint = torch.load(args.experiment / "model.pt", map_location="cpu", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    personal = ConditionalDenoiser(personal_checkpoint["vocabulary_sizes"]).to(device)
    background = ConditionalDenoiser(background_checkpoint["vocabulary_sizes"]).to(device)
    personal.load_state_dict(personal_checkpoint["state_dict"])
    background.load_state_dict(background_checkpoint["state_dict"])
    alpha_bar = noise_schedule(personal_checkpoint["steps"], device)
    features = joblib.load(personal_dir / "features.joblib")
    paths, vectors = read_vectors(default_vector_cache())
    normalized = ((unit_rows(vectors) - features["mean"]) / features["std"]).astype(np.float32)
    personal_mse, direction = probe(personal, normalized, alpha_bar, device, repeats=args.repeats)
    background_mse, _ = probe(background, normalized, alpha_bar, device, repeats=args.repeats)
    gain = background_mse - personal_mse
    with (args.experiment / "relative_ranking.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = {row["file"]: row for row in csv.DictReader(stream)}
    positive = [i for i, path in enumerate(paths) if rows[Path(path).name]["favorite_split"] == "test"]
    negative = [i for i, path in enumerate(paths) if rows[Path(path).name]["background_split"] == "background_test"]
    indices = positive + negative
    labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    result = {"timestep": 25, "repeats": args.repeats,
              "proxy_auc_single_step_personal_loss": float(roc_auc_score(labels, -personal_mse[indices])),
              "proxy_auc_single_step_relative_gain": float(roc_auc_score(labels, gain[indices])),
              "proxy_auc_direction_cosine": float(roc_auc_score(labels, direction[indices])),
              "multistep_proxy_auc_relative_gain": settings["proxy_auc_relative_gain"],
              "warning": "Unlabeled songs are not confirmed dislikes. Direction is cosine of standardized vectors."}
    write_json(args.experiment / "direction_probe.json", result)
    print(result)


if __name__ == "__main__":
    main()
