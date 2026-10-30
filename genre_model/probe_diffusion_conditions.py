"""Test whether real tag conditions help denoise held-out favorites."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch

from msclap_audio import unit_rows
from preference_data import read_json, write_json
from preference_knn import PreferenceModel
from train_conditional_diffusion import (ConditionalDenoiser, losses,
                                         noise_schedule, prepare_conditions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    metrics = read_json(args.model / "metrics.json")
    knn = PreferenceModel(metrics["knn_model_dir"])
    split = read_json(args.model / "split.json")
    indices = np.array([knn.by_name[name] for name in split["validation"]])
    training = np.array([knn.by_name[name] for name in split["train"]])
    features = joblib.load(args.model / "features.joblib")
    tags, numbers, masks, _, _ = prepare_conditions(
        knn.rows, training, features["vocabularies"], features["numeric_stats"])
    clean_np = ((unit_rows(knn.vectors[indices]) - features["mean"]) /
                features["std"]).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model / "model.pt", map_location="cpu", weights_only=True)
    net = ConditionalDenoiser(checkpoint["vocabulary_sizes"]).to(device)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    alpha_bar = noise_schedule(checkpoint["steps"], device)
    clean = torch.from_numpy(clean_np).to(device)
    real_tags = torch.from_numpy(tags[indices]).to(device)
    real_numbers = torch.from_numpy(numbers[indices]).to(device)
    real_masks = torch.from_numpy(masks[indices]).to(device)
    rng = np.random.default_rng(2026)
    perm = torch.from_numpy(rng.permutation(len(indices))).to(device)
    cases = {
        "real": (real_tags, real_numbers, real_masks),
        "shuffled": (real_tags[perm], real_numbers[perm], real_masks[perm]),
        "null": (torch.zeros_like(real_tags), torch.zeros_like(real_numbers), torch.zeros_like(real_masks)),
    }
    sums = {key: np.zeros(len(indices), dtype=np.float64) for key in cases}
    with torch.no_grad():
        for timestep in (5, 15, 30, 50):
            for _ in range(args.repeats):
                noise = torch.from_numpy(rng.standard_normal(clean_np.shape, dtype=np.float32)).to(device)
                t = torch.full((len(clean),), timestep, device=device, dtype=torch.long)
                for key, (case_tags, case_numbers, case_masks) in cases.items():
                    sums[key] += losses(net, clean, t, noise, alpha_bar,
                                        case_tags, case_numbers, case_masks).cpu().numpy()
    scores = {key: value / (4 * args.repeats) for key, value in sums.items()}
    rng = np.random.default_rng(317)
    delta = scores["shuffled"] - scores["real"]
    bootstrap = [float(np.mean(delta[rng.integers(len(delta), size=len(delta))]))
                 for _ in range(2000)]
    result = {"validation_favorites": len(indices), "repeats": args.repeats,
              "mean_mse_real": float(np.mean(scores["real"])),
              "mean_mse_shuffled": float(np.mean(scores["shuffled"])),
              "mean_mse_null": float(np.mean(scores["null"])),
              "shuffled_minus_real": float(np.mean(delta)),
              "shuffled_minus_real_bootstrap_95ci": np.quantile(bootstrap, [.025, .975]).tolist(),
              "per_song_real_better_than_shuffled": float(np.mean(delta > 0)),
              "warning": "A single deterministic tag permutation; validation songs were used to choose the original checkpoint."}
    write_json(args.model / "condition_probe.json", result)
    print(result)


if __name__ == "__main__":
    main()
