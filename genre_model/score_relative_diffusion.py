"""Score audio with paired personal/background denoising gain."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch

from msclap_audio import (PREPROCESS_ID, AudioEmbedder, default_vector_cache,
                          read_vectors, unit_rows)
from preference_data import ROOT, read_json, write_json
from train_conditional_diffusion import ConditionalDenoiser, noise_schedule, score_vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    experiment = args.experiment.resolve()
    settings = read_json(experiment / "metrics.json")
    if settings.get("preprocess_id") and settings["preprocess_id"] != PREPROCESS_ID:
        raise ValueError("Scoring preprocessing does not match the relative diffusion model")
    personal = Path(settings["personal_model"])
    personal_checkpoint = torch.load(personal / "model.pt", map_location="cpu", weights_only=True)
    background_checkpoint = torch.load(experiment / "model.pt", map_location="cpu", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    personal_net = ConditionalDenoiser(personal_checkpoint["vocabulary_sizes"]).to(device)
    background_net = ConditionalDenoiser(background_checkpoint["vocabulary_sizes"]).to(device)
    personal_net.load_state_dict(personal_checkpoint["state_dict"])
    background_net.load_state_dict(background_checkpoint["state_dict"])
    alpha_bar = noise_schedule(personal_checkpoint["steps"], device)
    features = joblib.load(personal / "features.joblib")
    paths, vectors = read_vectors(default_vector_cache())
    by_path = {Path(path).resolve(): vector for path, vector in zip(paths, vectors)}
    by_name = {Path(path).name: vector for path, vector in zip(paths, vectors)}
    reference_names = read_json(personal / "split.json")["validation"]
    reference = [by_name[name] for name in reference_names]
    encoder = None
    queries = []
    for path in args.audio:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        if resolved in by_path:
            queries.append(by_path[resolved])
        else:
            if encoder is None:
                encoder = AudioEmbedder()
            queries.append(encoder.embed_file(resolved))
    combined = unit_rows(np.vstack([*reference, *queries]))
    normalized = ((combined - features["mean"]) / features["std"]).astype(np.float32)
    personal_loss = score_vectors(personal_net, normalized, alpha_bar, device,
                                  repeats=args.repeats)
    background_loss = score_vectors(background_net, normalized, alpha_bar, device,
                                    repeats=args.repeats)
    gain = background_loss - personal_loss
    reference_gain = np.sort(gain[:len(reference)])
    records = []
    for path, p_loss, b_loss, value in zip(args.audio, personal_loss[len(reference):],
                                           background_loss[len(reference):], gain[len(reference):]):
        score = np.searchsorted(reference_gain, value, side="right") * 100 / len(reference_gain)
        record = {"audio_path": str(path.resolve()), "personal_mse": float(p_loss),
                  "background_mse": float(b_loss), "relative_denoising_gain": float(value),
                  "favorite_reference_percentile": float(score)}
        records.append(record)
        print(f"{path.name}: relative_match={score:.2f}; gain={value:.6f}; "
              f"personal={p_loss:.6f}; background={b_loss:.6f}", flush=True)
    if args.output:
        write_json(args.output, {"experiment": str(experiment), "noise_repeats": args.repeats,
                                 "scoring_noise_protocol": "shared Gaussian across songs and paired models; seed=2026",
                                 "score_meaning": "Held-out favorite percentile, not P(like)",
                                 "songs": records})


if __name__ == "__main__":
    main()
