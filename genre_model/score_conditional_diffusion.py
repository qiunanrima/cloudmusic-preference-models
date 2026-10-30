"""Score arbitrary audio against held-out favorites with a saved FiLM diffusion model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from msclap_audio import PREPROCESS_ID, AudioEmbedder, default_vector_cache, read_vectors, unit_rows
from preference_data import ROOT, read_json, write_json
from train_conditional_diffusion import ConditionalDenoiser, noise_schedule, score_vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model_dir = args.model or Path(read_json(ROOT / "models_archive/conditional_diffusion/latest.json")["model_dir"])
    if read_json(model_dir / "metrics.json").get("preprocess_id") != PREPROCESS_ID:
        raise ValueError("Audio preprocessing does not match the conditional diffusion model")
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    features = joblib.load(model_dir / "features.joblib")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ConditionalDenoiser(checkpoint["vocabulary_sizes"]).to(device)
    net.load_state_dict(checkpoint["state_dict"])
    alpha_bar = noise_schedule(checkpoint["steps"], device)
    library_paths, library_vectors = read_vectors(default_vector_cache())
    cached = {Path(path).resolve(): vector for path, vector in zip(library_paths, library_vectors)}
    by_name = {Path(path).name: vector for path, vector in zip(library_paths, library_vectors)}
    reference_names = read_json(model_dir / "split.json")["validation"]
    reference = np.stack([by_name[name] for name in reference_names])
    encoder = None
    query_vectors = []
    for audio in args.audio:
        path = audio.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path in cached:
            query_vectors.append(cached[path])
        else:
            if encoder is None:
                encoder = AudioEmbedder()
            query_vectors.append(encoder.embed_file(path))
    combined = unit_rows(np.vstack([reference, *query_vectors]))
    normalized = ((combined - features["mean"]) / features["std"]).astype(np.float32)
    all_losses = score_vectors(net, normalized, alpha_bar, device, repeats=args.repeats)
    reference_losses = np.sort(all_losses[:len(reference)])
    records = []
    for path, loss in zip(args.audio, all_losses[len(reference):]):
        below = np.searchsorted(reference_losses, loss, side="left")
        above = np.searchsorted(reference_losses, loss, side="right")
        score = (2 * len(reference_losses) - below - above) * 50 / len(reference_losses)
        record = {"audio_path": str(path.resolve()), "denoising_mse": float(loss),
                  "favorite_reference_percentile": float(score)}
        records.append(record)
        print(f"{path.name}: diffusion_match={score:.2f}, denoising_mse={loss:.6f}", flush=True)
    report = {"model_dir": str(model_dir.resolve()), "reference_favorites": len(reference),
              "noise_repeats": args.repeats, "timesteps": [5, 15, 30, 50],
              "score_meaning": "held-out favorite denoising percentile, not P(like)",
              "songs": records}
    if args.output:
        write_json(args.output, report)


if __name__ == "__main__":
    main()
