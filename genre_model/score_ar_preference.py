"""Score audio with saved personal/background continuous AR models."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
import torch

from msclap_audio import PREPROCESS_ID, AudioEmbedder, default_vector_cache, read_vectors, unit_rows
from preference_data import read_json, write_json
from train_ar_preference import ContinuousAR, score


def load_net(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    net = ContinuousAR(checkpoint["dimension"], checkpoint["chunks"],
                       checkpoint["width"], checkpoint["layers"],
                       checkpoint["mixtures"]).to(device)
    net.load_state_dict(checkpoint["state_dict"])
    return net


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = read_json(args.model / "metrics.json")
    if settings.get("preprocess_id") != PREPROCESS_ID:
        raise ValueError("Audio preprocessing does not match the AR model")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    personal = load_net(args.model / "model_personal.pt", device)
    background = load_net(args.model / "model_background.pt", device)
    features = joblib.load(args.model / "features.joblib")
    library_paths, library_vectors = read_vectors(default_vector_cache())
    cached = {Path(path).resolve(): vector for path, vector in zip(library_paths, library_vectors)}
    encoder = None
    vectors = []
    for audio in args.audio:
        path = audio.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path in cached:
            vectors.append(cached[path])
        else:
            if encoder is None:
                encoder = AudioEmbedder()
            vectors.append(encoder.embed_file(path))
    transformed = (features["pca"].transform(unit_rows(np.vstack(vectors))) /
                   features["scale"]).astype(np.float32)
    personal_ll = score(personal, transformed, device)
    background_ll = score(background, transformed, device)
    gains = personal_ll - background_ll
    with (args.model / "relative_ranking.csv").open(encoding="utf-8-sig", newline="") as stream:
        reference = np.sort([float(row["ar_log_likelihood_ratio"])
                             for row in csv.DictReader(stream)
                             if row["favorite_split"] == "validation"])
    records = []
    for path, p, b, gain in zip(args.audio, personal_ll, background_ll, gains):
        percentile = float(np.searchsorted(reference, gain, side="right") * 100 / len(reference))
        record = {"audio_path": str(path.resolve()), "personal_log_likelihood": float(p),
                  "background_log_likelihood": float(b),
                  "ar_log_likelihood_ratio": float(gain),
                  "favorite_reference_percentile": percentile}
        records.append(record)
        print(f"{path.name}: ar_match={percentile:.2f}; log_ratio={gain:.4f}", flush=True)
    if args.output:
        write_json(args.output, {"model_dir": str(args.model.resolve()),
                                 "score_meaning": "Held-out favorite percentile, not P(like)",
                                 "songs": records})


if __name__ == "__main__":
    main()
