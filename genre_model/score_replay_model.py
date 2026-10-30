"""Score arbitrary audio with the experimental replay net and current KNN."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from msclap_audio import AudioEmbedder
from preference_data import ROOT
from preference_knn import PreferenceModel, model_directory
from train_replay_model import ReplayNet, infer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--replay-model", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    knn = PreferenceModel(model_directory(None))
    replay_dir = args.replay_model or Path(__file__).resolve().parents[1] / "models_archive/replay_net"
    if replay_dir.name == "replay_net":
        from preference_data import read_json
        replay_dir = Path(read_json(replay_dir / "latest.json")["model_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(replay_dir / "model.pt", map_location=device, weights_only=True)
    net = ReplayNet().to(device)
    net.load_state_dict(checkpoint["state_dict"])
    mean, std = checkpoint["target_mean"], checkpoint["target_std"]
    reference = np.sort(infer(net, knn.vectors.astype(np.float32), device) * std + mean)
    encoder = None
    for path in args.audio:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        known = knn.by_name.get(path.name) if path.parent == ROOT else None
        if known is not None:
            vector = knn.vectors[known]
            metadata = knn.rows[known]
        else:
            if encoder is None:
                encoder = AudioEmbedder()
            vector = encoder.embed_file(path)
            metadata = None
        knn_result = knn.predict_vector(vector, metadata=metadata, filename=path.name if known is not None else None)
        pred = float(infer(net, np.asarray(vector, dtype=np.float32)[None], device)[0] * std + mean)
        replay_percentile = float((np.searchsorted(reference, pred, side="left") +
                                   np.searchsorted(reference, pred, side="right")) * 50 / len(reference))
        print(f"{path}\n  KNN={knn_result['match_score']:.2f} "
              f"({knn_result['query_features']}); experimental_replay={replay_percentile:.2f} "
              f"pred_log1p_plays={pred:.4f}", flush=True)


if __name__ == "__main__":
    main()
