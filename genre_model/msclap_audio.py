"""Offline MS-CLAP 2023 audio embeddings, compatible with the archived cache.

One deterministic central 7-second window, 44100 Hz mono, no peak scaling.
Only the audio tower is loaded; no tokenizer/text-model download is needed.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types

import numpy as np

ENCODER_ID = "msclap:2023"
DIM = int(os.environ.get("MSCLAP_VECTOR_DIM", "1024"))
DEFAULT_PREPROCESS_ID = "msclap2023-mono44100-center7-repeat-short-no-peak-v1"
VECTOR_CACHE_POINTER = Path(__file__).resolve().parent / "cache/vector_cache_current.json"


def _current_cache_settings():
    try:
        return json.loads(VECTOR_CACHE_POINTER.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


PREPROCESS_ID = os.environ.get("MSCLAP_PREPROCESS_ID") or _current_cache_settings().get(
    "preprocess_id", DEFAULT_PREPROCESS_ID)


def default_vector_cache():
    configured = os.environ.get("MSCLAP_CACHE")
    if configured:
        return Path(configured)
    current = _current_cache_settings().get("vector_cache")
    return Path(current) if current else Path(__file__).resolve().parent / "cache/library_msclap.npz"


def unit_rows(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Embeddings must be a finite 2D matrix")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms < 1e-8).any():
        raise ValueError("Zero audio embedding")
    return x / norms


def cache_protocol():
    path = default_vector_cache().with_suffix('.protocol.json')
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def _audio_import_compat():
    # This environment deliberately has no torchaudio. Decoding uses librosa.
    import transformers  # Import before the compatibility modules are installed.
    if importlib.util.find_spec("torchaudio") is not None:
        return
    for name in ("torchaudio", "torchaudio.transforms", "torchaudio.functional"):
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(
            name, loader=None, is_package=name == "torchaudio")
        if name == "torchaudio":
            mod.__path__ = []
        sys.modules[name] = mod
    def unsupported(*args, **kwargs):
        raise RuntimeError("Use librosa decoding; torchaudio is not installed")
    sys.modules["torchaudio"].load = unsupported
    sys.modules["torchaudio"].transforms = sys.modules["torchaudio.transforms"]
    sys.modules["torchaudio"].functional = sys.modules["torchaudio.functional"]


class AudioEmbedder:
    def __init__(self, device=None):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        import yaml
        _audio_import_compat()
        import msclap
        from msclap.models.clap import AudioEncoder
        from huggingface_hub import hf_hub_download

        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        package = Path(msclap.__file__).parent
        config = yaml.safe_load((package / "configs/config_2023.yml").read_text())
        self.model = AudioEncoder(
            config["audioenc_name"], config["out_emb"], config["d_proj"],
            config["sampling_rate"], config["window_size"], config["hop_size"],
            config["mel_bins"], config["fmin"], config["fmax"], config["num_classes"])
        checkpoint = hf_hub_download("microsoft/msclap", "CLAP_weights_2023.pth",
                                     local_files_only=True)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        prefix = "audio_encoder."
        state = {key[len(prefix):]: value for key, value in state.items()
                 if key.startswith(prefix)}
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.torch = torch

    def embed_file(self, path):
        import librosa
        transcode_bitrate = (os.environ.get("MSCLAP_TRANSCODE_BITRATE") or
                             _current_cache_settings().get("transcode_bitrate_kbps"))
        if transcode_bitrate:
            with tempfile.TemporaryDirectory(prefix="msclap_transcode_") as folder:
                transcoded = Path(folder) / "query.mp3"
                subprocess.run([
                    "ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
                    "-map", "0:a:0", "-vn", "-map_metadata", "-1",
                    "-ar", "44100", "-ac", "2", "-c:a", "libmp3lame",
                    "-b:a", f"{transcode_bitrate}k", str(transcoded)],
                    check=True, capture_output=True)
                return self._embed_file_decoded(transcoded, librosa)
        return self._embed_file_decoded(path, librosa)

    def _embed_file_decoded(self, path, librosa):
        # Keep full-file decoding identical to the archived encoder before slicing.
        wave, _ = librosa.load(str(path), sr=44100, mono=True)
        if wave.size < 1 or not np.isfinite(wave).all():
            raise ValueError(f"Invalid audio: {path}")
        size = 7 * 44100
        if len(wave) < size:
            wave = np.tile(wave, int(np.ceil(size / len(wave))))
        start = (len(wave) - size) // 2
        tensor = self.torch.from_numpy(wave[start:start + size].copy()).to(self.device)
        with self.torch.inference_mode():
            vector, _ = self.model(tensor[None])
        return unit_rows(vector.float().cpu().numpy())[0]

    def embed_files(self, paths, batch_size=4, preprocessed=False):
        """Decode files serially, then run small GPU batches for faster ranking."""
        import librosa
        paths = list(paths)
        bitrate = (os.environ.get("MSCLAP_TRANSCODE_BITRATE") or
                   _current_cache_settings().get("transcode_bitrate_kbps"))
        if bitrate and not preprocessed:
            with tempfile.TemporaryDirectory(prefix="msclap_batch_transcode_") as folder:
                originals, transcoded, failures = {}, [], []
                for index, path in enumerate(paths):
                    target = Path(folder) / f"{index:06d}.mp3"
                    try:
                        subprocess.run([
                            "ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
                            "-map", "0:a:0", "-vn", "-map_metadata", "-1",
                            "-ar", "44100", "-ac", "2", "-c:a", "libmp3lame",
                            "-b:a", f"{bitrate}k", str(target)],
                            check=True, capture_output=True)
                        originals[str(target)] = str(path)
                        transcoded.append(target)
                    except Exception as exc:
                        failures.append((str(path), str(exc)))
                kept, vectors, errors = self.embed_files(
                    transcoded, batch_size=batch_size, preprocessed=True)
                failures.extend((originals[p], message) for p, message in errors)
                return [originals[p] for p in kept], vectors, failures
        vectors, kept, failed = [], [], []
        size = 7 * 44100
        for start in range(0, len(paths), batch_size):
            waves, wave_paths = [], []
            for path in paths[start:start + batch_size]:
                try:
                    wave, _ = librosa.load(str(path), sr=44100, mono=True)
                    if wave.size < 1 or not np.isfinite(wave).all():
                        raise ValueError("empty or non-finite decoded audio")
                    if len(wave) < size:
                        wave = np.tile(wave, int(np.ceil(size / len(wave))))
                    offset = (len(wave) - size) // 2
                    waves.append(wave[offset:offset + size].astype(np.float32, copy=False))
                    wave_paths.append(path)
                except Exception as exc:
                    failed.append((str(path), str(exc)))
            if not waves:
                continue
            tensors = self.torch.from_numpy(np.stack(waves)).to(self.device)
            with self.torch.inference_mode():
                encoded, _ = self.model(tensors)
            vectors.extend(unit_rows(encoded.float().cpu().numpy()))
            kept.extend(map(str, wave_paths))
        mat = np.stack(vectors) if vectors else np.zeros((0, DIM), dtype=np.float32)
        return kept, mat, failed


def read_vectors(path, legacy=False, expected_preprocess_id=None):
    with np.load(path, allow_pickle=legacy) as data:
        if str(data["model_id"].item()) != ENCODER_ID:
            raise ValueError(f"Not MS-CLAP 2023: {path}")
        vectors = unit_rows(data["vectors"])
        paths = [str(p) for p in data["paths"]]
        if vectors.shape != (len(paths), DIM):
            raise ValueError(f"Invalid 1024-dimensional vector cache: {path}")
        if len(set(paths)) != len(paths):
            if not legacy:
                raise ValueError("Duplicate paths in embedding cache")
            unique = {}
            for p, v in zip(paths, vectors):
                if p in unique and not np.allclose(unique[p], v, atol=1e-6):
                    raise ValueError(f"Conflicting legacy embeddings: {p}")
                unique[p] = v
            paths, vectors = list(unique), np.stack(list(unique.values()))
        expected = expected_preprocess_id or PREPROCESS_ID
        if "preprocess_id" in data and str(data["preprocess_id"].item()) != expected:
            raise ValueError("Audio preprocessing differs from this model")
    return paths, vectors


def write_vectors(path, paths, vectors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, paths=np.asarray(paths, dtype=str),
                        vectors=unit_rows(vectors), model_id=ENCODER_ID,
                        preprocess_id=PREPROCESS_ID)
    temp.replace(path)
