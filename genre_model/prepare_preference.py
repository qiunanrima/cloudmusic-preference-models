"""Build a resumable MS-CLAP cache only for the user's labeled favorites."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from msclap_audio import AudioEmbedder, read_vectors, write_vectors
from preference_data import ROOT, DEFAULT_LABELS, read_json, validate_labels, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache", type=Path, default=Path(__file__).parent / "cache/favorites_msclap.npz")
    parser.add_argument("--import-cache", type=Path,
                        help="Trusted legacy msclap:2023 cache; imported without modifying it")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    rows = read_json(args.labels)
    validate_labels(rows)
    wanted = {r["文件"] for r in rows}
    saved = {}
    if args.import_cache:
        paths, vectors = read_vectors(args.import_cache, legacy=True)
        saved.update({Path(p).name: v for p, v in zip(paths, vectors) if Path(p).name in wanted})
    if args.cache.exists():
        paths, vectors = read_vectors(args.cache)
        saved.update({Path(p).name: v for p, v in zip(paths, vectors) if Path(p).name in wanted})
    missing = [r["文件"] for r in rows if r["文件"] not in saved and not (args.root / r["文件"]).exists()]
    pending = [r["文件"] for r in rows if r["文件"] not in saved and (args.root / r["文件"]).exists()]
    print(f"Favorites={len(rows)}, reusable={len(saved)}, pending={len(pending)}, missing_audio={len(missing)}", flush=True)
    if args.limit:
        pending = pending[:args.limit]
    def checkpoint():
        names = sorted(saved)
        if names:
            write_vectors(args.cache, names, np.stack([saved[n] for n in names]))
    checkpoint()
    failures = []
    if pending:
        embedder = AudioEmbedder()
        print(f"Encoder ready on {embedder.device}; extracting central 7-second windows", flush=True)
        # Verify preprocessing and audio-tower-only loading against old vectors.
        check_name = next((name for name in saved if (args.root / name).exists()), None)
        if check_name:
            cosine = float(embedder.embed_file(args.root / check_name) @ saved[check_name])
            print(f"Legacy compatibility cosine={cosine:.8f}: {check_name}", flush=True)
            if cosine < .9999:
                raise ValueError("Cached and newly extracted embeddings use different preprocessing")
        start = time.monotonic()
        for i, name in enumerate(pending, 1):
            try:
                saved[name] = embedder.embed_file(args.root / name)
            except Exception as exc:
                failures.append({"文件": name, "error": str(exc)})
                print(f"Failed: {name}: {exc}", flush=True)
            if i % 10 == 0 or i == len(pending):
                checkpoint()
                elapsed = time.monotonic() - start
                print(f"{i}/{len(pending)} processed, saved={len(saved)}, elapsed={elapsed:.0f}s", flush=True)
    checkpoint()
    write_json(args.cache.with_suffix(".audit.json"), {
        "labels": str(args.labels), "favorites": len(rows), "cached": len(saved),
        "missing_audio": missing, "failures": failures,
        "not_cached": [r["文件"] for r in rows if r["文件"] not in saved]})
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
