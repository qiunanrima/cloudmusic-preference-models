"""Score every top-level MP3 in the local library and write a CSV leaderboard."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from msclap_audio import (DIM, ENCODER_ID, PREPROCESS_ID, AudioEmbedder,
                          default_vector_cache, read_vectors, write_vectors,
                          unit_rows)
from preference_data import ROOT, write_json
from preference_knn import (PreferenceModel, similarity_components, percentile,
                            PositiveKNN, model_directory)


HERE = Path(__file__).resolve().parent
DEFAULT_CACHE = default_vector_cache()
DEFAULT_OUT = HERE / "outputs/library_preference_ranking_liked_files.csv"


def save_cache(path, vector_by_name, root):
    names = sorted(vector_by_name)
    write_vectors(path, [str(root / name) for name in names],
                  np.stack([vector_by_name[name] for name in names]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    songs = sorted(root.glob("*.mp3"), key=lambda p: p.name.casefold())
    if not songs:
        raise SystemExit(f"No root-level MP3 files found in {root}")
    names = [p.name for p in songs]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate filenames under library root")
    model = PreferenceModel(model_directory(args.model))
    cache_by_name = {Path(r["文件"]).name: v for r, v in zip(model.rows, model.vectors)}
    if args.cache.exists():
        cache_paths, cache_vectors = read_vectors(args.cache)
        cache_by_name.update({Path(p).name: v for p, v in zip(cache_paths, cache_vectors)
                              if Path(p).parent.resolve() == root})
    missing = [p for p in songs if p.name not in cache_by_name]
    print(f"Songs={len(songs)}, reusable vectors={len(songs)-len(missing)}, to encode={len(missing)}", flush=True)
    if missing:
        encoder = AudioEmbedder()
        start = time.monotonic()
        failed = []
        for offset in range(0, len(missing), args.checkpoint_every):
            chunk = missing[offset:offset + args.checkpoint_every]
            kept, matrix, errors = encoder.embed_files(chunk, batch_size=args.batch_size)
            cache_by_name.update({Path(p).name: v for p, v in zip(kept, matrix)})
            failed.extend(errors)
            save_cache(args.cache, cache_by_name, root)
            done = min(offset + len(chunk), len(missing))
            elapsed = time.monotonic() - start
            print(f"Encoded {done}/{len(missing)}; cached total={len(cache_by_name)}; errors={len(failed)}; elapsed={elapsed/60:.1f} min", flush=True)
        if failed:
            write_json(args.cache.with_suffix(".errors.json"),
                       [{"path": p, "error": message} for p, message in failed])
            print(f"Skipped {len(failed)} undecodable file(s); see {args.cache.with_suffix('.errors.json')}", flush=True)
    available = [p for p in songs if p.name in cache_by_name]
    if len(available) != len(songs):
        print(f"Leaderboard covers {len(available)}/{len(songs)} decodable MP3s", flush=True)
    vectors = unit_rows(np.stack([cache_by_name[p.name] for p in available]))
    query_rows, known_indices = [], []
    for path in available:
        index = model.by_name.get(path.name)
        if index is None:
            query_rows.append({})
        else:
            query_rows.append(model.rows[index])
        known_indices.append(index)
    print("Calculating audio and metadata similarities against favorites...", flush=True)
    audio, residual, coverage = similarity_components(query_rows, model.rows, vectors, model.vectors)
    tagged = np.array([i is not None and bool(coverage[j].max() > 0)
                       for j, i in enumerate(known_indices)])
    similarity = np.clip(audio + model.config["metadata_alpha"] * residual, -1., 1.)
    # Use the matching calibration for each query. Mixed calibration for a
    # single matrix call would mis-rank audio-only files against tagged songs.
    score = np.empty(len(available), dtype=np.float64)
    raw_all = np.empty(len(available), dtype=np.float64)
    neighbor_data = {}
    for has_metadata, reference in ((False, model.calibration_audio), (True, model.calibration)):
        query_ids = np.flatnonzero(tagged == has_metadata)
        if not len(query_ids):
            continue
        query_similarity = audio[query_ids] if not has_metadata else similarity[query_ids]
        exclude = query_ids[:, None] == -1  # allocate correctly shaped all-false mask
        exclude = np.broadcast_to(exclude, query_similarity.shape).copy()
        for local, original in enumerate(query_ids):
            same_audio = audio[original] > .995
            exclude[local] |= same_audio
            favorite_index = known_indices[original]
            if favorite_index is not None:
                exclude[local] |= model.recordings == model.recordings[favorite_index]
        scorer = model._index(has_metadata)
        raw, _, neighbors, sims, valid = scorer.predict(query_similarity, exclude)
        raw_all[query_ids] = raw
        score[query_ids] = percentile(raw, reference)
        contributions = np.exp((sims - 1.) / model.config["tau"]) * model.weights[neighbors]
        contributions[~valid] = 0
        contributions /= contributions.sum(axis=1, keepdims=True)
        for row_index, original in enumerate(query_ids):
            neighbor_data[int(original)] = [
                {"file": model.rows[int(n)]["文件"],
                 "genre": model.rows[int(n)].get("风格"),
                 "similarity": float(sims[row_index, j]),
                 "behavior_weight": float(model.weights[int(n)]),
                 "play_count": model.behavior[int(n)]["play_count"],
                 "contribution": float(contributions[row_index, j])}
                for j, n in enumerate(neighbors[row_index]) if valid[row_index, j]
            ]
    metadata_by_name = {row["文件"]: row for row in model.rows}
    rows = []
    for j, path in enumerate(available):
        metadata = metadata_by_name.get(path.name, {})
        neighbors = sorted(neighbor_data[j], key=lambda x: x["contribution"], reverse=True)
        artist = metadata.get("歌手", "")
        title = metadata.get("歌名", "")
        if not artist and " - " in path.stem:
            artist, title_from_file = path.stem.split(" - ", 1)
            title = title or title_from_file
        favorite_index = known_indices[j]
        behavior = model.behavior[favorite_index] if favorite_index is not None else {}
        rows.append({
            "score": float(score[j]), "file": path.name,
            "artist": artist, "title": title,
            "genre": metadata.get("风格", ""),
            "metadata_used": "audio+labels" if tagged[j] else "audio",
            "favorite": "yes" if favorite_index is not None else "no",
            "behavior_weight": float(model.weights[favorite_index]) if favorite_index is not None else "",
            "play_count": behavior.get("play_count") if behavior.get("play_count_known") else "unknown" if favorite_index is not None else "",
            "in_liked_playlist": "yes" if behavior.get("in_liked_playlist") else "no" if favorite_index is not None else "",
            "nearest_favorites": " | ".join(
                f"{n['file']} [sim={n['similarity']:.4f}, behavior_weight={n['behavior_weight']:.2f}, "
                f"share={n['contribution']:.1%}, plays={n['play_count'] if n['play_count'] is not None else 'unknown'}]"
                for n in neighbors[:5]),
            "audio_path": str(path),
            "raw_support": float(raw_all[j]),
        })
    rows.sort(key=lambda r: (-r["score"], r["file"].casefold()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "rank", "score", "file", "artist", "title", "genre", "metadata_used",
            "favorite", "behavior_weight", "play_count", "in_liked_playlist",
            "nearest_favorites", "audio_path", "raw_support"])
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            writer.writerow({"rank": rank, **row})
    manifest = {"model_dir": str(model.directory), "library_root": str(root),
                "input_files": len(songs), "ranked_files": len(rows),
                "ranked_with_metadata": int(tagged.sum()),
                "ranked_audio_only": int((~tagged).sum()),
                "format": "CSV UTF-8 with BOM; Excel-compatible",
                "score": "favorite-reference percentile, not a probability",
                "created_at": datetime.now().astimezone().isoformat(),
                "output": str(args.output.resolve()),
                "vector_cache": str(args.cache.resolve())}
    write_json(args.output.with_suffix(".json"), manifest)
    print(f"Saved {len(rows)} ranked songs: {args.output.resolve()}", flush=True)
    print("Top 10:")
    for rank, row in enumerate(rows[:10], 1):
        print(f"{rank:>4}. {row['score']:5.1f}  {row['file']}")


if __name__ == "__main__":
    main()
