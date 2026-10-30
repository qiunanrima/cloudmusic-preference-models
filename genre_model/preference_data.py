"""Conservative song matching and auditable implicit preference weights."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata

import numpy as np

ROOT = Path(os.environ.get("MUSIC_ROOT", str(Path(__file__).resolve().parents[1]))).resolve()
DEFAULT_LABELS = ROOT / "dataset_v3/labels_liked_files_20261029.json"
DEFAULT_SIGNALS = ROOT / "ncm-playback-dl/tmp/signals_final.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    if Path(path).name == "latest.json" and os.environ.get("MUSIC_EXPERIMENT_NO_LATEST") == "1":
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(value):
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(c for c in text if unicodedata.category(c)[0] in "LNM")


def artists(value):
    values = value if isinstance(value, list) else [value or ""]
    return {norm(part) for v in values for part in re.split(r"[,，、/&;；]", str(v))
            if norm(part)}


def title_keys(row):
    # Keep version qualifiers: off vocal, live and remix must remain distinct.
    keys = {norm(row.get("歌名"))}
    stem = Path(row["文件"]).stem
    if " - " in stem:
        keys.add(norm(stem.split(" - ", 1)[1]))
    return keys - {""}


def identity(row):
    return norm(row.get("歌手")) + "|" + norm(row.get("歌名"))


def link_signals(rows, signals):
    index = defaultdict(list)
    for j, song in enumerate(signals["songs"]):
        if norm(song.get("name")):
            index[norm(song["name"])].append(j)
    liked_ids = {str(p["id"]) for p in signals.get("playlists", [])
                 if "喜欢的音乐" in p.get("name", "")}
    result, report = [], []
    for row in rows:
        candidates = set(j for title in title_keys(row) for j in index[title])
        valid = []
        for j in candidates:
            song = signals["songs"][j]
            if not (artists(row.get("歌手")) & artists(song.get("artists"))):
                continue
            duration = song.get("durationMs")
            if duration and row.get("时长秒"):
                # Avoid different recordings with the same title and artist.
                if abs(duration / 1000 - row["时长秒"]) > max(5, row["时长秒"] * .03):
                    continue
            valid.append(song)
        # Multiple platforms/IDs can represent one recording. Different albums
        # are ambiguous unless the label's album identifies the recording.
        albums = {norm(s.get("album")) for s in valid if s.get("album")}
        ambiguous = len(albums) > 1
        if ambiguous:
            exact_album = [s for s in valid if norm(s.get("album")) == norm(row.get("专辑"))]
            valid = exact_album
        known = [s["playCount"] for s in valid if s.get("playCount") is not None]
        if any(not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 for v in known):
            raise ValueError("Invalid playCount in signals")
        # Do not sum counts across alternate IDs: these are cumulative signals.
        plays = max(known) if known else None
        liked = any(liked_ids.intersection(map(str, (s.get("at") or {}).keys())) for s in valid)
        weight = 0.5 + float(liked) + (float(np.log1p(plays * plays)) if plays is not None else 0.0)
        record = {"文件": row["文件"], "signal_ids": sorted(str(s["id"]) for s in valid),
                  "play_count": plays, "play_count_known": plays is not None,
                  "in_liked_playlist": liked, "weight": weight,
                  "match_status": "artist_title_duration" if valid else
                      "ambiguous_album" if ambiguous else "unmatched"}
        result.append(record)
        report.append(record)
    return result, report


def validate_labels(rows):
    if not isinstance(rows, list) or len(rows) < 10:
        raise ValueError("Expected a nonempty list of favorite-song labels")
    names = [r["文件"] for r in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate filenames in labels")
    if any(Path(n).name != n for n in names):
        raise ValueError("Labels must use basename filenames")
