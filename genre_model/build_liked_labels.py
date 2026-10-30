"""Build favorite metadata using liked-files.txt as the sole positive authority."""
from __future__ import annotations

from pathlib import Path

import soundfile as sf

from preference_data import ROOT, read_json, sha256, write_json


SOURCE = ROOT / "ncm-playback-dl/tmp/liked-files.txt"
OLD_LABELS = ROOT / "dataset_v3/labels_v5_clean.json"
OUTPUT = ROOT / "dataset_v3/labels_liked_files_20261029.json"


def main():
    lines = [line.strip().strip('"') for line in SOURCE.read_text(encoding="utf-8-sig").splitlines()
             if line.strip()]
    names = sorted({Path(line).name for line in lines}, key=str.casefold)
    old = {row["文件"]: row for row in read_json(OLD_LABELS)}
    missing_audio = [name for name in names if not (ROOT / name).is_file()]
    if missing_audio:
        raise FileNotFoundError(f"{len(missing_audio)} listed favorites lack audio: {missing_audio[:5]}")
    rows = []
    added = []
    for name in names:
        if name in old:
            rows.append(old[name])
            continue
        stem = Path(name).stem
        artist, title = stem.split(" - ", 1) if " - " in stem else ("", stem)
        duration = float(sf.info(str(ROOT / name)).duration)
        rows.append({"文件": name, "歌名": title, "歌手": artist,
                     "专辑": "", "时长秒": duration,
                     "风格": None, "风格细分": [], "语种": None,
                     "人声形式": None, "版本类型": [], "作品来源": [],
                     "场景标签": [], "文化标签": [], "BPM": None,
                     "发行年": None, "依据": {},
                     "标签说明": "liked-files.txt 正样本；原标签集中缺失，未猜测曲风"})
        added.append(name)
    if len(rows) != len(names) or {row["文件"] for row in rows} != set(names):
        raise AssertionError("Output favorites differ from liked-files.txt")
    write_json(OUTPUT, rows)
    write_json(OUTPUT.with_suffix(".audit.json"), {
        "source": str(SOURCE), "source_sha256": sha256(SOURCE),
        "previous_labels": str(OLD_LABELS), "previous_labels_sha256": sha256(OLD_LABELS),
        "unique_liked": len(names), "listed_lines": len(lines),
        "reused_metadata": len(rows) - len(added),
        "new_missing_metadata": added, "output_sha256": sha256(OUTPUT)})
    print(f"liked={len(rows)}; reused_metadata={len(rows)-len(added)}; "
          f"missing_metadata={len(added)}; output={OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
