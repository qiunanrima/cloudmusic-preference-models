"""Positive-only, behavior-weighted nearest-neighbor music preference model.

The score is a percentile of weighted neighborhood support among favorites,
never a probability of liking. Only labels define favorites; playback records
provide weights and an auxiliary replay-intensity validation target.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors

from msclap_audio import (DIM, ENCODER_ID, PREPROCESS_ID, AudioEmbedder,
                          default_vector_cache, read_vectors, unit_rows)
from preference_data import (ROOT, DEFAULT_LABELS, DEFAULT_SIGNALS, artists, identity,
                             link_signals, norm, read_json, sha256, validate_labels, write_json)

HERE = Path(__file__).resolve().parent
DEFAULT_CACHE = default_vector_cache()
DEFAULT_MODELS = ROOT / "models_archive/knn_preference"

# No song title, artist ID, filename, or album ID enters the distance function.
TAG_WEIGHTS = {"风格": 4., "风格细分": 2., "语种": 1., "人声形式": .6,
               "版本类型": .3, "作品来源": .6, "场景标签": .2, "文化标签": .3}
NUMERIC_WEIGHTS = {"BPM": (1., .5), "发行年": (.3, 20.), "时长秒": (.3, .7)}
TOTAL_META_WEIGHT = sum(TAG_WEIGHTS.values()) + sum(w for w, _ in NUMERIC_WEIGHTS.values())


def confidence(row, field):
    value = row.get(field)
    if not value:
        return 0.
    status = (row.get("依据", {}).get(field) or {}).get("状态")
    if status in ("待复核", "冲突待复核", "缺失"):
        return 0.
    return .4 if status == "规则推断" else 1.


def tag_values(row, field):
    value = row.get(field)
    return {norm(v) for v in (value if isinstance(value, list) else [value]) if norm(v)}


def similarity_components(query_rows, reference_rows, query_vectors, reference_vectors):
    """Return audio cosines and metadata residuals, with missingness masking.

    s = audio + alpha * sum(field_weight * joint_confidence * (field_sim-audio))
        / sum(all_field_weights).
    An absent field has exactly zero effect, including unknown == unknown.
    All scales are fixed ahead of evaluation, with no test-fitted scaler.
    """
    audio = np.clip(unit_rows(query_vectors) @ unit_rows(reference_vectors).T, -1., 1.)
    residual = np.zeros_like(audio)
    coverage = np.zeros_like(audio)
    for field, weight in TAG_WEIGHTS.items():
        qsets = [tag_values(r, field) for r in query_rows]
        rsets = [tag_values(r, field) for r in reference_rows]
        vocab = sorted(set().union(*qsets, *rsets))
        if not vocab:
            continue
        lookup = {v: j for j, v in enumerate(vocab)}
        def encode(sets):
            out = np.zeros((len(sets), len(vocab)), dtype=np.float32)
            for i, values in enumerate(sets):
                if values:
                    out[i, [lookup[v] for v in values]] = 1 / np.sqrt(len(values))
            return out
        sim = encode(qsets) @ encode(rsets).T
        joint = np.outer([confidence(r, field) for r in query_rows],
                         [confidence(r, field) for r in reference_rows]).astype(np.float32)
        mass = weight * joint / TOTAL_META_WEIGHT
        residual += mass * (sim - audio)
        coverage += mass
    for field, (weight, scale) in NUMERIC_WEIGHTS.items():
        def numeric(rows):
            values = np.array([r.get(field) or np.nan for r in rows], dtype=np.float32)
            valid = np.isfinite(values) & (values > 0)
            if field != "发行年":
                values = np.log(np.where(valid, values, 1.))
            return np.where(valid, values, 0.), valid
        q, qvalid = numeric(query_rows)
        r, rvalid = numeric(reference_rows)
        sim = np.exp(-np.abs(q[:, None] - r[None, :]) / scale)
        mass = weight * np.outer(qvalid, rvalid) / TOTAL_META_WEIGHT
        residual += mass * (sim - audio)
        coverage += mass
    return audio, residual, coverage


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def root(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, a, b):
        self.parent[self.root(a)] = self.root(b)


def recording_groups(rows, audio_similarity):
    """Union exact metadata identities and nearly identical audio embeddings."""
    uf, seen = UnionFind(len(rows)), {}
    for i, row in enumerate(rows):
        key = identity(row)
        if key in seen:
            uf.join(i, seen[key])
        seen[key] = i
    # Near-identical central windows group alternate encodings of a recording.
    ii, jj = np.where(np.triu(audio_similarity > .995, 1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        uf.join(i, j)
    return np.array([uf.root(i) for i in range(len(rows))])


def artist_groups(rows, recordings):
    """Connected artist components keep collaborations and duplicates together."""
    from msclap_audio import cache_protocol
    fixed = cache_protocol().get('fixed_favorite_groups')
    if fixed:
        manifest = read_json(fixed)
        mapping = dict(zip(manifest['files'], manifest['groups']))
        if set(mapping) != {row['文件'] for row in rows}:
            raise ValueError('Fixed favorite group identity mismatch')
        return np.array([mapping[row['文件']] for row in rows])
    uf, seen = UnionFind(len(rows)), {}
    for i, row in enumerate(rows):
        for artist in artists(row.get("歌手")):
            if artist in seen:
                uf.join(i, seen[artist])
            seen[artist] = i
    for group in np.unique(recordings):
        indices = np.flatnonzero(recordings == group)
        for i in indices[1:]:
            uf.join(int(indices[0]), int(i))
    roots = np.array([uf.root(i) for i in range(len(rows))])
    # Union roots depend on set iteration order; use each component's first
    # song index as a stable label for reproducible grouped data splits.
    smallest = np.full(len(rows), len(rows), dtype=int)
    np.minimum.at(smallest, roots, np.arange(len(rows)))
    return smallest[roots]


class PositiveKNN:
    def __init__(self, similarity, weights, counts, k=20, tau=.1):
        if not isinstance(k, (int, np.integer)) or k < 1 or not np.isfinite(tau) or tau <= 0:
            raise ValueError("k must be positive and tau finite and > 0")
        self.weights = np.asarray(weights, dtype=np.float64)
        self.counts = np.asarray(counts, dtype=np.float64)
        if not np.isfinite(self.weights).all() or (self.weights <= 0).any():
            raise ValueError("Positive finite behavior weights required")
        if len(self.weights) != len(self.counts) or similarity.shape != (len(weights), len(weights)):
            raise ValueError("Reference size mismatch")
        self.k, self.tau = int(k), float(tau)
        distances = np.clip(1. - similarity, 0., 2.)
        np.fill_diagonal(distances, 0.)
        self.index = NearestNeighbors(metric="precomputed", algorithm="brute", n_jobs=1)
        self.index.fit(distances)

    def predict(self, similarities, exclude=None):
        distances = np.clip(1. - np.asarray(similarities, dtype=np.float32), 0., 2.)
        if exclude is None:
            exclude = np.zeros(distances.shape, dtype=bool)
        if exclude.shape != distances.shape or np.any(np.sum(~exclude, axis=1) < 1):
            raise ValueError("Each query needs at least one non-excluded neighbor")
        distances = distances.copy()
        distances[exclude] = 3.
        count = min(self.k, distances.shape[1])
        d, indices = self.index.kneighbors(distances, n_neighbors=count)
        valid = d <= 2.
        sims = 1. - d
        logterms = (sims - 1.) / self.tau + np.log(self.weights[indices])
        logterms[~valid] = -np.inf
        raw = logsumexp(logterms, axis=1) - np.log(valid.sum(axis=1))
        # Auxiliary playback prediction uses only observed counts. Unknown is
        # never silently turned into zero and never used as a negative example.
        observed = valid & np.isfinite(self.counts[indices])
        coeffs = np.exp(logterms - np.max(logterms, axis=1, keepdims=True)) * observed
        targets = np.log1p(np.where(observed, self.counts[indices], 0.))
        total = coeffs.sum(axis=1)
        replay = np.divide((coeffs * targets).sum(axis=1), total,
                           out=np.full(len(total), np.nan), where=total > 0)
        return raw, replay, indices, sims, valid


def replay_metrics(y, predictions):
    finite = np.isfinite(y) & np.isfinite(predictions)
    if finite.sum() < 3:
        return {"n": int(finite.sum()), "mae_log1p": None, "spearman": None}
    target, pred = y[finite], predictions[finite]
    rho = float(spearmanr(target, pred).statistic) if np.std(pred) > 1e-12 else None
    return {"n": int(finite.sum()), "mae_log1p": float(mean_absolute_error(target, pred)),
            "spearman": rho if rho is not None and np.isfinite(rho) else None}


def evaluate_parameters(audio, residual, weights, counts, groups, seed):
    n = len(weights)
    indices = np.arange(n)
    dev, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
                     .split(indices, groups=groups))
    folds = list(GroupKFold(n_splits=4).split(dev, groups=groups[dev]))
    targets = np.log1p(counts)
    records = []
    for alpha in (0., .15, .3, .45):
        similarity = np.clip(audio + alpha * residual, -1., 1.)
        for k in (5, 10, 20, 40):
            predictions = np.full(n, np.nan)
            baseline = np.full(n, np.nan)
            for train, val in folds:
                tr, va = dev[train], dev[val]
                model = PositiveKNN(similarity[np.ix_(tr, tr)], weights[tr], counts[tr], k=k)
                _, replay, *_ = model.predict(similarity[np.ix_(va, tr)])
                predictions[va] = replay
                baseline[va] = np.nanmedian(targets[tr])
            metrics = replay_metrics(targets[dev], predictions[dev])
            records.append({"k": k, "metadata_alpha": alpha, **metrics})
    eligible = [r for r in records if r["metadata_alpha"] > 0 and r["mae_log1p"] is not None]
    if not eligible:
        raise ValueError("Not enough observed playback counts for grouped parameter selection")
    best = min(eligible, key=lambda r: r["mae_log1p"])
    audio_best = min((r for r in records if r["metadata_alpha"] == 0), key=lambda r: r["mae_log1p"])
    similarity = np.clip(audio + best["metadata_alpha"] * residual, -1., 1.)
    model = PositiveKNN(similarity[np.ix_(dev, dev)], weights[dev], counts[dev], k=best["k"])
    _, pred, *_ = model.predict(similarity[np.ix_(test, dev)])
    median = float(np.nanmedian(targets[dev]))
    heldout = replay_metrics(targets[test], pred)
    baseline_test = replay_metrics(targets[test], np.full(len(test), median))
    assert not set(groups[dev]).intersection(groups[test])
    return best, {
        "method": "artist-connected-component 20% held-out; 4-fold grouped CV on development only",
        "seed": seed, "development_songs": len(dev), "heldout_songs": len(test),
        "development_groups": len(set(groups[dev])), "heldout_groups": len(set(groups[test])),
        "cv_grid": records, "selected": best, "best_audio_only_cv": audio_best,
        "cv_median_baseline": replay_metrics(targets[dev], baseline[dev]),
        "heldout_replay": heldout, "heldout_median_baseline": baseline_test,
        "limitation": "Replay-intensity prediction among favorites only; not like/dislike accuracy. All favorites remain positive regardless of counts.",
    }, {"development_indices": dev.tolist(), "heldout_indices": test.tolist(),
         "groups": groups.tolist()}


def percentile(raw, reference):
    reference = np.sort(np.asarray(reference))
    return (np.searchsorted(reference, raw, side="left") +
            np.searchsorted(reference, raw, side="right")) * 50. / len(reference)


def train(args):
    rows = read_json(args.labels)
    validate_labels(rows)
    liked_source = ROOT / "ncm-playback-dl/tmp/liked-files.txt"
    liked_names = {Path(line.strip().strip('"')).name
                   for line in liked_source.read_text(encoding="utf-8-sig").splitlines()
                   if line.strip()}
    label_names = {row["文件"] for row in rows}
    if label_names != liked_names:
        raise ValueError(f"Favorite labels differ from liked-files.txt: "
                         f"missing={len(liked_names-label_names)}, extra={len(label_names-liked_names)}; "
                         "run build_liked_labels.py")
    signals = read_json(args.signals)
    paths, cached = read_vectors(args.cache)
    cache = {Path(p).name: v for p, v in zip(paths, cached)}
    all_records, _ = link_signals(rows, signals)
    available = [i for i, row in enumerate(rows) if row["文件"] in cache]
    dropped = [row["文件"] for row in rows if row["文件"] not in cache]
    # Refuse to silently train on a partially completed embedding run.
    unexpected = [name for name in dropped if (args.root / name).exists()]
    if unexpected:
        raise ValueError(f"{len(unexpected)} existing favorites lack embeddings; run prepare_preference.py first")
    selected = [rows[i] for i in available]
    records = [all_records[i] for i in available]
    vectors = np.stack([cache[row["文件"]] for row in selected])
    weights = np.array([r["weight"] for r in records])
    counts = np.array([r["play_count"] if r["play_count_known"] else np.nan for r in records])
    print(f"Training on {len(selected)} favorites; observed counts={np.isfinite(counts).sum()}; missing audio={len(dropped)}", flush=True)
    audio, residual, _ = similarity_components(selected, selected, vectors, vectors)
    recordings = recording_groups(selected, audio)
    groups = artist_groups(selected, recordings)
    print(f"Artist components={len(set(groups))}; largest={max(np.bincount(groups))}; recording groups={len(set(recordings))}", flush=True)
    best, metrics, splits = evaluate_parameters(audio, residual, weights, counts, groups, args.seed)
    k, alpha = best["k"], best["metadata_alpha"]
    print(f"Selected k={k}, metadata_alpha={alpha}; heldout={metrics['heldout_replay']}", flush=True)
    # Reference percentiles: leave the entire recording group out, not merely
    # the query's filename. This is a reference distribution, not test accuracy.
    exclude = recordings[:, None] == recordings[None, :]
    combined = np.clip(audio + alpha * residual, -1., 1.)
    model = PositiveKNN(combined, weights, counts, k=k)
    calibration, *_ = model.predict(combined, exclude)
    audio_model = PositiveKNN(audio, weights, counts, k=k)
    calibration_audio, *_ = audio_model.predict(audio, exclude)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or DEFAULT_MODELS / stamp
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "model.npz", vectors=vectors.astype(np.float32),
                        weights=weights, counts=counts, recording_groups=recordings,
                        calibration=calibration, calibration_audio=calibration_audio)
    write_json(out / "songs.json", [{"labels": r, "behavior": b} for r, b in zip(selected, records)])
    write_json(out / "metrics.json", metrics)
    splits["files"] = [row["文件"] for row in selected]
    write_json(out / "splits.json", splits)
    listed = liked_source
    listed_names = {Path(s.strip()).name for s in listed.read_text(encoding="utf-8-sig").splitlines() if s.strip()} if listed.exists() else set()
    audit = {"total_favorites": len(rows), "trained_favorites": len(selected), "missing_audio": dropped,
             "matched_signals": sum(bool(r["signal_ids"]) for r in records),
             "known_play_counts": int(np.isfinite(counts).sum()),
             "unknown_play_counts": int(np.isnan(counts).sum()),
             "main_liked_playlist": sum(r["in_liked_playlist"] for r in records),
             "recording_groups": len(set(recordings)),
             "liked_file_list_unique": len(listed_names),
             "liked_file_list_not_in_labels": sorted(listed_names - {r["文件"] for r in rows}),
             "matching": "normalized artist+version-preserving title; duration within max(5s,3%); album disambiguation; no title-only or fuzzy fallback",
             "all_label_records": all_records}
    write_json(out / "data_audit.json", audit)
    config = {"format_version": 1, "encoder_id": ENCODER_ID, "preprocess_id": PREPROCESS_ID,
              "dimension": DIM, "k": k, "metadata_alpha": alpha, "tau": .1,
              "favorite_definition": "The unique files in liked-files.txt are favorites; signals do not create positive or negative labels",
              "weight_formula": "0.5 + 1 * in_main_liked_playlist + log1p(playCount^2) if known; unknown retains collection weight only",
              "score_definition": "0-100 percentile of leave-recording-group-out weighted KNN log support among favorite songs; NOT P(like)",
              "score_formula": "raw=log(mean_j(w_j * exp((similarity_j-1)/tau))), j in k nearest favorites",
              "metadata_tag_weights": TAG_WEIGHTS, "metadata_numeric_weights": NUMERIC_WEIGHTS,
              "labels_path": str(args.labels.resolve()), "signals_path": str(args.signals.resolve()),
              "labels_sha256": sha256(args.labels), "liked_files_sha256": sha256(liked_source),
              "signals_sha256": sha256(args.signals),
              "cache_sha256": sha256(args.cache), "trained_songs": len(selected),
              "created_at": datetime.now().astimezone().isoformat()}
    write_json(out / "config.json", config)
    report = make_report(config, audit, metrics, out)
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    write_json(DEFAULT_MODELS / "latest.json", {"model_dir": str(out.resolve())})
    print(f"Saved model and independent evaluation: {out}", flush=True)


def make_report(config, audit, metrics, out):
    held, baseline = metrics["heldout_replay"], metrics["heldout_median_baseline"]
    return f"""# 个人口味加权 KNN

全部标签记录都是收藏正样本。听歌记录只调整权重，不把低播放或未知次数标成不喜欢。

- 收藏 {audit['total_favorites']} 首，参与训练 {audit['trained_favorites']} 首；缺音频 {len(audit['missing_audio'])} 首，逐首见 data_audit.json。
- 训练集中关联听歌记录 {audit['matched_signals']} 首，已知播放次数 {audit['known_play_counts']} 首，未知 {audit['unknown_play_counts']} 首。
- 编码器：本地 MS-CLAP 2023，1024 维，44.1 kHz 单声道中间 7 秒，与旧缓存预处理一致。
- 权重：`0.5 + 1×是否进入“喜欢的音乐” + log(1+已知播放次数²)`。未知次数不加对数项。
- KNN：k={config['k']}，元数据混合系数 {config['metadata_alpha']}，温度 {config['tau']}。
- 曲风、细分、语种、BPM、年份、时长和其他有效标签共同计算邻近程度；缺失字段没有影响，弱规则标签降权。
- 新歌未提供标签时直接用音频近邻，并切换到音频专用参考分布。
- 分数为相对于收藏集的加权近邻支持度百分位，**不是喜欢概率**。50 分代表接近收藏参考中位水平，低分代表与现有收藏距离较远。

## 独立验证

先按相连歌手（含合作歌手）和重复录音分组留出测试集，仅在开发集做 4 折选参。测试集不参与选参。选参辅助目标是已收藏歌曲的 `log(1+播放次数)` 预测误差；只对已知次数评价。这无法验证新歌喜欢/不喜欢的准确率。

- 开发集 {metrics['development_songs']} 首；独立测试集 {metrics['heldout_songs']} 首，其中 {held['n']} 首可评价。
- 测试集播放强度 MAE（log1p）：{held['mae_log1p']:.4f}；训练中位数基线 {baseline['mae_log1p']:.4f}。
- 测试集 Spearman：{held['spearman']}。
- 完整网格、音频基线、划分和数据指纹见 metrics.json、splits.json、config.json。

参考分布按完整录音组留出自身计算，避免同一首歌及重复文件自匹配。它用于解释分数，不能当作独立验证结果。用全部可用收藏拟合并保存最终模型；无明确不喜欢样本，不能报告二分类准确率、AUC 或喜欢概率。

## 使用

在 <project-root>\\genre_model 执行：

```powershell
$env:PYTHONIOENCODING='utf-8'
$py = 'python'
& $py .\\preference_knn.py predict '<project-root>\\某首歌.mp3'
& $py .\\preference_knn.py predict '<project-root>\\某首歌.mp3' --metadata '{{"风格":"摇滚","BPM":140}}'
```

模型目录：`{out}`。预测会给出匹配分、相似收藏、播放次数及各近邻的贡献权重。
"""


class PreferenceModel:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.config = read_json(self.directory / "config.json")
        if self.config.get("format_version") != 1 or self.config.get("encoder_id") != ENCODER_ID or self.config.get("preprocess_id") != PREPROCESS_ID:
            raise ValueError("Incompatible model/encoder/preprocessing version")
        songs = read_json(self.directory / "songs.json")
        self.rows = [s["labels"] for s in songs]
        self.behavior = [s["behavior"] for s in songs]
        with np.load(self.directory / "model.npz", allow_pickle=False) as data:
            self.vectors = unit_rows(data["vectors"])
            self.weights, self.counts = data["weights"], data["counts"]
            self.recordings = data["recording_groups"]
            self.calibration, self.calibration_audio = data["calibration"], data["calibration_audio"]
        if self.vectors.shape != (len(self.rows), DIM):
            raise ValueError("Model shape mismatch")
        self.by_name = {r["文件"]: i for i, r in enumerate(self.rows)}
        self._models = {}

    def _index(self, tagged):
        if tagged not in self._models:
            audio, residual, _ = similarity_components(self.rows, self.rows, self.vectors, self.vectors)
            sim = audio + (self.config["metadata_alpha"] * residual if tagged else 0)
            self._models[tagged] = PositiveKNN(np.clip(sim, -1., 1.), self.weights, self.counts,
                                              self.config["k"], self.config["tau"])
        return self._models[tagged]

    def predict_vector(self, vector, metadata=None, filename=None):
        row = metadata or {}
        tagged = any(confidence(row, field) > 0 for field in TAG_WEIGHTS) or any(
            row.get(field) for field in NUMERIC_WEIGHTS)
        vector = unit_rows(np.asarray(vector)[None])
        audio, residual, coverage = similarity_components([row], self.rows, vector, self.vectors)
        similarity = audio + (self.config["metadata_alpha"] * residual if tagged else 0)
        # Audio-identical windows are excluded even for an unseen filename.
        exclude = audio > .995
        known = self.by_name.get(filename)
        if known is not None:
            exclude |= self.recordings[None, :] == self.recordings[known]
        raw, replay, neighbors, sims, valid = self._index(tagged).predict(similarity, exclude)
        reference = self.calibration if tagged else self.calibration_audio
        score = float(percentile(raw, reference)[0])
        ids = neighbors[0][valid[0]]
        contributions = np.exp((sims[0][valid[0]] - 1.) / self.config["tau"]) * self.weights[ids]
        contributions /= contributions.sum()
        result = {"file": filename, "already_favorite": known is not None,
                  "match_score": round(score, 2), "raw_support": float(raw[0]),
                  "score_meaning": "收藏集近邻支持度百分位，不是喜欢概率",
                  "query_features": "audio_and_metadata" if tagged else "audio_only",
                  "k": len(ids), "nearest_audio_cosine": float(np.max(audio[0][~exclude[0]])),
                  "neighbors": []}
        for j, sim, contribution in zip(ids.tolist(), sims[0][valid[0]].tolist(), contributions.tolist()):
            result["neighbors"].append({"file": self.rows[j]["文件"], "genre": self.rows[j].get("风格"),
                                        "similarity": round(sim, 5), "audio_cosine": round(float(audio[0, j]), 5),
                                        "metadata_coverage": round(float(coverage[0, j]), 5),
                                        "play_count": self.behavior[j]["play_count"],
                                        "behavior_weight": round(float(self.weights[j]), 4),
                                        "contribution": round(contribution, 5)})
        return result


def model_directory(path):
    return path if path else Path(read_json(DEFAULT_MODELS / "latest.json")["model_dir"])


def predict(args):
    model = PreferenceModel(model_directory(args.model))
    audio_path = args.audio.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    name = audio_path.name
    known = model.by_name.get(name) if audio_path.parent == args.root.resolve() else None
    cached_index = known if known is not None else None
    if cached_index is not None:
        vector = model.vectors[cached_index]
    else:
        vector = AudioEmbedder().embed_file(audio_path)
    metadata = model.rows[known] if known is not None and not args.audio_only else None
    if args.metadata and args.audio_only:
        raise ValueError("--audio-only and --metadata cannot be combined")
    if args.metadata:
        metadata = read_json(args.metadata[1:]) if args.metadata.startswith("@") else json.loads(args.metadata)
    result = model.predict_vector(vector, metadata, name if known is not None else None)
    result["file"] = str(audio_path)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(f"{name}\n口味匹配分：{result['match_score']:.1f}/100（收藏参考百分位，不是概率）")
        print("特征：" + ("音频 + 已知标签" if result["query_features"] == "audio_and_metadata" else "音频；未提供标签"))
        if result["already_favorite"]:
            print("这是已有收藏；已排除自身及重复录音。")
        print(f"最近的收藏（模型用 {result['k']} 首近邻）：")
        for item in result["neighbors"][:args.explain]:
            count = item["play_count"] if item["play_count"] is not None else "未知"
            print(f"  {item['file']} | 风格={item['genre'] or '未知'} | 相似度={item['similarity']:.3f} | 播放={count} | 贡献={item['contribution']:.1%}")
    if args.output:
        write_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    tr = commands.add_parser("train", help="Fit and evaluate a positive-only weighted KNN")
    tr.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    tr.add_argument("--signals", type=Path, default=DEFAULT_SIGNALS)
    tr.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    tr.add_argument("--root", type=Path, default=ROOT)
    tr.add_argument("--out", type=Path)
    tr.add_argument("--seed", type=int, default=42)
    pr = commands.add_parser("predict", help="Score a song and explain nearest favorites")
    pr.add_argument("audio", type=Path)
    pr.add_argument("--model", type=Path)
    pr.add_argument("--root", type=Path, default=ROOT)
    pr.add_argument("--metadata", help="JSON object or @path/to/metadata.json")
    pr.add_argument("--audio-only", action="store_true")
    pr.add_argument("--json", action="store_true")
    pr.add_argument("--explain", type=int, default=5)
    pr.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
