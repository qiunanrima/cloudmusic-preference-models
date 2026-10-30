"""Regression checks for leakage, missing signals and weighted KNN semantics."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from msclap_audio import DIM, ENCODER_ID, read_vectors, write_vectors
from preference_data import link_signals
from preference_knn import (PositiveKNN, artist_groups, percentile, recording_groups,
                            similarity_components)


def row(artist="Singer", title="Song", **extra):
    return {"文件": f"{artist} - {title}.mp3", "歌手": artist, "歌名": title, **extra}


def song(artist="Singer", title="Song", count=None, **extra):
    return {"id": "1", "name": title, "artists": [artist], "playCount": count, "at": {}, **extra}


class SignalTests(unittest.TestCase):
    def signals(self, *songs):
        return {"playlists": [{"id": "liked", "name": "测试喜欢的音乐"}], "songs": list(songs)}

    def test_unknown_stays_unknown_and_positive(self):
        result, _ = link_signals([row()], self.signals(song(count=None, at={"liked": None})))
        self.assertIsNone(result[0]["play_count"])
        self.assertFalse(result[0]["play_count_known"])
        self.assertEqual(result[0]["weight"], 1.5)

    def test_zero_is_known_and_not_dislike(self):
        result, _ = link_signals([row()], self.signals(song(count=0)))
        self.assertTrue(result[0]["play_count_known"])
        self.assertEqual(result[0]["play_count"], 0)
        self.assertEqual(result[0]["weight"], 0.5)

    def test_foreign_artist_same_title_rejected(self):
        result, _ = link_signals([row()], self.signals(song(artist="Another Singer", count=999)))
        self.assertEqual(result[0]["match_status"], "unmatched")
        self.assertEqual(result[0]["weight"], 0.5)

    def test_off_vocal_version_stays_distinct(self):
        result, _ = link_signals([row(title="Song (off vocal)")], self.signals(song(count=99)))
        self.assertEqual(result[0]["match_status"], "unmatched")

    def test_duration_guard(self):
        result, _ = link_signals([row(**{"时长秒": 180})],
                                self.signals(song(count=99, durationMs=200000)))
        self.assertEqual(result[0]["match_status"], "unmatched")

    def test_alternate_ids_do_not_double_playcounts(self):
        result, _ = link_signals([row()], self.signals(song(count=12), song(count=20, id="2")))
        self.assertEqual(result[0]["play_count"], 20)
        self.assertAlmostEqual(result[0]["weight"], .5 + np.log1p(20**2))

    def test_different_albums_are_ambiguous(self):
        result, _ = link_signals([row()], self.signals(song(album="A"), song(album="B", id="2")))
        self.assertEqual(result[0]["match_status"], "ambiguous_album")

    def test_collaborator_matches_without_substring(self):
        result, _ = link_signals([row(artist="Singer,Guest")], self.signals(song(artist="Guest")))
        self.assertTrue(result[0]["signal_ids"])
        result, _ = link_signals([row(artist="Li")], self.signals(song(artist="Lily")))
        self.assertFalse(result[0]["signal_ids"])


class SimilarityTests(unittest.TestCase):
    def test_missing_metadata_has_no_effect(self):
        audio, residual, coverage = similarity_components([{}], [row(**{"风格": "摇滚", "BPM": 120})],
                                                          [[1., 0.]], [[.8, .6]])
        np.testing.assert_array_equal(residual, 0)
        np.testing.assert_array_equal(coverage, 0)
        self.assertAlmostEqual(float(audio[0, 0]), .8, places=5)

    def test_matching_genre_increases_similarity(self):
        q = row(**{"风格": "摇滚"})
        refs = [row(**{"风格": "摇滚"}), row(**{"风格": "流行"})]
        audio, residual, _ = similarity_components([q], refs, [[1., 0.]], [[.8, .6], [.8, .6]])
        self.assertGreater(float(residual[0, 0]), 0)
        self.assertLess(float(residual[0, 1]), 0)

    def test_conflicting_tags_are_ignored(self):
        q = row(**{"风格": "摇滚", "依据": {"风格": {"状态": "冲突待复核"}}})
        _, residual, _ = similarity_components([q], [row(**{"风格": "摇滚"})], [[1., 0.]], [[.8, .6]])
        np.testing.assert_array_equal(residual, 0)

    def test_weak_rule_has_less_influence(self):
        strong = row(**{"风格": "摇滚"})
        weak = row(**{"风格": "摇滚", "依据": {"风格": {"状态": "规则推断"}}})
        _, residual, _ = similarity_components([strong, weak], [strong], [[1., 0.], [1., 0.]], [[.8, .6]])
        self.assertAlmostEqual(float(residual[1, 0] / residual[0, 0]), .4, places=5)

    def test_recording_duplicates_grouped(self):
        rows = [row(title="A"), row(title="B"), row(artist="Other", title="C")]
        sim = np.eye(3)
        sim[0, 2] = sim[2, 0] = 1
        groups = recording_groups(rows, sim)
        self.assertEqual(groups[0], groups[2])
        self.assertNotEqual(groups[0], groups[1])

    def test_collaborators_cannot_cross_artist_split(self):
        rows = [row(artist="A,B"), row(artist="B,C"), row(artist="C"), row(artist="D")]
        groups = artist_groups(rows, np.arange(4))
        self.assertEqual(groups[0], groups[2])
        self.assertNotEqual(groups[0], groups[3])


class KNNTests(unittest.TestCase):
    def test_support_increases_with_playback_weight(self):
        similarity = np.eye(2)
        one = PositiveKNN(similarity, [1., 1.], [np.nan, 3.], k=1)
        boosted = PositiveKNN(similarity, [5., 1.], [np.nan, 3.], k=1)
        raw1, *_ = one.predict([[.9, .3]])
        raw2, *_ = boosted.predict([[.9, .3]])
        self.assertGreater(raw2[0], raw1[0])

    def test_known_counts_only_for_replay_target(self):
        model = PositiveKNN(np.eye(2), [1., 1.], [np.nan, 3.], k=2)
        _, replay, *_ = model.predict([[.9, .5]])
        self.assertAlmostEqual(replay[0], np.log1p(3))

    def test_all_unknown_replay_is_nan(self):
        model = PositiveKNN(np.eye(2), [1., 1.], [np.nan, np.nan], k=2)
        raw, replay, *_ = model.predict([[.9, .5]])
        self.assertTrue(np.isfinite(raw[0]))
        self.assertTrue(np.isnan(replay[0]))

    def test_excluded_recording_never_a_neighbor(self):
        model = PositiveKNN(np.eye(3), [1., 2., 3.], [1., 2., 3.], k=3)
        _, _, ids, _, valid = model.predict([[1., .9, .8]], np.array([[True, True, False]]))
        self.assertEqual(ids[0][valid[0]].tolist(), [2])

    def test_all_excluded_errors(self):
        model = PositiveKNN(np.eye(2), [1., 1.], [1., 2.], k=1)
        with self.assertRaises(ValueError):
            model.predict([[1., .5]], np.ones((1, 2), dtype=bool))

    def test_percentile_ties_and_bounds(self):
        np.testing.assert_array_equal(percentile(np.array([-1., 2., 5.]), [1., 2., 3.]), [0., 50., 100.])

    def test_model_dimension_guard_and_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cache.npz"
            vec = np.zeros((2, DIM), dtype=np.float32)
            vec[0, 0], vec[1, 1] = 1, 1
            write_vectors(path, ["a.mp3", "b.mp3"], vec)
            names, loaded = read_vectors(path)
            self.assertEqual(names, ["a.mp3", "b.mp3"])
            np.testing.assert_array_equal(loaded, vec)
            np.savez(path, paths=np.array(names), vectors=np.ones((2, 512)), model_id=ENCODER_ID)
            with self.assertRaises(ValueError):
                read_vectors(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
