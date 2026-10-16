"""Focused tests for the rule-based benchmark selector."""

import unittest

import pandas as pd

from selector import recommend_for_all_sizes, select_index


def sample_summary() -> pd.DataFrame:
    rows = [
        (2000, "flat", 1.00, 10.0, 12.0, 12.0, "success"),
        (2000, "hnsw", 0.96, 3.0, 4.0, 6.0, "success"),
        (2000, "pq", 0.70, 5.0, 6.0, 1.0, "success"),
        (10000, "flat", 1.00, 20.0, 24.0, 30.0, "success"),
        (10000, "hnsw", 0.95, 6.0, 8.0, 12.0, "success"),
        (10000, "broken", 1.00, 0.1, 0.1, 0.1, "failed"),
    ]
    return pd.DataFrame(rows, columns=[
        "corpus_size", "index_type", "avg_recall_at_k", "p50_latency_ms",
        "p95_latency_ms", "memory_mb", "status",
    ])


class SelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summary = sample_summary()

    def test_constraints_reject_rows_with_amounts(self) -> None:
        result = select_index(
            self.summary, 2000, min_recall=0.95,
            max_latency_ms=5, max_memory_mb=7,
        )
        self.assertEqual(result["winner"], "hnsw")
        table = {row["index_type"]: row for row in result["table"]}
        self.assertEqual(table["hnsw"]["status"], "recommended")
        self.assertEqual(table["flat"]["status"], "rejected")
        self.assertIn("latency", table["flat"]["reason"])
        self.assertIn("memory", table["flat"]["reason"])
        self.assertIn("7.00", table["flat"]["reason"])
        self.assertEqual(table["pq"]["status"], "rejected")
        self.assertIn("recall", table["pq"]["reason"])
        self.assertNotIn("broken", table)

    def test_no_survivor_reports_closest_miss(self) -> None:
        result = select_index(self.summary, 2000, min_recall=1.01)
        self.assertIsNone(result["winner"])
        self.assertEqual(result["closest_miss"], "flat")
        self.assertIn("no index meets", result["explanation"])
        self.assertIn("flat", result["explanation"])
        self.assertTrue(all(row["status"] == "rejected" for row in result["table"]))

    def test_single_survivor_has_full_score(self) -> None:
        result = select_index(self.summary, 2000, min_recall=1.0)
        self.assertEqual(result["winner"], "flat")
        flat = next(row for row in result["table"] if row["index_type"] == "flat")
        self.assertAlmostEqual(flat["score"], 1.0)

    def test_presets_choose_different_winners(self) -> None:
        summary = self.summary.copy()
        summary.loc[(summary["corpus_size"] == 2000) & (summary["index_type"] == "hnsw"), "avg_recall_at_k"] = 0.85
        expected = {"fastest": "hnsw", "accurate": "flat", "smallest": "pq"}
        for preset, winner in expected.items():
            with self.subTest(preset=preset):
                self.assertEqual(select_index(summary, 2000, preset)["winner"], winner)
        with self.assertRaises(ValueError):
            select_index(self.summary, 2000, "unknown")

    def test_nearest_size_uses_smaller_size_on_tie(self) -> None:
        result = select_index(self.summary, 6000)
        self.assertEqual(result["corpus_size_used"], 2000)
        self.assertIn("6,000", result["explanation"])
        self.assertIn("2,000", result["explanation"])

    def test_recommend_for_all_sizes(self) -> None:
        result = recommend_for_all_sizes(self.summary, "balanced")
        self.assertEqual(result["corpus_size"].tolist(), [2000, 10000])
        self.assertEqual(result["winner"].tolist(), ["hnsw", "hnsw"])


if __name__ == "__main__":
    unittest.main()
