"""Choose an index from measured benchmark results without rebuilding an index."""

# How the score works:
# First, discard failed benchmark runs and any index that breaks a requested
# recall, latency, or index-size limit. Among the remaining indexes, rescale
# each measurement to 0..1, where 1 is best: high recall is good, while low
# latency and small on-disk index size are good. If a measurement is tied, all
# indexes get 1 for it; a sole survivor gets 1 for every measurement. Multiply
# those three scores by the selected preference weights and add them. The
# highest total wins. Build time is shown elsewhere but is not scored here.

from pathlib import Path

import pandas as pd


WEIGHTS = {
    "fastest": {"latency": 0.7, "recall": 0.2, "memory": 0.1},
    "accurate": {"latency": 0.2, "recall": 0.7, "memory": 0.1},
    "smallest": {"latency": 0.1, "recall": 0.2, "memory": 0.7},
    "balanced": {"latency": 1 / 3, "recall": 1 / 3, "memory": 1 / 3},
}


def _normalise(values: list[float], higher_is_better: bool) -> list[float]:
    low, high = min(values), max(values)
    if low == high:
        return [1.0] * len(values)
    if higher_is_better:
        return [(value - low) / (high - low) for value in values]
    return [(high - value) / (high - low) for value in values]


def select_index(
    summary_df: pd.DataFrame,
    corpus_size: int,
    preset: str = "balanced",
    min_recall: float | None = None,
    max_latency_ms: float | None = None,
    max_memory_mb: float | None = None,
    latency_metric: str = "p95",
) -> dict:
    """Rank successful indexes at the nearest available corpus size."""
    if preset not in WEIGHTS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {', '.join(WEIGHTS)}")
    if latency_metric not in ("p50", "p95"):
        raise ValueError("latency_metric must be 'p50' or 'p95'")

    successful = summary_df.loc[summary_df["status"].eq("success")]
    sizes = sorted(successful["corpus_size"].unique())
    if not sizes:
        raise ValueError("No successful benchmark rows are available")
    used_size = min(sizes, key=lambda size: (abs(size - corpus_size), size))
    rows = successful.loc[successful["corpus_size"].eq(used_size)]
    latency_col = f"{latency_metric}_latency_ms"
    weights = WEIGHTS[preset].copy()
    prefix = f"At {used_size:,} passages"
    if used_size != corpus_size:
        prefix += f" (nearest benchmarked size to the requested {corpus_size:,})"

    table = []
    violation_totals = {}
    for row in rows.itertuples():
        recall = float(row.avg_recall_at_k)
        latency = float(getattr(row, latency_col))
        memory = float(row.memory_mb)
        violations = []
        total = 0.0
        if min_recall is not None and recall < min_recall:
            amount = min_recall - recall
            violations.append(f"recall {recall:.3f} is {amount:.3f} below {min_recall:.3f}")
            total += amount / max(abs(min_recall), 1e-9)
        if max_latency_ms is not None and latency > max_latency_ms:
            amount = latency - max_latency_ms
            violations.append(f"{latency_metric} latency {latency:.3f} ms is {amount:.3f} ms above {max_latency_ms:.3f} ms")
            total += amount / max(abs(max_latency_ms), 1e-9)
        if max_memory_mb is not None and memory > max_memory_mb:
            amount = memory - max_memory_mb
            violations.append(f"memory (index size) {memory:.3f} MB is {amount:.3f} MB above {max_memory_mb:.3f} MB")
            total += amount / max(abs(max_memory_mb), 1e-9)
        if violations:
            violation_totals[row.index_type] = total
        table.append({
            "index_type": row.index_type,
            "status": "rejected" if violations else "eligible",
            "score": None,
            "recall": recall,
            "latency": latency,
            "memory_mb": memory,
            "reason": "; ".join(violations),
        })

    eligible = [row for row in table if row["status"] == "eligible"]
    if not eligible:
        closest = min(table, key=lambda row: (violation_totals[row["index_type"]], row["index_type"]))
        other_rejections = "; ".join(
            f"{row['index_type']} ({row['reason']})" for row in table if row is not closest
        )
        explanation = (
            f"{prefix}, no index meets all constraints. The closest miss is "
            f"{closest['index_type']}: {closest['reason']}."
        )
        if other_rejections:
            explanation += f" Other rejected indexes: {other_rejections}."
        return {
            "winner": None, "closest_miss": closest["index_type"],
            "table": table, "weights": weights,
            "corpus_size_used": used_size, "explanation": explanation,
        }

    normalised = {
        "latency": _normalise([row["latency"] for row in eligible], False),
        "recall": _normalise([row["recall"] for row in eligible], True),
        "memory": _normalise([row["memory_mb"] for row in eligible], False),
    }
    for position, row in enumerate(eligible):
        row["score"] = sum(weights[metric] * values[position] for metric, values in normalised.items())

    winner = min(eligible, key=lambda row: (-row["score"], row["index_type"]))
    winner["status"] = "recommended"
    winner["reason"] = f"Highest weighted score ({winner['score']:.3f})"
    for row in eligible:
        if row is not winner:
            row["reason"] = f"Weighted score {row['score']:.3f} below {winner['score']:.3f}"

    comparisons = "; ".join(
        f"{row['index_type']} ({row['reason']})" for row in table if row is not winner
    )
    explanation = (
        f"{prefix}, {winner['index_type']} is recommended with score {winner['score']:.3f} "
        f"under the {preset} preset (latency {weights['latency']:.0%}, "
        f"recall {weights['recall']:.0%}, index size {weights['memory']:.0%}). "
        f"Its recall is {winner['recall']:.3f}, {latency_metric} latency is "
        f"{winner['latency']:.3f} ms, and on-disk index size is "
        f"{winner['memory_mb']:.3f} MB."
    )
    if comparisons:
        explanation += f" Other indexes: {comparisons}."
    return {
        "winner": winner["index_type"], "closest_miss": None,
        "table": table, "weights": weights,
        "corpus_size_used": used_size, "explanation": explanation,
    }


def recommend_for_all_sizes(summary_df: pd.DataFrame, preset: str, **constraints) -> pd.DataFrame:
    """Apply the same rule to each successfully benchmarked corpus size."""
    sizes = sorted(summary_df.loc[summary_df["status"].eq("success"), "corpus_size"].unique())
    recommendations = []
    for size in sizes:
        result = select_index(summary_df, size, preset, **constraints)
        recommendations.append({
            "corpus_size": size,
            "winner": result["winner"],
            "explanation": result["explanation"],
        })
    return pd.DataFrame(recommendations, columns=["corpus_size", "winner", "explanation"])


if __name__ == "__main__":
    summary = pd.read_csv(Path(__file__).with_name("benchmark_summary.csv"))
    largest = summary["corpus_size"].max()
    examples = [
        ("Balanced", {"preset": "balanced"}),
        ("Accurate, recall >= 0.95", {"preset": "accurate", "min_recall": 0.95}),
        ("Smallest, latency <= 10 ms", {"preset": "smallest", "max_latency_ms": 10}),
    ]
    for label, options in examples:
        result = select_index(summary, largest, **options)
        print(f"{label}: {result['explanation']}")
