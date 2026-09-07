"""Benchmark Flat, HNSW, IVF, and standalone PQ on shared MS MARCO embeddings.

The runner measures index build time, serialized-index size, per-query search
latency, and recall@k against the exact Flat L2 baseline. It intentionally
imports index *wrappers*, rather than importing the existing runner scripts:
those scripts parse command-line arguments and rebuild indexes at import time.

Example:
    python benchmark.py
    python benchmark.py --k 10 --output results/benchmark_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from functools import partial
from typing import Any, Callable, Optional, Protocol

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from hnsw_index import HNSWIndex
from ivf_index import IVFIndex
from pq_index import PQIndex
from test_queries import TEST_QUERIES


PROJECT_DIR = Path(__file__).resolve().parent
PASSAGES_PATH = PROJECT_DIR / "data" / "passages.csv"
EMBEDDINGS_PATH = PROJECT_DIR / "data" / "embeddings.npy"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "benchmark_results.csv"
DEFAULT_K = 5

# Keep these explicit so an experiment can record and later sweep them.
HNSW_PARAMETERS = {"M": 32, "efConstruction": 200, "efSearch": 128}
# FAISS recommends roughly 39 training vectors per IVF centroid. Thus, use at
# most about 51 lists for this 2K corpus; 50 is a safe default. For planned
# 10K/50K/200K corpora, suitable upper bounds are about 256/1282/5128 lists.
DEFAULT_IVF_NLIST = 50
IVF_NPROBE = 10
PQ_PARAMETERS = {"m": 48, "nbits": 4}


class SearchableIndex(Protocol):
    def search(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...

    def save(self, path: str | Path) -> None: ...


class FlatIndexAdapter:
    """Small adapter for Flat, whose reference implementation is procedural."""

    def __init__(self, dimension: int) -> None:
        self.index = faiss.IndexFlatL2(dimension)

    def build(self, vectors: np.ndarray) -> None:
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.index.search(np.ascontiguousarray(query_vectors, dtype=np.float32), k)

    def save(self, path: str | Path) -> None:
        faiss.write_index(self.index, str(path))


@dataclass
class IndexRun:
    index_type: str
    index: SearchableIndex
    build_time_s: float
    memory_mb: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-k results (default: 5).")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="CSV results path."
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=None,
        help="Use only the first N fixed queries for a shorter smoke test.",
    )
    parser.add_argument(
        "--ivf-nlist",
        type=int,
        default=DEFAULT_IVF_NLIST,
        help="IVF coarse clusters (default: 50, appropriate for the 2K corpus).",
    )
    return parser.parse_args()


def load_corpus() -> tuple[pd.DataFrame, np.ndarray]:
    """Load the one corpus/embedding snapshot shared by every benchmarked index."""
    missing = [str(path) for path in (PASSAGES_PATH, EMBEDDINGS_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing baseline artifacts: " + ", ".join(missing) + ". Run `python run_pipeline.py` first."
        )
    passages = pd.read_csv(PASSAGES_PATH)
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32, copy=False)
    if embeddings.ndim != 2 or len(passages) != len(embeddings):
        raise ValueError("passages.csv and embeddings.npy must have matching, two-dimensional data")
    return passages, embeddings


def build_flat(vectors: np.ndarray) -> SearchableIndex:
    index = FlatIndexAdapter(vectors.shape[1])
    index.build(vectors)
    return index


def build_hnsw(vectors: np.ndarray) -> SearchableIndex:
    index = HNSWIndex(vectors.shape[1], **HNSW_PARAMETERS)
    index.build(vectors)
    return index


def build_ivf(vectors: np.ndarray, nlist: int = DEFAULT_IVF_NLIST) -> SearchableIndex:
    if nlist < 1:
        raise ValueError("IVF nlist must be a positive integer")
    effective_nlist = min(nlist, len(vectors))
    index = IVFIndex(nlist=effective_nlist, nprobe=min(IVF_NPROBE, effective_nlist))
    index.build(vectors)
    return index


def build_pq(vectors: np.ndarray) -> SearchableIndex:
    index = PQIndex(vectors.shape[1], **PQ_PARAMETERS)
    index.build(vectors)
    return index


INDEX_BUILDERS: tuple[tuple[str, Callable[[np.ndarray], SearchableIndex]], ...] = (
    ("flat", build_flat),
    ("hnsw", build_hnsw),
    ("ivf", build_ivf),
    ("pq", build_pq),
)


def build_and_measure(index_type: str, builder: Callable[[np.ndarray], SearchableIndex], vectors: np.ndarray) -> IndexRun:
    """Build an index and measure its native FAISS serialized size on disk."""
    started = time.perf_counter()
    index = builder(vectors)
    build_time_s = time.perf_counter() - started
    # Temporary serialization avoids overwriting each developer's reusable index.
    with tempfile.TemporaryDirectory(prefix=f"rag-benchmark-{index_type}-") as directory:
        path = Path(directory) / f"{index_type}.index"
        index.save(path)
        memory_mb = path.stat().st_size / (1024 * 1024)
    return IndexRun(index_type, index, build_time_s, memory_mb)


def valid_ids(ids: np.ndarray, corpus_size: int) -> list[int]:
    """Remove FAISS's -1 missing-result marker before recall/payload handling."""
    return [int(vector_id) for vector_id in ids if 0 <= int(vector_id) < corpus_size]


def recall_at_k(actual_ids: list[int], ground_truth_ids: list[int]) -> Optional[float]:
    if not ground_truth_ids:
        return None
    return len(set(actual_ids).intersection(ground_truth_ids)) / len(ground_truth_ids)


def result_payload(passages: pd.DataFrame, ids: list[int]) -> tuple[str, str]:
    return json.dumps(ids), json.dumps([str(passages.iloc[idx]["text"]) for idx in ids])


CSV_COLUMNS = [
    "index_type",
    "query_id",
    "query_text",
    "latency_ms",
    "recall_at_5",
    "build_time_s",
    "memory_mb",
    "returned_ids",
    "returned_texts",
    "error_message",
]


def failed_row(index_type: str, query_id: int, query: str, error: Exception, run: Optional[IndexRun] = None) -> dict[str, Any]:
    return {
        "index_type": index_type,
        "query_id": query_id,
        "query_text": query,
        "latency_ms": None,
        "recall_at_5": None,
        "build_time_s": None if run is None else run.build_time_s,
        "memory_mb": None if run is None else run.memory_mb,
        "returned_ids": "[]",
        "returned_texts": "[]",
        "error_message": f"{type(error).__name__}: {error}",
    }


def benchmark_index(
    run: IndexRun,
    query_embeddings: np.ndarray,
    queries: list[str],
    passages: pd.DataFrame,
    k: int,
    flat_result_ids: list[list[int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_id, query in enumerate(queries):
        try:
            # TODO: repeat this search N times here and report p50/p95 latency.
            started = time.perf_counter_ns()
            _, ids = run.index.search(query_embeddings[query_id : query_id + 1], k)
            latency_ms = (time.perf_counter_ns() - started) / 1_000_000
            returned_ids = valid_ids(ids[0], len(passages))
            ids_json, texts_json = result_payload(passages, returned_ids)
            rows.append(
                {
                    "index_type": run.index_type,
                    "query_id": query_id,
                    "query_text": query,
                    "latency_ms": latency_ms,
                    # Flat is the exact ground truth, so its self-comparison
                    # is always recall 1.0 when its search succeeded.
                    "recall_at_5": 1.0
                    if run.index_type == "flat"
                    else recall_at_k(returned_ids, flat_result_ids[query_id]),
                    "build_time_s": run.build_time_s,
                    "memory_mb": run.memory_mb,
                    "returned_ids": ids_json,
                    "returned_texts": texts_json,
                    "error_message": "",
                }
            )
        except Exception as error:  # Keep another index/query from being lost to one failure.
            print(f"WARNING: {run.index_type} failed for query {query_id}: {error}")
            rows.append(failed_row(run.index_type, query_id, query, error, run))
    return rows


def write_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    rows: list[dict[str, Any]],
    index_builders: tuple[tuple[str, Callable[[np.ndarray], SearchableIndex]], ...],
) -> None:
    print("\nBenchmark summary (index size is serialized FAISS file size):")
    print(f"{'Index':<8} {'Avg latency (ms)':>17} {'Avg recall@k':>15} {'Build (s)':>12} {'Disk (MB)':>11}")
    for index_type, _ in index_builders:
        index_rows = [row for row in rows if row["index_type"] == index_type]
        successful = [row for row in index_rows if row["latency_ms"] is not None]
        if not index_rows:
            continue
        latency = np.mean([row["latency_ms"] for row in successful]) if successful else float("nan")
        recalls = [row["recall_at_5"] for row in successful if row["recall_at_5"] is not None]
        recall_text = f"{np.mean(recalls):.3f}" if recalls else "n/a"
        print(
            f"{index_type:<8} {latency:>17.3f} {recall_text:>15} "
            f"{index_rows[0]['build_time_s']:>12.3f} {index_rows[0]['memory_mb']:>11.3f}"
        )


def main() -> None:
    args = parse_arguments()
    if args.k < 1:
        raise ValueError("--k must be a positive integer")
    if args.query_limit is not None and args.query_limit < 1:
        raise ValueError("--query-limit must be a positive integer")
    if args.ivf_nlist < 1:
        raise ValueError("--ivf-nlist must be a positive integer")

    passages, embeddings = load_corpus()
    if args.k > len(passages):
        raise ValueError("--k cannot exceed the number of corpus passages")
    queries = TEST_QUERIES if args.query_limit is None else TEST_QUERIES[: args.query_limit]

    print(f"Embedding {len(queries)} fixed benchmark queries once...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embeddings = model.encode(queries).astype(np.float32, copy=False)

    all_rows: list[dict[str, Any]] = []
    flat_result_ids: list[list[int]] = [[] for _ in queries]
    index_builders = tuple(
        (index_type, partial(build_ivf, nlist=args.ivf_nlist))
        if index_type == "ivf"
        else (index_type, builder)
        for index_type, builder in INDEX_BUILDERS
    )
    # TODO: wrap this block in a corpus-size loop for 10K/50K/200K experiments.
    for index_type, builder in index_builders:
        print(f"Building {index_type} index...")
        try:
            run = build_and_measure(index_type, builder, embeddings)
        except Exception as error:
            print(f"WARNING: could not build {index_type}: {error}")
            all_rows.extend(failed_row(index_type, i, query, error) for i, query in enumerate(queries))
            continue
        rows = benchmark_index(run, query_embeddings, queries, passages, args.k, flat_result_ids)
        all_rows.extend(rows)
        if index_type == "flat":
            flat_result_ids = [json.loads(row["returned_ids"]) for row in rows]

    output_path = args.output if args.output.is_absolute() else PROJECT_DIR / args.output
    write_results(all_rows, output_path)
    print_summary(all_rows, index_builders)
    print(f"\nWrote {len(all_rows)} per-query rows to {output_path}")


if __name__ == "__main__":
    main()
