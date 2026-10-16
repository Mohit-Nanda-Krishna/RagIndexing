"""Benchmark Flat, HNSW, IVF, and PQ on ordered MS MARCO corpus prefixes.

Run ``python benchmark.py`` to process the default sizes, or provide explicit
sizes with ``python benchmark.py --sizes 2000 10000``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
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
DATA_DIR = PROJECT_DIR / "data"
PASSAGES_PATH = DATA_DIR / "passages.csv"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
RESULTS_PATH = PROJECT_DIR / "benchmark_results.csv"
SUMMARY_PATH = PROJECT_DIR / "benchmark_summary.csv"

# Run all sizes on the same machine in one sitting so their latencies are comparable.
CORPUS_SIZES = [2000, 10000, 50000]
K = 5
N_REPEATS = 10
WARMUP_SEARCHES = 3
MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_DIMENSION = 384
MAX_50000_EMBEDDING_SECONDS = 30 * 60
MIB = 1024 * 1024

# Keep HNSW fixed so corpus size is the only HNSW setting swept by this run.
HNSW_PARAMETERS = {"M": 32, "efConstruction": 200, "efSearch": 128}
# Keep PQ fixed to compare the same 48-way, 4-bit compression at every size.
# The 384-dimensional embeddings divide evenly by m=48, and the smallest
# default corpus has enough rows for the 16-centroid PQ codebooks.
PQ_PARAMETERS = {"m": 48, "nbits": 4}


class SearchableIndex(Protocol):
    def search(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...


class FlatIndexAdapter:
    """Procedural FAISS Flat index behind the shared index interface."""

    def __init__(self, dimension: int) -> None:
        self.index = faiss.IndexFlatL2(dimension)

    def build(self, vectors: np.ndarray) -> None:
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.index.search(np.ascontiguousarray(query_vectors, dtype=np.float32), k)


@dataclass
class IndexRun:
    index_type: str
    index: SearchableIndex
    build_time_s: float
    memory_mb: float
    parameters: dict[str, Any]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=CORPUS_SIZES,
        help="Corpus prefix sizes (default: 2000 10000 50000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_PATH,
        help="Per-query CSV output path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=SUMMARY_PATH,
        help="Per-index summary CSV output path.",
    )
    return parser.parse_args()


def _cached_train_shards() -> list[Path]:
    """Find the already-cached MS MARCO v2.1 Arrow shards without downloading."""
    from datasets import config

    cache_root = Path(config.HF_DATASETS_CACHE) / "ms_marco" / "v2.1" / "0.0.0"
    shards = sorted(cache_root.glob("*/ms_marco-train-*.arrow"))
    return shards


def load_ordered_passages(count: int) -> pd.DataFrame:
    """Return the first ``count`` nonempty passages in train dataset order.

    ``source_id`` is the original row position in the unsliced train split.
    Returned/index IDs are always positions in this filtered, reset-index frame.
    The cached Arrow stream reader avoids making Hugging Face cache lock files
    outside the project workspace.
    """
    if count < 1:
        raise ValueError("count must be positive")

    shards = _cached_train_shards()
    if not shards:
        raise FileNotFoundError(
            "Cached MS MARCO v2.1 train Arrow shards were not found. "
            "Populate the Hugging Face datasets cache before benchmarking."
        )

    import pyarrow.ipc as ipc

    rows: list[dict[str, Any]] = []
    source_id = 0
    for shard_path in shards:
        with shard_path.open("rb") as shard_file:
            reader = ipc.open_stream(shard_file)
            for batch in reader:
                for row in batch.to_pylist():
                    passage_texts = (row.get("passages") or {}).get("passage_text") or []
                    if passage_texts:
                        rows.append({"source_id": source_id, "text": str(passage_texts[0])})
                        if len(rows) == count:
                            break
                    source_id += 1
                if len(rows) == count:
                    break
        if len(rows) == count:
            break

    if len(rows) < count:
        raise ValueError(
            f"Requested {count} valid passages, but the cached train split supplied {len(rows)}."
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def verify_2000_passages(passages: pd.DataFrame) -> None:
    """Check that the first 2,000 valid rows exactly match the legacy CSV."""
    if len(passages) != 2000:
        raise ValueError("The legacy 2,000-row check requires exactly 2,000 passages")
    if not PASSAGES_PATH.exists():
        raise FileNotFoundError(f"Required legacy corpus file is missing: {PASSAGES_PATH}")
    legacy = pd.read_csv(PASSAGES_PATH)
    if list(legacy.columns) != ["id", "text"]:
        raise ValueError("data/passages.csv must retain its existing id,text columns")
    ids_match = np.array_equal(legacy["id"].to_numpy(), passages["source_id"].to_numpy())
    texts_match = legacy["text"].astype(str).tolist() == passages["text"].astype(str).tolist()
    if len(legacy) != 2000 or not ids_match or not texts_match:
        raise ValueError(
            "The first 2,000 valid cached passages do not exactly reproduce data/passages.csv"
        )
    print("Verified: first 2,000 valid passages exactly match data/passages.csv.")


def recall_at_k(actual_ids: list[int], ground_truth_ids: list[int]) -> Optional[float]:
    """Recall against Flat, using Flat's valid returned count as the denominator."""
    if not ground_truth_ids:
        return None
    return len(set(actual_ids).intersection(ground_truth_ids)) / len(ground_truth_ids)


def valid_ids(ids: np.ndarray, corpus_size: int) -> list[int]:
    """Remove FAISS missing-result markers and keep positional corpus IDs only."""
    return [int(vector_id) for vector_id in ids if 0 <= int(vector_id) < corpus_size]


def _parameters(index_type: str, corpus_size: int) -> dict[str, Any]:
    if index_type == "flat":
        return {"metric": "L2"}
    if index_type == "hnsw":
        return {"metric": "L2", **HNSW_PARAMETERS}
    if index_type == "ivf":
        nlist = min(round(4 * math.sqrt(corpus_size)), corpus_size // 39)
        nprobe = max(10, round(nlist / 10)) if nlist else 0
        return {
            "metric": "L2",
            "nlist": nlist,
            "nprobe": nprobe,
            "training_points": corpus_size,
        }
    if index_type == "pq":
        return {"metric": "L2", **PQ_PARAMETERS, "training_points": corpus_size}
    raise ValueError(f"Unknown index type: {index_type}")


def _build_index(index_type: str, vectors: np.ndarray, parameters: dict[str, Any]) -> SearchableIndex:
    dimension = vectors.shape[1]
    if index_type == "flat":
        index = FlatIndexAdapter(dimension)
        index.build(vectors)
        return index
    if index_type == "hnsw":
        index = HNSWIndex(dimension, **HNSW_PARAMETERS)
        index.build(vectors)
        return index
    if index_type == "ivf":
        if parameters["nlist"] < 1:
            raise ValueError("IVF needs at least 39 corpus points to train one centroid")
        index = IVFIndex(nlist=parameters["nlist"], nprobe=parameters["nprobe"])
        index.build(vectors)
        return index
    if index_type == "pq":
        if corpus_size_for_pq(vectors) < (1 << PQ_PARAMETERS["nbits"]):
            raise ValueError("PQ has fewer training vectors than codebook centroids")
        index = PQIndex(dimension, **PQ_PARAMETERS)
        index.build(vectors)
        return index
    raise ValueError(f"Unknown index type: {index_type}")


def corpus_size_for_pq(vectors: np.ndarray) -> int:
    return int(vectors.shape[0])


def _faiss_index(index: SearchableIndex) -> faiss.Index:
    if isinstance(index, FlatIndexAdapter):
        return index.index
    if isinstance(index, IVFIndex):
        if index._index is None:
            raise RuntimeError("IVF index is not built")
        return index._index
    if isinstance(index, (HNSWIndex, PQIndex)):
        return index.index
    raise TypeError(f"Cannot serialize {type(index).__name__}")


def _serialized_size_mb(index: SearchableIndex) -> float:
    # memory_mb is FAISS serialized index size (the disk-size measure), not process RAM.
    serialized = faiss.serialize_index(_faiss_index(index))
    return len(serialized) / MIB


def build_and_measure(
    index_type: str, vectors: np.ndarray, parameters: dict[str, Any]
) -> IndexRun:
    started = time.perf_counter()
    index = _build_index(index_type, vectors, parameters)
    build_time_s = time.perf_counter() - started
    memory_mb = _serialized_size_mb(index)
    return IndexRun(index_type, index, build_time_s, memory_mb, parameters)


def _result_payload(passages: pd.DataFrame, ids: list[int]) -> tuple[str, str, str]:
    # IDs index rows positionally; source_id is retained separately for traceability.
    selected = passages.iloc[ids]
    source_ids = selected["source_id"].astype(int).tolist()
    texts = selected["text"].astype(str).tolist()
    return json.dumps(ids), json.dumps(source_ids), json.dumps(texts)


def _failed_result_row(
    corpus_size: int, index_type: str, query_id: int, query: str, error: Exception
) -> dict[str, Any]:
    return {
        "corpus_size": corpus_size,
        "index_type": index_type,
        "query_id": query_id,
        "query_text": query,
        "k": K,
        "latency_ms": "",
        "recall_at_k": "",
        "returned_ids": "[]",
        "returned_source_ids": "[]",
        "returned_texts": "[]",
        "status": "failed",
        "error_message": f"{type(error).__name__}: {error}",
    }


def benchmark_index(
    run: IndexRun,
    query_embeddings: np.ndarray,
    queries: list[str],
    passages: pd.DataFrame,
    flat_result_ids: list[list[int]],
) -> tuple[list[dict[str, Any]], list[float], bool]:
    """Warm each index, then time every query N times and retain per-query medians."""
    rows: list[dict[str, Any]] = []
    all_timings: list[float] = []
    failed = False

    try:
        for warmup_id in range(WARMUP_SEARCHES):
            query_vector = query_embeddings[warmup_id % len(query_embeddings) : warmup_id % len(query_embeddings) + 1]
            run.index.search(query_vector, K)
    except Exception as error:
        print(f"WARNING: {run.index_type} warm-up failed: {type(error).__name__}: {error}")
        return (
            [
                _failed_result_row(len(passages), run.index_type, i, query, error)
                for i, query in enumerate(queries)
            ],
            all_timings,
            True,
        )

    for query_id, query in enumerate(queries):
        query_timings: list[float] = []
        ids: list[int] = []
        try:
            query_vector = query_embeddings[query_id : query_id + 1]
            for _ in range(N_REPEATS):
                started = time.perf_counter_ns()
                _, result_ids = run.index.search(query_vector, K)
                query_timings.append((time.perf_counter_ns() - started) / 1_000_000)
                ids = valid_ids(result_ids[0], len(passages))
            all_timings.extend(query_timings)
            if run.index_type == "flat":
                recall = 1.0
            else:
                recall = recall_at_k(ids, flat_result_ids[query_id])
            ids_json, source_ids_json, texts_json = _result_payload(passages, ids)
            rows.append(
                {
                    "corpus_size": len(passages),
                    "index_type": run.index_type,
                    "query_id": query_id,
                    "query_text": query,
                    "k": K,
                    "latency_ms": float(np.median(query_timings)),
                    "recall_at_k": recall,
                    "returned_ids": ids_json,
                    "returned_source_ids": source_ids_json,
                    "returned_texts": texts_json,
                    "status": "success",
                    "error_message": "",
                }
            )
        except Exception as error:
            failed = True
            print(f"WARNING: {run.index_type} failed for query {query_id}: {type(error).__name__}: {error}")
            rows.append(_failed_result_row(len(passages), run.index_type, query_id, query, error))

    return rows, all_timings, failed


def _embedding_model() -> SentenceTransformer:
    # The model is already cached locally; offline mode prevents fallback to random weights.
    return SentenceTransformer(MODEL_NAME, local_files_only=True)


def _encode(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.shape != (len(texts), MODEL_DIMENSION):
        raise ValueError(
            f"Expected embeddings with shape ({len(texts)}, {MODEL_DIMENSION}), got {vectors.shape}"
        )
    return vectors


def _load_or_encode(
    size: int,
    passages: pd.DataFrame,
    model: SentenceTransformer,
    legacy_embeddings: Optional[np.ndarray],
) -> np.ndarray:
    if size == 2000 and legacy_embeddings is not None and legacy_embeddings.shape == (size, MODEL_DIMENSION):
        print("Using existing data/embeddings.npy for corpus size 2000.")
        return np.ascontiguousarray(legacy_embeddings, dtype=np.float32)

    cache_path = DATA_DIR / f"embeddings_{size}.npy"
    if cache_path.exists():
        vectors = np.load(cache_path).astype(np.float32, copy=False)
        if vectors.shape != (size, MODEL_DIMENSION):
            raise ValueError(f"Embedding cache has wrong shape: {cache_path} {vectors.shape}")
        print(f"Using cached embeddings: {cache_path}")
        return np.ascontiguousarray(vectors)

    vectors = _encode(model, passages["text"].tolist())
    np.save(cache_path, vectors)
    print(f"Cached embeddings: {cache_path}")
    return vectors


def _write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    output_path = path if path.is_absolute() else PROJECT_DIR / path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


RESULT_COLUMNS = [
    "corpus_size",
    "index_type",
    "query_id",
    "query_text",
    "k",
    "latency_ms",
    "recall_at_k",
    "returned_ids",
    "returned_source_ids",
    "returned_texts",
    "status",
    "error_message",
]
SUMMARY_COLUMNS = [
    "corpus_size",
    "index_type",
    "k",
    "n_repeats",
    "warmup_searches",
    "avg_recall_at_k",
    "p50_latency_ms",
    "p95_latency_ms",
    "build_time_s",
    "memory_mb",
    "status",
    "parameters",
]


def _summary_row(
    corpus_size: int,
    index_type: str,
    parameters: dict[str, Any],
    result_rows: list[dict[str, Any]],
    raw_timings: list[float],
    run: Optional[IndexRun],
    failed: bool,
) -> dict[str, Any]:
    recalls = [
        float(row["recall_at_k"])
        for row in result_rows
        if row["status"] == "success" and row["recall_at_k"] != ""
    ]
    return {
        "corpus_size": corpus_size,
        "index_type": index_type,
        "k": K,
        "n_repeats": N_REPEATS,
        "warmup_searches": WARMUP_SEARCHES,
        "avg_recall_at_k": float(np.mean(recalls)) if recalls else "",
        "p50_latency_ms": float(np.percentile(raw_timings, 50)) if raw_timings else "",
        "p95_latency_ms": float(np.percentile(raw_timings, 95)) if raw_timings else "",
        "build_time_s": "" if run is None else run.build_time_s,
        "memory_mb": "" if run is None else run.memory_mb,
        "status": "failed" if failed else "success",
        "parameters": json.dumps(parameters, sort_keys=True),
    }


def _print_size_summary(corpus_size: int, rows: list[dict[str, Any]]) -> None:
    print(f"\nCorpus size {corpus_size} summary (K={K}; p50/p95 use all query × repeat timings):")
    print(f"{'Index':<8} {'Avg recall':>12} {'p50 ms':>12} {'p95 ms':>12} {'Build s':>12} {'Size MB':>12} {'Status':>10}")
    for row in rows:
        values = [row[key] for key in ("avg_recall_at_k", "p50_latency_ms", "p95_latency_ms", "build_time_s", "memory_mb")]
        rendered = ["n/a" if value == "" else f"{float(value):.3f}" for value in values]
        print(
            f"{row['index_type']:<8} {rendered[0]:>12} {rendered[1]:>12} {rendered[2]:>12} "
            f"{rendered[3]:>12} {rendered[4]:>12} {row['status']:>10}"
        )


def main() -> None:
    args = parse_arguments()
    requested_sizes = sorted(set(args.sizes))
    if not requested_sizes or any(size < 1 for size in requested_sizes):
        raise ValueError("--sizes must contain positive integers")

    largest_needed = max(max(requested_sizes), 500)
    ordered_passages = load_ordered_passages(largest_needed)
    if 2000 in requested_sizes:
        verify_2000_passages(ordered_passages.iloc[:2000].reset_index(drop=True))

    model = _embedding_model()
    pilot_texts = ordered_passages.iloc[:500]["text"].tolist()
    pilot_started = time.perf_counter()
    _encode(model, pilot_texts)
    pilot_seconds = time.perf_counter() - pilot_started
    projected_10000 = pilot_seconds * (10000 / 500)
    projected_50000 = pilot_seconds * (50000 / 500)
    print(
        f"500-passage embedding pilot: {pilot_seconds:.2f} s "
        f"({pilot_seconds / 500:.4f} s/passage); projections: "
        f"10,000={projected_10000 / 60:.1f} min, 50,000={projected_50000 / 60:.1f} min."
    )

    run_sizes = [size for size in requested_sizes if size != 50000]
    if 50000 in requested_sizes:
        cache_50000 = DATA_DIR / "embeddings_50000.npy"
        if cache_50000.exists() or projected_50000 < MAX_50000_EMBEDDING_SECONDS:
            run_sizes.append(50000)
        else:
            print(
                "WARNING: projected 50,000-passage embedding exceeds 30 minutes; "
                "stopping after the requested smaller corpus sizes."
            )

    legacy_embeddings = None
    if EMBEDDINGS_PATH.exists():
        legacy_embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    all_result_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []

    query_embeddings = _encode(model, TEST_QUERIES)
    for size in run_sizes:
        passages = ordered_passages.iloc[:size].reset_index(drop=True)
        vectors = _load_or_encode(size, passages, model, legacy_embeddings)
        if vectors.shape[0] != len(passages):
            raise ValueError(f"Passage/embedding row mismatch at corpus size {size}")

        flat_result_ids: list[list[int]] = [[] for _ in TEST_QUERIES]
        size_rows: list[dict[str, Any]] = []
        for index_type in ("flat", "hnsw", "ivf", "pq"):
            parameters = _parameters(index_type, size)
            print(f"Building {index_type} for corpus size {size}: {parameters}")
            run: Optional[IndexRun] = None
            try:
                run = build_and_measure(index_type, vectors, parameters)
            except Exception as error:
                print(f"WARNING: could not build {index_type} at size {size}: {type(error).__name__}: {error}")
                failed_rows = [
                    _failed_result_row(size, index_type, i, query, error)
                    for i, query in enumerate(TEST_QUERIES)
                ]
                all_result_rows.extend(failed_rows)
                summary = _summary_row(size, index_type, parameters, failed_rows, [], None, True)
                all_summary_rows.append(summary)
                size_rows.append(summary)
                continue

            rows, raw_timings, query_failed = benchmark_index(
                run, query_embeddings, TEST_QUERIES, passages, flat_result_ids
            )
            all_result_rows.extend(rows)
            if index_type == "flat":
                flat_result_ids = [
                    json.loads(row["returned_ids"]) if row["status"] == "success" else []
                    for row in rows
                ]
            summary = _summary_row(
                size, index_type, parameters, rows, raw_timings, run, query_failed
            )
            all_summary_rows.append(summary)
            size_rows.append(summary)

        _print_size_summary(size, size_rows)

    _write_csv(all_result_rows, args.output, RESULT_COLUMNS)
    _write_csv(all_summary_rows, args.summary_output, SUMMARY_COLUMNS)
    print(f"\nWrote {len(all_result_rows)} per-query rows to {args.output}.")
    print(f"Wrote {len(all_summary_rows)} summary rows to {args.summary_output}.")


if __name__ == "__main__":
    main()
