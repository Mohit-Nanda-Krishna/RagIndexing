# Project context: RagIndexing

Current handoff for the DBMS vector-index benchmarking project. This document describes the repository as it exists now; the earlier planning handoff remains available in Git history. Start with [README.md](README.md) for setup and run commands.

## Goal and scope

Measure the retrieval trade-offs of four FAISS index strategies over a shared text corpus and embedding model. The project stops at retrieval: it does not generate RAG answers. PostgreSQL/pgvector was an earlier idea but has not been implemented; `psycopg2-binary` in `requirements.txt` does not imply a working database integration.

## Current state

| Area | Implementation |
| --- | --- |
| Source corpus | MS MARCO v2.1 train passages; the first passage text from each row with a nonempty passage list |
| Embeddings | `all-MiniLM-L6-v2`, 384 dimensions, float32, searched with L2 distance |
| Indexes | Flat exact baseline, HNSW, IVF, and PQ in FAISS |
| Benchmark | `benchmark.py` measures recall, search latency, build time, and serialized index size |
| Presentation | `dashboard.py` reads committed CSVs and shows five Streamlit tabs |
| Recommendation | `selector.py` filters indexes by constraints and scores survivors with preset weights |
| Tests | `tests/` covers indexes and benchmark utilities; `test_selector.py` covers selector rules |

The committed snapshot has corpus sizes **2,000 / 10,000 / 50,000**, four index summaries per size (12 rows in `benchmark_summary.csv`), and 20 query results per index and size (240 rows in `benchmark_results.csv`). These are results from a particular benchmark run, not constants in the dashboard or selector. Both components read available sizes and measurements from the CSVs.

## Data flow and contracts

```text
Cached MS MARCO train Arrow shards
    -> ordered, nonempty passages
    -> shared sentence-transformer embeddings
    -> Flat / HNSW / IVF / PQ indexes
    -> benchmark_summary.csv + benchmark_results.csv
    -> dashboard.py and selector.py
```

- `data/passages.csv` is the earlier 2,000-passage corpus with `id,text` columns. When size 2,000 is benchmarked, `benchmark.py` checks that its first 2,000 valid cached passages exactly match this file.
- Returned index IDs are **positions in the filtered corpus**, from `0` to `corpus_size - 1`. `source_id` is the original MS MARCO row position. Do not substitute one for the other.
- In `benchmark_results.csv`, `returned_ids`, `returned_source_ids`, and `returned_texts` are JSON arrays encoded inside CSV cells. Rows also contain query ID/text, latency, recall, status, and error message.
- `benchmark_summary.csv` has one row per corpus size and index type. Its `parameters` cell is JSON. `memory_mb` is the FAISS **serialized index size in MiB**, not RAM use; the dashboard labels it “Index size (MB).”
- `status` can be `success` or `failed`. The dashboard and selector ignore unsuccessful rows; the dashboard warns about them.

## Benchmark methodology

The current benchmark uses `k=5`, 20 fixed questions from `test_queries.py`, 10 timed searches per question, and three warm-up searches for each index. Every index receives the same corpus embeddings and query embeddings. Flat's exact top-k IDs are the ground truth for recall. Per-query `latency_ms` is the median of that query's repeats; summary p50 and p95 are calculated across **all** query-by-repeat search timings. Build time excludes embedding generation. `memory_mb` is measured from a serialized FAISS index.

Settings in `benchmark.py`:

| Index | Settings |
| --- | --- |
| Flat | Exact `IndexFlatL2` |
| HNSW | `M=32`, `efConstruction=200`, `efSearch=128` |
| IVF | `nlist` and `nprobe` derived from corpus size; actual values saved per run |
| PQ | `m=48`, `nbits=4` at every size |

These are selected configurations, not an exhaustive parameter search. In particular, PQ's recall reflects its fixed 4-bit compression setting.

## Selector contract

`select_index(summary_df, corpus_size, preset="balanced", min_recall=None, max_latency_ms=None, max_memory_mb=None, latency_metric="p95")` is a pure function. It uses the nearest successfully benchmarked corpus size if an exact size is missing. It rejects indexes that violate limits, records each violation, min-max normalizes latency/recall/index size among survivors, and returns a winner, a per-index table, weights, size used, and an explanation. When nothing survives, it returns no winner and identifies the closest miss.

Presets in `selector.py` are `balanced` (equal thirds), `fastest` (70% latency), `accurate` (70% recall), and `smallest` (70% index size). Build time is measured and displayed but is not part of selector scoring. `recommend_for_all_sizes` applies one preset to every successful benchmark size.

## Running and editing

- Use the commands in [README.md](README.md) for environment setup, dashboard, selector demo, and tests.
- The dashboard needs only the two committed benchmark CSVs. It does not need `data/embeddings.npy`, the Hugging Face cache, or saved FAISS indexes.
- Re-running `benchmark.py` is a separate, resource-intensive task. It requires cached MS MARCO v2.1 train Arrow shards and a locally cached `all-MiniLM-L6-v2` model. The model is loaded with `local_files_only=True`; a fresh clone alone cannot regenerate the CSVs. Use `--output` and `--summary-output` to avoid overwriting committed results while experimenting.
- `run_pipeline.py` is the original 2,000-passage Flat prototype. It downloads data and writes `data/passages.csv` and `indexes/flat.index`; it is not needed for viewing the completed benchmark.
- For team changes, branch from current `main`, keep changes scoped, run relevant tests, and review CSV changes before committing them.

## Interpretation limits

All latency numbers are machine-specific and should be compared as ratios or trends. The experiment has 20 fixed queries, three corpus sizes, one embedding model, one distance metric, and one chosen configuration for HNSW and PQ at each size. The measured “index size” is disk serialization, not live process memory. This project provides retrieval/indexing evidence, not end-to-end answer quality or a PostgreSQL comparison.
