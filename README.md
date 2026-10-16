# RagIndexing

A DBMS course project that compares four FAISS vector indexes for the retrieval step of a RAG system. It measures how search speed, recall, build time, and index size change as the passage corpus grows. A Streamlit dashboard presents the results and a rule-based selector recommends an index for a given set of constraints.

This project benchmarks **retrieval**. It does not generate answers with an LLM, and it does not currently run a PostgreSQL/pgvector index.

## Run the dashboard

From the repository root in Windows PowerShell, with Python 3.11 installed:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m streamlit run dashboard.py
```

If you already have a working `venv`, skip its creation and the install step. Open the local URL printed by Streamlit. The committed `benchmark_summary.csv` and `benchmark_results.csv` are enough to use the dashboard; no model download or benchmark rerun is needed.

The dashboard has five views:

| View | What it shows |
| --- | --- |
| Overview | Winners, a comparison table, and speed versus recall for the selected corpus size |
| Scaling | Latency, recall, build time, and index size across corpus sizes |
| Query Explorer | Per-query measurements and top-k passage overlap with exact Flat results |
| How it works | Index definitions, actual parameters, and methodology limits |
| Index Selector | A recommendation based on preset weights and optional recall, latency, and size limits |

Use **Reload data** in the sidebar after replacing either benchmark CSV. The app reads the CSVs and does not build indexes.

## What was measured

```text
MS MARCO passages → all-MiniLM-L6-v2 embeddings → Flat / HNSW / IVF / PQ
                                                → benchmark CSVs → dashboard + selector
```

The committed benchmark covers **2,000, 10,000, and 50,000 passages**, with **20 fixed queries** and **k = 5**. All four indexes use the same 384-dimensional embeddings and L2 distance. Each query is searched 10 times after three warm-up searches.

| Measure | Meaning |
| --- | --- |
| Recall@5 | Share of Flat's exact top-5 passage IDs also returned by an index |
| p50 / p95 latency | Median / 95th percentile of the measured search timings, in milliseconds |
| Build time | Time spent constructing the index, in seconds |
| Index size | Size of the serialized FAISS index, in MiB; this is **not** process RAM usage |

Flat is the exact baseline. HNSW is a graph index, IVF searches selected clusters, and PQ compresses vectors. The benchmark parameters for each run are stored in the `parameters` column of `benchmark_summary.csv` and shown in **How it works**.

The selector scores only successful benchmark rows. It first filters out indexes that violate constraints, then normalizes latency, recall, and index size among the survivors and applies the chosen preset weights. Run its command-line examples with:

```powershell
.\venv\Scripts\python.exe selector.py
```

## Tests

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe -m unittest test_selector -v
```

The first command covers the benchmark and index implementations; the second covers selector rules.

## Files and reproducibility

| Path | Purpose |
| --- | --- |
| `benchmark.py` | Builds and times each index; writes the two benchmark CSVs |
| `benchmark_summary.csv` | One summary row per corpus size and index |
| `benchmark_results.csv` | One row per corpus size, index, and query, including returned passages |
| `dashboard.py` | Read-only Streamlit presentation |
| `selector.py` | Pure recommendation functions and a small CLI demo |
| `hnsw_index.py`, `ivf_index.py`, `pq_index.py` | FAISS index implementations |
| `data/passages.csv` | Legacy 2,000-passage corpus used for a consistency check |
| `project_context.md` | Current architecture, data contract, and handoff notes |

Reproducing the measurements is an optional, slower workflow. `benchmark.py` expects the MS MARCO v2.1 train Arrow shards **and** the `all-MiniLM-L6-v2` model to be present in local caches. Neither cache is included in the repository. Embedding arrays (`*.npy`) are intentionally gitignored. Once those prerequisites are available, write regenerated measurements to separate files so the committed results stay intact:

```powershell
.\venv\Scripts\python.exe benchmark.py --output regenerated_results.csv --summary-output regenerated_summary.csv
```

See `project_context.md` for the benchmark's data contract and limitations. Absolute timings depend on the machine; compare ratios and scaling trends rather than treating a millisecond value as universal.
