# PROJECT HANDOFF DOCUMENT
## RagIndexing — Vector Index Benchmarking Project

Generated from a prior Claude conversation to hand off context to independent developers/Claude instances. Every claim below is labeled:

- **[FACT]** — directly established in the source conversation (code shown, command run, decision explicitly made)
- **[ASSUMPTION]** — reasonable inference, not explicitly confirmed
- **[UNKNOWN]** — not covered in the source conversation at all
- **[DECISION NEEDED]** — the team must decide this before parallel work can safely proceed

---

## 1. PROJECT OVERVIEW

**What we are building** [FACT]
A DBMS-course team project that benchmarks different vector index strategies used in Retrieval-Augmented Generation (RAG) systems. The project does **not** build a full RAG pipeline with an LLM generation step — it focuses specifically on the retrieval/indexing layer.

**Main objective** [FACT]
Given a fixed text corpus and a fixed embedding model, measure how different Approximate Nearest Neighbor (ANN) indexing strategies — Flat (exact), IVF, HNSW, PQ — trade off:
- Recall@k (retrieval accuracy)
- Query latency (p50 / p95)
- Memory / storage footprint
- Index build time

as corpus size scales, and produce a practical decision framework ("given corpus size + latency budget + memory constraint, use index X").

**What problem it solves** [FACT]
Most RAG implementations pick a vector index by convention/tutorial rather than measurement. This project replaces that guesswork with empirical, reproducible benchmarks.

**Current overall architecture** [FACT — as planned, only partially implemented]
Five conceptual layers:
1. Data ingestion (corpus loading/cleaning)
2. Embedding layer (text → dense vectors)
3. Index/storage layer (Flat, IVF, HNSW, PQ in FAISS; pgvector in PostgreSQL as a DBMS comparison point)
4. Benchmark engine (recall/latency/memory measurement across indexes)
5. Presentation layer (Streamlit dashboard)

**How the indexing system fits into the RAG pipeline** [FACT]
This project only covers steps up through "retrieval." There is **no generation/LLM step** implemented or currently planned in this project. See Section 7 for the exact pipeline scope.

**Important — do not confuse with a separate project** [FACT]
The same team/individual has a **different, separate** solo project called "adaptive RAG retrieval" (query routing across retrieval methods, hallucination-rate evaluation, intended for possible publication). That is a **different codebase and different scope**. This handoff document is ONLY about the DBMS vector-index-benchmarking project (repo: RagIndexing). Do not merge concepts from the other project in unless explicitly told to.

---

## 2. CURRENT PROJECT STATUS

### COMPLETED [FACT]
- Repository created and shared: `https://github.com/Mohit-Nanda-Krishna/RagIndexing.git`
- `requirements.txt` created and pinned (see Section 10)
- `.gitignore` created
- Data ingestion: downloads a ~2,000-passage subset of the MS MARCO v2.1 dataset
- Data cleaning: extracts passage text into a `pandas` DataFrame, saved to `data/passages.csv`
- Embedding generation: all passages embedded using `sentence-transformers` model `all-MiniLM-L6-v2`, saved to `data/embeddings.npy`
- **Flat (exact) index implemented**: FAISS `IndexFlatL2`, built and saved to `indexes/flat.index`
- One end-to-end test query proven to work (`"What causes inflation?"` → top-5 results printed)
- Team environment issues resolved (see Known Bugs)
- A standalone embedding-visualization script was written (PCA scatter plot) — **[UNKNOWN whether it was actually added to the repo by the user; it was only handed to them as a file]**

### PARTIALLY COMPLETED
- None currently — the pipeline is either done (Flat, steps 1–4 of ingestion) or not started (everything else)

### NOT STARTED [FACT]
- HNSW index implementation
- IVF index implementation
- PQ index implementation
- pgvector / PostgreSQL integration (only planned conceptually, zero code written)
- Benchmark harness (recall@k / latency p50-p95 / memory / build-time measurement across index types)
- Multi-scale corpus testing (10K / 50K / 200K docs — currently only ~2,000 passages used)
- Streamlit dashboard
- Any automated tests
- A common index interface/class abstraction (see Section 5 — this does not exist yet)

### KNOWN BUGS / ISSUES [FACT]
1. `faiss-cpu` was originally pinned to `==1.8.0` in `requirements.txt`, which does not exist on PyPI for most current Python versions → caused install failures. **Fixed**: changed to `faiss-cpu>=1.9.0`.
2. One teammate's machine had **Python 3.14** installed, which has no pre-built `faiss-cpu` wheels, causing installation to fail (would require building from source, which fails without a C++ toolchain on Windows). **Resolved** by creating the venv with an already-installed Python 3.12 instead (`py -3.12 -m venv venv`).
3. The `datasets` library (used to load MS MARCO) and `scikit-learn` (used only in the optional visualization script) were installed via ad-hoc `pip install` commands but were **never added to `requirements.txt`**. This is a reproducibility gap — a fresh clone + `pip install -r requirements.txt` will NOT have these packages.
4. `run_pipeline.py` has **no caching/idempotency check** — every run re-downloads the dataset and regenerates embeddings from scratch, even if `data/embeddings.npy` already exists locally.
5. `run_pipeline.py` has **no error handling** (no try/except blocks) — [ASSUMPTION based on the code as written].
6. The 4-person team has, up to this point, worked directly on the `main` branch with no feature branches (an explicit team decision, made when they were working in-person together). This will likely cause conflicts once 3 developers work independently/asynchronously on HNSW/IVF/PQ — flagged in Section 13.

### TODO
See Section 15 for the full prioritized list.

---

## 3. REPOSITORY / FILE STRUCTURE

Current structure, as directly observed (via a screenshot of one teammate's clone) [FACT]:

```
RagIndexing/
├── data/
│   └── passages.csv          [FACT — committed to repo]
├── indexes/
│   └── flat.index            [FACT — committed to repo]
├── venv/                     [FACT — local only, gitignored, never committed]
├── .gitignore                [FACT]
├── requirements.txt          [FACT]
└── run_pipeline.py           [FACT]
```

**Not committed / not present in a fresh clone** [FACT, due to `.gitignore`]:
- `data/embeddings.npy` — excluded via `*.npy` in `.gitignore`; each developer must regenerate locally by running `run_pipeline.py`
- `venv/` — excluded

**Files referenced in conversation but NOT confirmed to exist in the repo:**
- `visualize_embeddings.py` — [UNKNOWN — handed to the user as a standalone file; not confirmed added/committed]
- `data/embeddings_plot.png` — would be generated by the above script if run; [UNKNOWN if it exists]

**File-by-file breakdown:**

#### `run_pipeline.py` [FACT — full contents known, see Section 4]
- Purpose: single linear script that does data download → clean → embed → build Flat index → run one test query
- No classes, no functions except the implicit script flow — everything is top-level procedural code
- Interacts with: reads/writes `data/passages.csv`, `data/embeddings.npy`, `indexes/flat.index`

#### `requirements.txt` [FACT — full contents known, see Section 10]
- Pinned/floor-pinned dependency versions for FAISS, sentence-transformers, pandas, numpy, psycopg2-binary, streamlit, matplotlib
- **Missing**: `datasets`, `scikit-learn` (both used ad-hoc, not pinned — see Known Bugs #3)

#### `.gitignore` [FACT — full contents known]
```
venv/
__pycache__/
*.pyc
.env
*.npy
.DS_Store
```

#### `data/passages.csv` [FACT]
- Generated output, not hand-written
- Columns: `id` (int, sequential from `enumerate()`), `text` (string, passage content)
- ~2,000 rows (subset of MS MARCO v2.1 `train` split)

#### `indexes/flat.index` [FACT]
- Generated output, FAISS binary index file, written via `faiss.write_index()`
- Contains a `faiss.IndexFlatL2` built over the embeddings

#### `data/embeddings.npy` [FACT — generated, gitignored, not in fresh clones]
- NumPy array, shape `(num_passages, 384)`, dtype [ASSUMPTION: float32, the default `sentence-transformers` output dtype — not explicitly verified in conversation]

---

## 4. FLAT INDEX IMPLEMENTATION (REFERENCE)

This is the **only** index implemented so far. Exact code as written [FACT]:

```python
import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

os.makedirs("data", exist_ok=True)
os.makedirs("indexes", exist_ok=True)

print("Step 1: Downloading a small dataset...")
ds = load_dataset("ms_marco", "v2.1", split="train[:2000]")

print("Step 2: Cleaning it into a table...")
passages = []
for i, row in enumerate(ds):
    if row["passages"]["passage_text"]:
        passages.append({"id": i, "text": row["passages"]["passage_text"][0]})
df = pd.DataFrame(passages)
df.to_csv("data/passages.csv", index=False)
print(f"Saved {len(df)} passages to data/passages.csv")

print("Step 3: Generating embeddings (this takes a few minutes)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)
np.save("data/embeddings.npy", embeddings)
print("Saved embeddings to data/embeddings.npy")

print("Step 4: Building the search index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "indexes/flat.index")
print("Saved index to indexes/flat.index")

print("Step 5: Testing with a real search query...")
query = "What causes inflation?"
q_embedding = model.encode([query])
distances, indices = index.search(q_embedding, k=5)

print("\nTop 5 results for query:", query)
for idx in indices[0]:
    print("-", df.iloc[idx]['text'][:150])

print("\nDONE. Pipeline works end to end.")
```

**Details:**
- **Class name**: None — uses `faiss.IndexFlatL2` directly, no custom wrapper class exists [FACT]
- **Constructor/API**: `faiss.IndexFlatL2(dim)` where `dim = embeddings.shape[1]` (384) [FACT]
- **Add/build process**: `index.add(embeddings)` — bulk add of the entire NumPy array at once [FACT]
- **Search process**: `index.search(query_vector, k)` → returns `(distances, indices)` as NumPy arrays [FACT]
- **Distance/similarity metric**: L2 (Euclidean distance), via `IndexFlatL2` [FACT]. **[DECISION NEEDED]**: whether cosine similarity should be used instead (common for sentence-transformer embeddings) — not discussed in the conversation.
- **Vector format**: dense NumPy float array, shape `(n, 384)` [FACT for shape; ASSUMPTION for exact dtype]
- **ID handling**: implicit — FAISS uses the row's position in the `add()` call as its internal ID; this is matched back to `df.iloc[idx]` by positional index, NOT by the `id` column in the CSV explicitly [FACT, based on code — `df.iloc[idx]` is positional, and the CSV's `id` column happens to equal position, but this is fragile]. **[DECISION NEEDED]**: use FAISS's `IndexIDMap` or similar to bind explicit IDs, rather than relying on positional alignment, before more indexes are built independently.
- **Metadata handling**: none beyond the `text` column — no other metadata is attached to vectors [FACT]
- **Return format**: raw FAISS `(distances, indices)` tuple; the script then manually looks up text via `df.iloc[idx]['text']` — there is no structured result object (e.g., no `{id, text, score}` dict/class) [FACT]
- **Dependencies**: `faiss`, `numpy`, `pandas`, `sentence_transformers`, `datasets` (the last one not yet in `requirements.txt`)
- **Edge cases**: NOT handled — no check for empty corpus, no check for embedding/index dimension mismatch, no error handling at all [ASSUMPTION, based on absence of any try/except in the shown code]
- **Tests**: none exist [FACT]
- **Important implementation decisions already made**:
  - Corpus source is MS MARCO v2.1, `train[:2000]` slice specifically (small, deliberately kept minimal for a first working pipeline) [FACT]
  - Embedding model fixed as `all-MiniLM-L6-v2` (384-dim) [FACT] — **[DECISION NEEDED]**: is this locked in for the whole project, or was it just for the initial pipeline test? The conversation treats it as the standing choice but never explicitly says "this is final."

---

## 5. COMMON INDEX INTERFACE

**[DECISION NEEDED — this does NOT currently exist.]** No common interface, base class, or protocol has been designed or written. The Flat index is implemented as inline procedural code, not as a class implementing any interface.

**Proposed (NOT yet implemented, NOT yet agreed — this is a suggestion only, to be confirmed by the team before 3 developers branch off):**

```python
class BaseIndex:
    def build(self, embeddings: np.ndarray, ids: list[int]) -> None: ...
    def add(self, embeddings: np.ndarray, ids: list[int]) -> None: ...
    def search(self, query_embedding: np.ndarray, k: int) -> list[dict]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

This is an **[ASSUMPTION / PROPOSAL]** only — not an existing API. If the team wants HNSW/IVF/PQ to be developed in parallel without integration chaos, **agreeing on a common interface like this (or any interface) is the single most important open decision**, and should be settled before Developer 1/2/3 start independent work. See Section 14.

---

## 6. VECTOR / DATA CONTRACT

- **Vector dimensionality**: 384 (fixed by `all-MiniLM-L6-v2`) [FACT]
- **dtype**: [ASSUMPTION] float32 (default `sentence-transformers .encode()` output) — not explicitly verified
- **Shape convention**: `(n_passages, 384)` for the corpus embedding matrix; `(1, 384)` for a single query embedding [FACT, from code]
- **Normalization**: NOT applied — embeddings are used raw from `.encode()` with no explicit L2-normalization step shown [FACT, based on absence in code]. **[DECISION NEEDED]**: normalization matters if switching to cosine similarity for other index types.
- **Distance metric**: L2/Euclidean for Flat [FACT]. **[DECISION NEEDED]** for HNSW/IVF/PQ — should be the same metric for a fair comparison, but this hasn't been explicitly agreed.
- **Document/chunk IDs**: sequential integers assigned via `enumerate()` during data cleaning; no chunking is performed — each MS MARCO passage is used whole, as one unit [FACT]
- **Metadata**: none beyond raw text [FACT]
- **Query format**: a raw text string, encoded at query time with the same model used for indexing [FACT]
- **Result format**: currently just `(distances, indices)` from FAISS, manually joined back to text via positional DataFrame lookup — no standardized result schema exists yet [FACT — see Section 5 for the proposed fix]
- **Top-k semantics**: standard nearest-neighbor top-k; `k=5` was used in the one test query so far, but there is no fixed evaluation `k` decided for the actual benchmark phase [DECISION NEEDED]

---

## 7. RAG PIPELINE (ACTUAL SCOPE OF THIS PROJECT)

```
MS MARCO passages (raw text)
        ↓
[NO chunking step — each passage used as-is]
        ↓
Embedding (sentence-transformers, all-MiniLM-L6-v2, 384-dim)
        ↓
Indexing (FAISS Flat implemented; HNSW/IVF/PQ pending; pgvector pending)
        ↓
Retrieval (index.search → top-k nearest neighbors)
        ↓
Results (raw text passages returned/printed)
        ↓
[NO generation step — this project stops at retrieval. There is no LLM
 call, no answer synthesis, no RAG "generation" phase implemented or
 currently planned in this codebase.]
```

[FACT] This project is scoped as an indexing/retrieval benchmarking study, not a full RAG question-answering system.

---

## 8. HNSW / IVF / PQ CURRENT PLAN

For all three: **only conceptual/educational explanations exist in the conversation (what the algorithm is, at a high level). No code, no library parameter choices, no API design, and no integration decisions have been made for any of them.** Everything below not marked [FACT] is therefore [UNKNOWN] or [DECISION NEEDED].

### HNSW (Hierarchical Navigable Small World)
- **Already decided**: Conceptual understanding only — HNSW builds a multi-layer graph connecting each vector to nearest neighbors; search descends through layers. Generally best recall-to-speed ratio, higher memory use than IVF/PQ. [FACT — this explanation was given, but it is educational, not an implementation decision]
- **Remaining to implement**: everything — index construction, parameter choices (`M`, `efConstruction`, `efSearch` in FAISS terms), integration with the (not-yet-built) common interface, save/load, benchmarking hooks
- **Expected API**: [UNKNOWN] — depends on Section 5's outcome
- **Expected integration points**: should plug into the same benchmark harness as Flat (not yet built) and expose the same `search()` result shape (not yet defined)
- **Constraints**: should use the same embedding model/vectors as Flat, for a fair comparison [ASSUMPTION, follows from the project's stated goal]
- **Library**: [ASSUMPTION] FAISS's `IndexHNSWFlat`, based on the project's existing FAISS dependency — not explicitly confirmed in conversation as the class to use

### IVF (Inverted File Index)
- **Already decided**: Conceptual only — partitions vector space into clusters via k-means; searches only nearest clusters at query time. [FACT — explanation given, not an implementation]
- **Remaining to implement**: everything — clustering/training step (IVF requires a `train()` call before `add()`, unlike Flat/HNSW), parameter choices (`nlist`, `nprobe` in FAISS terms), integration, save/load
- **Expected API**: [UNKNOWN]
- **Expected integration points**: same as HNSW above
- **Constraints**: IVF's `train()` step needs representative data — for this project that's presumably the same corpus embeddings [ASSUMPTION]
- **Library**: [ASSUMPTION] FAISS's `IndexIVFFlat` — not explicitly confirmed

### PQ (Product Quantization)
- **Already decided**: Conceptual only — compresses vectors into small codes via learned codebooks, reducing memory at some accuracy cost. [FACT — explanation given, not an implementation]
- **Remaining to implement**: everything
- **Expected API**: [UNKNOWN]
- **Standalone vs. combined with IVF**: **[DECISION NEEDED — explicitly undecided in the conversation.]** FAISS commonly offers both a standalone `IndexPQ` and a combined `IndexIVFPQ`. The conversation never specifies which one this project intends to use, or whether both should be tested. This must be settled before Developer 3 starts.
- **Library**: [ASSUMPTION] FAISS's `IndexPQ` and/or `IndexIVFPQ` — not explicitly confirmed

---

## 9. ARCHITECTURAL CONSTRAINTS (MUST NOT casually change)

Only constraints actually supported by the existing project:

- **Embedding model**: `all-MiniLM-L6-v2` (384-dim) is the one used throughout so far — changing it would invalidate any cross-index comparison [FACT-derived constraint]
- **Corpus**: MS MARCO v2.1 is the dataset used so far [FACT-derived constraint] — though corpus **size** is explicitly planned to change/scale up (10K/50K/200K), per the project's stated benchmarking goal [FACT]
- **File locations**: `data/` for corpus/embeddings, `indexes/` for built index files — established convention from the existing script [FACT-derived]
- **`requirements.txt` version pins**: do not silently change these without updating for the whole team — the `faiss-cpu` pin was already a source of real breakage (see Known Bugs #1–2); any future changes should be pushed immediately and communicated
- **No public API/persistence format exists yet** to be constrained — this is itself a gap (see Section 5 and 14), not an established convention to preserve

Explicitly **NOT** yet a fixed constraint, because it hasn't been decided (do not assume otherwise):
- Distance metric consistency across index types [DECISION NEEDED]
- Common interface shape [DECISION NEEDED]
- Result object format [DECISION NEEDED]

---

## 10. DEPENDENCIES

**Language**: Python. Version actually used varies by teammate's machine — [FACT] one teammate had to specifically avoid Python 3.14 (no FAISS wheels) and instead use Python 3.12. [ASSUMPTION] Python 3.11–3.12 is the safe working range for this project's dependencies; this was never written down as an explicit team-wide rule, only discovered/fixed reactively for one person.

**`requirements.txt`** (current, exact) [FACT]:
```
faiss-cpu>=1.9.0
sentence-transformers==3.0.1
pandas==2.2.2
numpy==1.26.4
psycopg2-binary==2.9.9
streamlit==1.36.0
matplotlib==3.9.0
```

**Used but MISSING from `requirements.txt`** [FACT — reproducibility gap, see Known Bugs #3]:
- `datasets` (Hugging Face — used to load MS MARCO)
- `scikit-learn` (used only in the optional `visualize_embeddings.py` script, for PCA)

**Testing libraries**: none chosen [UNKNOWN]

**Build/package tools**: plain `pip` + `venv` — no Poetry/PDM/etc. [FACT]

**Configuration files**: only `.gitignore` and `requirements.txt` exist; no `.env`, no config YAML/JSON, no `pyproject.toml` [FACT]

**Planned but not yet installed/used in code**: PostgreSQL 16 + `pgvector` extension, intended to run via Docker (`docker run ... pgvector/pgvector:pg16` was given as a command, but never actually executed/confirmed working in this conversation) [FACT that the command was given; UNKNOWN whether it was ever run successfully]

---

## 11. HOW TO RUN THE PROJECT

**Installation** [FACT, as instructed to the team]:
```bash
git clone https://github.com/Mohit-Nanda-Krishna/RagIndexing.git
cd RagIndexing
py -3.12 -m venv venv        # use 3.11 or 3.12 specifically — avoid 3.13+/3.14
venv\Scripts\activate        # Windows; `source venv/bin/activate` on Mac/Linux
pip install -r requirements.txt
pip install datasets scikit-learn   # NOT yet in requirements.txt — install manually for now
```

**Running the existing pipeline** [FACT]:
```bash
python run_pipeline.py
```
Expect several minutes of runtime (embedding generation is the slow step), requires internet access on every run (dataset + model are re-fetched each time — no caching implemented, see Known Bugs #4).

**Running tests**: [UNKNOWN] — none exist

**Linting/type checking**: [UNKNOWN] — none configured

**Examples/demo**: `run_pipeline.py` itself is the only demo; it ends by printing top-5 results for the hardcoded query `"What causes inflation?"`

---

## 12. TESTING

- **Existing tests**: none [FACT]
- **Test locations**: N/A
- **Coverage**: N/A
- **Expected behavior documented**: only informally — "pipeline should print 'DONE. Pipeline works end to end.' and 5 relevant passages" was used as the manual/eyeball correctness check, not an automated test [FACT]
- **Benchmarks/evaluation**: none implemented yet. The entire benchmark harness (recall@k, p50/p95 latency, memory, build time measurement) is planned but not started — this is arguably the highest-priority NOT STARTED item, since HNSW/IVF/PQ are only useful once they can be measured against Flat.

---

## 13. GIT / COLLABORATION CONSIDERATIONS

**Planned developer split** (per current task) [FACT, as instructed]:
- Developer 1 → HNSW
- Developer 2 → IVF
- Developer 3 → PQ

**[UNKNOWN / NEEDS VERIFICATION]**: The team has 4 members total (established earlier in the project), but this handoff only assigns 3 developers to HNSW/IVF/PQ. What the 4th team member is doing (benchmark harness? pgvector integration? dashboard?) is not specified in this handoff request — needs clarification from the team.

**Important prior team behavior to flag** [FACT]: this team has, until now, worked **directly on `main`, with no feature branches**, by explicit choice — because they were working in person together and coordinating verbally before each push. That workflow does **not** safely extend to 3 people independently implementing 3 different index types at the same time — simultaneous pushes to `main` from people working on unrelated features are much more likely to conflict or silently overwrite each other's work. **[DECISION NEEDED]**: the team should very likely switch to feature branches (e.g. `feature/hnsw-index`, `feature/ivf-index`, `feature/pq-index`) for this phase specifically, merging into `main` via pull request once each index is working. This is a recommendation, not something already decided by the team — raise it with them explicitly.

**Files each developer should modify**:
- Each developer should create their own new file (e.g., `hnsw_index.py`, `ivf_index.py`, `pq_index.py`) rather than all three editing `run_pipeline.py` directly — this avoids the most obvious merge conflict
- **[DECISION NEEDED]**: how the benchmark harness will import/call all four index types — this shared integration point doesn't exist yet, and whoever builds it will need to coordinate with all three developers on the common interface (Section 5)

**Shared files that should NOT be edited independently without coordination**:
- `requirements.txt` — if two developers both add new dependencies for their index type independently, this will conflict; coordinate before pushing
- `run_pipeline.py` — currently the single source of truth for data/embeddings; if it needs to be refactored to expose reusable functions (e.g., "load embeddings" as an importable function rather than inline script code) for the other index scripts to reuse, that refactor should be a **single, coordinated change**, not something 3 people do independently
- `.gitignore` — low risk, but still shared

**Likely merge-conflict areas**:
- `run_pipeline.py`, if multiple people try to modify it directly instead of creating separate files
- `requirements.txt`, when adding FAISS-adjacent dependencies (unlikely to need new ones, but possible if someone picks a different library)
- Any future shared "common interface" file, once it exists — this should probably be written once, by one person or by consensus, before HNSW/IVF/PQ work starts, not evolved independently by three people simultaneously

---

## 14. OPEN DESIGN QUESTIONS (must be decided, not assumed)

1. Will there be a common base class/interface for all index types, and if so, what exact methods/signatures? (Section 5)
2. Should all index types use the same distance metric (currently L2 for Flat), or should cosine similarity be considered, especially since sentence-transformer embeddings are often used with cosine similarity? (Section 6)
3. Should embeddings be normalized? (Section 6)
4. What is the standardized result format returned by `search()` — raw FAISS output, or a structured object (e.g., list of `{id, text, score}`)? (Section 5/6)
5. Should PQ be implemented standalone (`IndexPQ`), combined with IVF (`IndexIVFPQ`), or both? (Section 8)
6. What exact FAISS classes will be used for HNSW and IVF specifically (e.g., `IndexHNSWFlat` vs. alternatives; `IndexIVFFlat` vs. `IndexIVFPQ`)? Not confirmed.
7. What parameter values will be used for each index (HNSW's `M`/`efSearch`, IVF's `nlist`/`nprobe`, PQ's subvector count/bits), and will these be fixed or tuned/swept during benchmarking?
8. What is the plan for the 4th team member, given this handoff only covers 3 developers for HNSW/IVF/PQ?
9. Will the team switch to feature branches for this phase, given the collision risk of 3 people working in parallel? (Section 13)
10. Will `run_pipeline.py` be refactored into reusable functions/modules so HNSW/IVF/PQ scripts can import shared data-loading logic instead of duplicating it, and who owns that refactor?
11. Is `all-MiniLM-L6-v2` locked in as final, or still open to change before more indexes are built on top of it?
12. When will pgvector/PostgreSQL integration actually start, and who owns it?
13. What `k` value(s) will the actual benchmark use (only `k=5` has been used so far, in a single manual test query)?

---

## 15. EXACT CURRENT TODO (prioritized)

1. **[BLOCKING]** Resolve Section 14's open design questions #1–6 (interface, metric, result format, PQ variant, exact FAISS classes) — without these, HNSW/IVF/PQ work done independently will likely be incompatible and need rework
2. Decide and communicate the branching strategy for this phase (Section 13, question #9)
3. Add missing dependencies (`datasets`, `scikit-learn` if kept) to `requirements.txt` and push
4. Refactor `run_pipeline.py` if needed so data-loading/embedding logic is reusable by the new index scripts, rather than each developer re-copying it
5. Developer 1: implement HNSW index (new file, e.g. `hnsw_index.py`), following whatever interface is agreed in item 1
6. Developer 2: implement IVF index (new file, e.g. `ivf_index.py`)
7. Developer 3: implement PQ index (new file, e.g. `pq_index.py`)
8. Build the benchmark harness (recall@k, p50/p95 latency, memory, build time) that can run against all four index types uniformly
9. Scale corpus size testing to 10K / 50K / 200K passages (currently only ~2,000)
10. Begin pgvector/PostgreSQL integration (owner unassigned — see open question #8)
11. Build the Streamlit dashboard
12. Add basic tests/sanity checks (currently zero exist)

---

## 16. HANDOFF INSTRUCTIONS FOR A NEW CLAUDE

If you are a fresh Claude instance picking this up:

- Treat this document as your project context — it is a reconstruction from a prior conversation, not the ground truth of the actual repository. **Before making any assumptions, inspect the actual repository files yourself** (the real `run_pipeline.py`, `requirements.txt`, `.gitignore`, and any new files added since this document was generated) — the repo may have changed since this was written.
- Preserve the existing architecture and the Flat index implementation as the reference — do not rewrite or "clean up" working code unless explicitly asked.
- Do not silently resolve any item marked **[DECISION NEEDED]** in Sections 8, 9, or 14 — surface it to the person you're working with and get an explicit answer before building on top of an assumption.
- If you are helping with only ONE of HNSW, IVF, or PQ (per the 3-developer split), focus only on that component. Do not modify another developer's index file, and be cautious about modifying shared files (`run_pipeline.py`, `requirements.txt`, any common interface file) without flagging that the change affects other developers.
- Ask before making any major architectural change (e.g., changing the distance metric, changing the embedding model, redesigning the interface) — these affect the whole team, not just one component.
- If asked to implement only an assigned component, do exactly that — do not also try to "complete" the benchmark harness, dashboard, or other developers' index types unless asked.

---

## 17. COMPACT CONTEXT VERSION

*(Paste this shorter version into a new conversation if the full document above is too long.)*

> We're building a DBMS course project benchmarking vector index strategies (Flat, IVF, HNSW, PQ) for RAG retrieval — NOT a full RAG system with generation, just the indexing/retrieval layer. Repo: `github.com/Mohit-Nanda-Krishna/RagIndexing`. Team of 4; 3 developers are splitting HNSW/IVF/PQ implementation (4th person's task unassigned/unknown).
>
> **Done**: `run_pipeline.py` downloads a ~2,000-passage MS MARCO v2.1 subset, embeds it with `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim), and builds a FAISS `IndexFlatL2` (exact/baseline index) — saved to `indexes/flat.index`. One test query proven working (L2 distance, `k=5`, positional ID lookup back into `data/passages.csv`). No classes/interfaces exist yet — it's all procedural script code.
>
> **Not started**: HNSW, IVF, PQ implementations; pgvector/PostgreSQL integration; the benchmark harness (recall@k, p50/p95 latency, memory, build time); multi-scale corpus testing (10K/50K/200K); Streamlit dashboard; any tests.
>
> **Critical open decisions before parallel work starts**: (1) a common index interface (`build`/`add`/`search`/`save`/`load` — proposed but NOT agreed), (2) whether to keep L2 distance or switch to cosine for all indexes, (3) standardized result format (currently raw FAISS output, no structured object), (4) whether PQ is standalone (`IndexPQ`) or combined with IVF (`IndexIVFPQ`), (5) exact FAISS classes/parameters for HNSW and IVF, (6) branching strategy — team has been pushing directly to `main` with no branches, which is risky for 3 people working in parallel now.
>
> **Known gotchas**: `faiss-cpu` needs `>=1.9.0` (not `==1.8.0`, doesn't exist); Python 3.14 has no FAISS wheels, use 3.11/3.12; `datasets` and `scikit-learn` are used but missing from `requirements.txt`.
>
> Do not invent answers to the open decisions above — ask the team. Do not touch the separate "adaptive RAG retrieval" solo project — different codebase, different scope.