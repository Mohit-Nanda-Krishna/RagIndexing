"""Build and query an IVF-Flat index over the MS MARCO artifacts.

Run ``python run_pipeline.py`` once first to create ``data/passages.csv`` and
``data/embeddings.npy``.  This script deliberately leaves that shared baseline
pipeline unchanged, then lets IVF experiments use any number of queries.

Examples
--------
python run_ivf.py
python run_ivf.py --query "What causes inflation?" --query "How do vaccines work?"
python run_ivf.py --nlist 100 --nprobe 10 --k 5 --query "What causes inflation?"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from ivf_index import IVFIndex


DEFAULT_QUERIES = [
    "What causes inflation?",
    "How do vaccines work?",
    "What is climate change?",
]
DATA_PATH = Path("data/passages.csv")
EMBEDDINGS_PATH = Path("data/embeddings.npy")
INDEX_PATH = Path("indexes/ivf.index")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Query to retrieve. Repeat this option for multiple queries.",
    )
    parser.add_argument("--k", type=int, default=5, help="Results per query (default: 5).")
    parser.add_argument(
        "--nlist", type=int, default=100, help="IVF clusters to train (default: 100)."
    )
    parser.add_argument(
        "--nprobe", type=int, default=10, help="IVF clusters searched per query (default: 10)."
    )
    parser.add_argument(
        "--index-path", type=Path, default=INDEX_PATH, help="Where to save the IVF index."
    )
    return parser.parse_args()


def load_corpus() -> tuple[pd.DataFrame, np.ndarray]:
    """Load the MS MARCO passages and their embeddings from the baseline run."""
    missing = [str(path) for path in (DATA_PATH, EMBEDDINGS_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing baseline artifact(s): "
            + ", ".join(missing)
            + ". Run `python run_pipeline.py` once before running this script."
        )

    passages = pd.read_csv(DATA_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    if len(passages) != len(embeddings):
        raise ValueError(
            "The number of passages and embeddings differs. Re-run `python run_pipeline.py`."
        )
    return passages, embeddings


def print_results(
    passages: pd.DataFrame, query: str, distances: np.ndarray, ids: np.ndarray
) -> None:
    print(f"\nTop {len(ids)} IVF results for query: {query}")
    for rank, (distance, vector_id) in enumerate(zip(distances, ids), start=1):
        if vector_id < 0:
            continue  # FAISS uses -1 when fewer than k results are available.
        print(f"{rank}. [L2={distance:.4f}] {passages.iloc[vector_id]['text'][:150]}")


def main(arguments: argparse.Namespace | None = None) -> None:
    args = parse_arguments() if arguments is None else arguments
    if args.k < 1:
        raise ValueError("--k must be a positive integer")
    if args.nlist < 1:
        raise ValueError("--nlist must be a positive integer")
    if args.nprobe < 1 or args.nprobe > args.nlist:
        raise ValueError("--nprobe must be between 1 and --nlist")

    passages, embeddings = load_corpus()
    if args.nlist > len(embeddings):
        raise ValueError("--nlist cannot exceed the number of MS MARCO passages")

    print(f"Building IVF-Flat index for {len(embeddings)} MS MARCO passages...")
    index = IVFIndex(nlist=args.nlist, nprobe=args.nprobe)
    index.build(embeddings)
    args.index_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(args.index_path)
    print(f"Saved IVF index to {args.index_path} (nlist={args.nlist}, nprobe={args.nprobe})")

    print("Loading embedding model for queries...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    queries: Sequence[str] = args.queries or DEFAULT_QUERIES
    query_embeddings = model.encode(list(queries))
    distances, ids = index.search(query_embeddings, k=min(args.k, index.ntotal))

    for query, query_distances, query_ids in zip(queries, distances, ids):
        print_results(passages, query, query_distances, query_ids)


if __name__ == "__main__":
    main()
