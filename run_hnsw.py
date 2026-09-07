"""Run text queries against an HNSW index built from the MS MARCO embeddings.

Example:
    python run_hnsw.py "What causes inflation?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from indexes.hnsw_index import HNSWIndex


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search the MS MARCO embeddings with the FAISS HNSW index."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="What causes inflation?",
        help="Text query to search for.",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of results to return.")
    parser.add_argument("--M", type=int, default=32, help="HNSW graph connectivity.")
    parser.add_argument(
        "--efConstruction",
        type=int,
        default=200,
        help="HNSW construction candidate-list size.",
    )
    parser.add_argument(
        "--efSearch",
        type=int,
        default=128,
        help="HNSW query candidate-list size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    project_dir = Path(__file__).resolve().parent
    passages_path = project_dir / "data" / "passages.csv"
    embeddings_path = project_dir / "data" / "embeddings.npy"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {embeddings_path}. Run run_pipeline.py first."
        )

    passages = pd.read_csv(passages_path)
    embeddings = np.load(embeddings_path).astype(np.float32, copy=False)
    if len(passages) != len(embeddings):
        raise ValueError(
            "passages.csv and embeddings.npy have different row counts; regenerate them "
            "together with run_pipeline.py."
        )

    index = HNSWIndex(
        dimension=embeddings.shape[1],
        M=args.M,
        efConstruction=args.efConstruction,
        efSearch=args.efSearch,
    )
    index.build(embeddings)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([args.query]).astype(np.float32, copy=False)
    distances, indices = index.search(query_embedding, args.k)

    print(f"Built HNSW index with {index.ntotal} vectors of dimension {index.dimension}.")
    print(
        f"Parameters: M={index.M}, efConstruction={index.efConstruction}, "
        f"efSearch={index.efSearch}"
    )
    print(f"\nTop {args.k} HNSW results for query: {args.query}\n")
    for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx == -1:
            continue
        print(f"{rank}. ID={idx}, squared L2 distance={distance:.4f}")
        print(f"   {passages.iloc[idx]['text'][:200]}\n")


if __name__ == "__main__":
    main()
