"""Build and query a standalone Product Quantization index over MS MARCO.

Run ``python run_pipeline.py`` once first to generate the corpus embeddings.

Example:
    python run_pq.py "What causes inflation?" --m 48 --nbits 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from pq_index import PQIndex


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search MS MARCO embeddings with a standalone FAISS IndexPQ."
    )
    parser.add_argument("query", nargs="?", default="What causes inflation?")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return.")
    parser.add_argument("--m", type=int, default=48, help="Number of PQ subvectors.")
    parser.add_argument(
        "--nbits",
        type=int,
        default=4,
        help="Bits per subvector code (default: 4, suitable for the 2K corpus).",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("indexes/pq.index"),
        help="Where to save the built PQ index.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    project_dir = Path(__file__).resolve().parent
    passages_path = project_dir / "data" / "passages.csv"
    embeddings_path = project_dir / "data" / "embeddings.npy"
    if not passages_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            "MS MARCO artifacts are missing. Run `python run_pipeline.py` first."
        )

    passages = pd.read_csv(passages_path)
    embeddings = np.load(embeddings_path).astype(np.float32, copy=False)
    if len(passages) != len(embeddings):
        raise ValueError("passages.csv and embeddings.npy have different row counts")

    index = PQIndex(embeddings.shape[1], m=args.m, nbits=args.nbits)
    index.build(embeddings)
    index_path = args.index_path if args.index_path.is_absolute() else project_dir / args.index_path
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(index_path)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([args.query]).astype(np.float32, copy=False)
    distances, ids = index.search(query_embedding, args.k)

    print(f"Built standalone PQ index with {index.ntotal} vectors of dimension {index.dimension}.")
    print(
        f"Saved PQ index to {index_path} (m={index.m}, nbits={index.nbits}, "
        f"code size={index.code_size} bytes/vector)"
    )
    print(f"\nTop {args.k} PQ results for query: {args.query}\n")
    for rank, (vector_id, distance) in enumerate(zip(ids[0], distances[0]), start=1):
        if vector_id == -1:
            continue
        print(f"{rank}. ID={vector_id}, approximate squared L2 distance={distance:.4f}")
        print(f"   {passages.iloc[vector_id]['text'][:200]}\n")


if __name__ == "__main__":
    main()
