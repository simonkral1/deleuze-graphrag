#!/usr/bin/env python3
"""
Load pre-computed embeddings from vectors.parquet into Chroma vector DB.
This avoids re-embedding and saves API costs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load pre-computed vectors into Chroma.")
    parser.add_argument(
        "--vectors-path",
        type=Path,
        default=Path("deleuze_corpus/vectors.parquet"),
        help="Path to the vectors parquet file with pre-computed embeddings.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("vector_store"),
        help="Directory where the Chroma DB will persist.",
    )
    parser.add_argument(
        "--collection",
        default="deleuze_quotes",
        help="Name of the Chroma collection.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of chunks to add per Chroma batch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading vectors from {args.vectors_path}...")
    df = pd.read_parquet(args.vectors_path)
    total_chunks = len(df)
    print(f"Found {total_chunks} chunks with embeddings")

    # Verify vector dimensions
    sample_vec = df.iloc[0]["vector"]
    vec_dim = len(sample_vec)
    print(f"Vector dimensions: {vec_dim}")

    # Initialize Chroma
    print(f"Initializing Chroma at {args.persist_dir}...")
    chroma_client = chromadb.PersistentClient(
        path=str(args.persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )

    # Create or get collection
    collection = chroma_client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Adding {total_chunks} chunks to collection '{args.collection}'...")

    # Batch insert
    for start_idx in range(0, total_chunks, args.batch_size):
        end_idx = min(start_idx + args.batch_size, total_chunks)
        batch_df = df.iloc[start_idx:end_idx]

        ids = batch_df["id"].tolist()
        documents = batch_df["text"].tolist()
        embeddings = [vec.tolist() if isinstance(vec, np.ndarray) else vec
                     for vec in batch_df["vector"].tolist()]
        metadatas = [
            {
                "doc_id": row["doc_id"],
                "language": row["language"],
                "collection": row["collection"],
                "doc_type": row["doc_type"],
                "source_path": row["source_path"],
            }
            for _, row in batch_df.iterrows()
        ]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        progress = (end_idx / total_chunks) * 100
        sys.stdout.write(f"\rProgress: {progress:.1f}% ({end_idx}/{total_chunks})")
        sys.stdout.flush()

    sys.stdout.write("\n")
    print(f"✓ Successfully loaded {total_chunks} chunks into Chroma collection '{args.collection}'")

    # Verify
    count = collection.count()
    print(f"✓ Collection now contains {count} documents")


if __name__ == "__main__":
    main()
