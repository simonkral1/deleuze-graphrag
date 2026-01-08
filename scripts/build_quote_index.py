#!/usr/bin/env python3
"""
Embeds the chunked corpus with Cohere embed-v4.0 and stores it in a Chroma vector DB.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List

import chromadb
from chromadb.config import Settings
import cohere


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build quote-level vector index with Cohere.")
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("deleuze_corpus/chunks/chunks.jsonl"),
        help="Path to the chunk JSONL file created by prepare_corpus.py.",
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
        default=64,
        help="Number of chunks to embed per Cohere request.",
    )
    parser.add_argument(
        "--model",
        default="embed-v4.0",
        help="Cohere embedding model to use.",
    )
    parser.add_argument(
        "--input-type",
        default="search_document",
        help="Cohere input_type parameter (search_document|search_query|classification|clustering).",
    )
    return parser.parse_args()


def load_chunks(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def batch_iter(sequence: List[dict], batch_size: int) -> Iterable[List[dict]]:
    for i in range(0, len(sequence), batch_size):
        yield sequence[i : i + batch_size]


def render_progress(done: int, total: int, total_chunks: int, processed_chunks: int) -> None:
    width = 32
    filled = int(width * done / max(total, 1))
    bar = "=" * filled + "." * (width - filled)
    sys.stdout.write(
        f"\r[{bar}] batches {done}/{total} | chunks {processed_chunks}/{total_chunks}"
    )
    sys.stdout.flush()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY environment variable is required.")

    chunks = load_chunks(args.chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {args.chunks_path}")

    co = cohere.Client(api_key)
    chroma_client = chromadb.PersistentClient(
        path=str(args.persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = len(chunks)
    total_batches = math.ceil(total_chunks / args.batch_size)
    processed = 0
    start = time.time()
    for batch_idx, batch in enumerate(batch_iter(chunks, args.batch_size), start=1):
        texts = [item["text"] for item in batch]
        ids = [item["id"] for item in batch]
        metadatas = [
            {
                "doc_id": item["doc_id"],
                "language": item["language"],
                "collection": item["collection"],
                "doc_type": item["doc_type"],
                "source_path": item["source_path"],
            }
            for item in batch
        ]
        response = co.embed(
            model=args.model,
            texts=texts,
            input_type=args.input_type,
        )
        embeddings = response.embeddings
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        processed += len(batch)
        render_progress(batch_idx, total_batches, total_chunks, processed)

    elapsed = time.time() - start
    sys.stdout.write("\n")
    print(
        f"Finished indexing {processed} chunks into collection '{args.collection}' "
        f"in {elapsed:.1f}s."
    )


if __name__ == "__main__":
    main()
