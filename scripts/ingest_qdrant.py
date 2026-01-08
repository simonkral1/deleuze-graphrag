#!/usr/bin/env python3
"""
Embed chunks with Cohere embed-v4.0 and upsert into a Qdrant collection.
This avoids string IDs (uses numeric IDs) and throttles requests to respect rate limits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm
import cohere
from cohere.errors import TooManyRequestsError
from cohere.core.api_error import ApiError
from qdrant_client import QdrantClient
from qdrant_client.http import models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest chunk JSONL into Qdrant using Cohere embeddings.")
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("deleuze_corpus/chunks/chunks.jsonl"),
        help="Path to chunk JSONL produced by prepare_corpus.py.",
    )
    parser.add_argument(
        "--collection",
        default="deleuze_corpus",
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Chunks per embedding request.",
    )
    parser.add_argument(
        "--model",
        default="embed-v4.0",
        help="Cohere embedding model.",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=1536,
        help="Embedding dimension (set to 512 if you request that dimension from Cohere).",
    )
    parser.add_argument(
        "--min-interval-seconds",
        type=float,
        default=0.75,
        help="Minimum spacing between embedding calls to avoid rate limits (100 RPM -> 0.6s).",
    )
    parser.add_argument(
        "--truncate",
        default="NONE",
        choices=["NONE", "START", "END", "LEFT", "RIGHT"],
        help="Truncation mode for Cohere embeddings.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the Qdrant collection before ingest.",
    )
    parser.add_argument(
        "--qdrant-timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds for Qdrant requests.",
    )
    return parser.parse_args()


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def main() -> None:
    args = parse_args()

    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_key = os.environ.get("QDRANT_API_KEY")
    cohere_key = os.environ.get("COHERE_API_KEY")
    if not (qdrant_url and qdrant_key and cohere_key):
        sys.exit("QDRANT_URL, QDRANT_API_KEY, and COHERE_API_KEY must be set.")

    chunks = load_chunks(args.chunks_path)
    total = len(chunks)
    if total == 0:
        sys.exit(f"No chunks found at {args.chunks_path}")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=args.qdrant_timeout)
    if args.recreate or not client.collection_exists(args.collection):
        client.recreate_collection(
            collection_name=args.collection,
            vectors_config=models.VectorParams(size=args.vector_size, distance=models.Distance.COSINE),
        )

    co = cohere.Client(cohere_key)

    last_call = 0.0

    def embed_batch(texts: List[str]) -> List[List[float]]:
        nonlocal last_call
        backoff = 5.0
        while True:
            # throttle
            since = time.time() - last_call
            if since < args.min_interval_seconds:
                time.sleep(args.min_interval_seconds - since)
            try:
                resp = co.embed(
                    model=args.model,
                    texts=texts,
                    input_type="search_document",
                    truncate=args.truncate,
                )
                last_call = time.time()
                return resp.embeddings
            except (TooManyRequestsError, ApiError):
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
            except Exception:
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    point_id = 0
    for i in tqdm(range(0, total, args.batch_size), desc="Embedding+Upsert", unit="batch"):
        batch = chunks[i : i + args.batch_size]
        texts = [b["text"] for b in batch]
        ids = list(range(point_id, point_id + len(batch)))
        payloads = [
            {
                "chunk_id": b["id"],
                "doc_id": b["doc_id"],
                "language": b["language"],
                "collection": b["collection"],
                "doc_type": b["doc_type"],
                "source_path": b["source_path"],
            }
            for b in batch
        ]
        vectors = embed_batch(texts)
        backoff = 5.0
        while True:
            try:
                client.upsert(
                    collection_name=args.collection,
                    points=models.Batch(ids=ids, vectors=vectors, payloads=payloads),
                )
                break
            except Exception:
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
        point_id += len(batch)

    print(f"Finished: {total} chunks into Qdrant collection '{args.collection}' (dim={args.vector_size}).")


if __name__ == "__main__":
    main()
