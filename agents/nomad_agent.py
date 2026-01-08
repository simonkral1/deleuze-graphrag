#!/usr/bin/env python3
"""
Nomad agent that routes questions between GraphRAG community summaries and a Cohere-powered
quote-level vector index.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Sequence

import chromadb
from chromadb.config import Settings
import cohere
import numpy as np
import pandas as pd
from openai import OpenAI


def load_dataframe(path: Path) -> pd.DataFrame:
    if path.is_dir():
        for candidate in ["community_reports.parquet", "community_reports.parq"]:
            candidate_path = path / candidate
            if candidate_path.exists():
                path = candidate_path
                break
    if not path.exists():
        raise FileNotFoundError(f"Missing community report file at {path}")
    if path.suffix == ".parquet" or path.suffix == ".parq":
        return pd.read_parquet(path)
    return pd.read_json(path, orient="records", lines=True)


def extract_text_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"None of {candidates} found in dataframe columns {df.columns}")


class GraphCommunityIndex:
    def __init__(self, report_path: Path, co_client: cohere.Client, embed_model: str):
        self.cohere = co_client
        self.model = embed_model
        df = load_dataframe(report_path)
        summary_col = extract_text_column(df, ["summary", "summary_text", "community_summary", "content"])
        title_col = (
            extract_text_column(df, ["title", "name", "community", "community_title"])
            if any(col in df.columns for col in ["title", "name", "community", "community_title"])
            else summary_col
        )
        self.records = []
        for idx, row in df.iterrows():
            summary = str(row[summary_col]).strip()
            if not summary:
                continue
            title = str(row[title_col]).strip()
            rec_id = str(row.get("id", row.get("community_id", idx)))
            self.records.append(
                {
                    "id": rec_id,
                    "title": title,
                    "summary": summary,
                    "level": row.get("level"),
                }
            )
        if not self.records:
            raise ValueError("No community records found.")
        self.embeddings = self.embed_records([rec["summary"] for rec in self.records])

    def embed_records(self, texts: List[str]) -> np.ndarray:
        batch_size = 64
        vectors: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            resp = self.cohere.embed(model=self.model, texts=batch, input_type="search_document")
            vectors.extend(resp.embeddings)
        matrix = np.array(vectors, dtype=np.float32)
        self.magnitudes = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix

    def search(self, question_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        q = question_embedding.reshape(1, -1)
        q_norm = np.linalg.norm(q)
        scores = (self.embeddings @ q.T).flatten() / (self.magnitudes.flatten() * q_norm + 1e-8)
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            rec = self.records[int(idx)].copy()
            rec["score"] = float(scores[idx])
            results.append(rec)
        return results


class QuoteVectorIndex:
    def __init__(self, persist_dir: Path, collection: str, co_client: cohere.Client, embed_model: str):
        self.cohere = co_client
        self.model = embed_model
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_collection(name=collection)

    def search(self, question: str, top_k: int = 5) -> List[dict]:
        response = self.cohere.embed(
            model=self.model,
            texts=[question],
            input_type="search_query",
        )
        query_embedding = response.embeddings[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "ids", "distances"],
        )
        hits = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for idx, doc, meta, dist in zip(ids, documents, metadatas, distances):
            hits.append(
                {
                    "id": idx,
                    "text": doc,
                    "metadata": meta,
                    "distance": dist,
                }
            )
        return hits


class ResponseSynthesizer:
    def __init__(self, model: str):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for synthesis.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def build_prompt(self, question: str, communities: List[dict], quotes: List[dict]) -> List[dict]:
        community_lines = "\n".join(
            f"- ({c['score']:.2f}) {c['title']}: {c['summary']}" for c in communities
        )
        quote_lines = "\n".join(
            f"- ({q['metadata'].get('doc_id')} #{q['id']}) {q['text']}" for q in quotes
        )
        user_content = f"""Question: {question}

Graph insights:
{community_lines or 'None'}

Quotes:
{quote_lines or 'None'}

Synthesize an answer that keeps the Deleuzian terminology intact. Cite doc_id + chunk id for quotes."""
        return [
            {
                "role": "system",
                "content": "You are the Nomad agent. Blend conceptual connections with precise citations.",
            },
            {"role": "user", "content": user_content},
        ]

    def generate(self, question: str, communities: List[dict], quotes: List[dict]) -> str:
        messages = self.build_prompt(question, communities, quotes)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )
        return completion.choices[0].message.content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Deleuze Nomad agent.")
    parser.add_argument("--question", required=True, help="User question.")
    parser.add_argument(
        "--community-path",
        type=Path,
        default=Path("graphrag_project/output/community_reports.parquet"),
        help="Path to GraphRAG community summary parquet/JSON.",
    )
    parser.add_argument(
        "--vector-store",
        type=Path,
        default=Path("vector_store"),
        help="Chroma persistent directory containing the quote index.",
    )
    parser.add_argument(
        "--collection",
        default="deleuze_quotes",
        help="Quote collection name.",
    )
    parser.add_argument(
        "--cohere-model",
        default="embed-v4.0",
        help="Cohere embedding model to use.",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="LLM model for synthesis.",
    )
    parser.add_argument(
        "--top-communities",
        type=int,
        default=5,
        help="Number of graph communities to retrieve.",
    )
    parser.add_argument(
        "--top-quotes",
        type=int,
        default=5,
        help="Number of quotes to retrieve.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    co_api_key = os.environ.get("COHERE_API_KEY")
    if not co_api_key:
        raise ValueError("COHERE_API_KEY is required.")
    co_client = cohere.Client(co_api_key)

    question_embed = co_client.embed(
        model=args.cohere_model,
        texts=[args.question],
        input_type="search_query",
    ).embeddings[0]
    question_vec = np.array(question_embed, dtype=np.float32)

    graph_index = GraphCommunityIndex(args.community_path, co_client, args.cohere_model)
    communities = graph_index.search(question_vec, top_k=args.top_communities)

    quote_index = QuoteVectorIndex(args.vector_store, args.collection, co_client, args.cohere_model)
    quotes = quote_index.search(args.question, top_k=args.top_quotes)

    synthesizer = ResponseSynthesizer(model=args.openai_model)
    answer = synthesizer.generate(args.question, communities, quotes)

    payload = {
        "question": args.question,
        "communities": communities,
        "quotes": quotes,
        "answer": answer,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
