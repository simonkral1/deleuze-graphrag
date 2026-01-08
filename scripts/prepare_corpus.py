#!/usr/bin/env python3
"""
Pipeline that converts the PDF corpus (books + seminars) into clean text files
for GraphRAG and produces chunked JSONL suitable for Cohere embeddings.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

# Optional dependencies: langdetect + tiktoken. Fallbacks keep script usable offline.
try:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 42
except Exception:  # pragma: no cover - fallback when langdetect missing
    DetectorFactory = None
    detect = None

try:
    import tiktoken
except Exception:  # pragma: no cover - fallback when tiktoken missing
    tiktoken = None


@dataclass
class Document:
    doc_id: str
    path: Path
    language: str
    collection: str
    doc_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Deleuze corpus for GraphRAG + Cohere.")
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        default=["pdf_books", "seminars"],
        help="Directories that contain the PDF corpus.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("deleuze_corpus"),
        help="Root directory for generated artifacts.",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=1200,
        help="Target token length per chunk for Cohere embeddings.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=float,
        default=0.15,
        help="Overlap ratio between chunks (0-0.5).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild text outputs even if they already exist.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional limit for quick experiments.",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text


def discover_documents(source_dirs: Sequence[str]) -> List[Document]:
    docs: List[Document] = []
    root = Path.cwd()
    for dir_name in source_dirs:
        base = root / dir_name
        if not base.exists():
            continue
        for pdf_path in sorted(base.rglob("*.pdf")):
            rel_parts = pdf_path.relative_to(root).parts
            language = "unknown"
            for part in rel_parts:
                if part.lower() in {"english", "french"}:
                    language = part.lower()
                    break
            doc_id = slugify(pdf_path.stem)
            collection = rel_parts[0]
            doc_type = rel_parts[1] if len(rel_parts) > 1 else collection
            docs.append(
                Document(
                    doc_id=doc_id,
                    path=pdf_path,
                    language=language,
                    collection=collection,
                    doc_type=doc_type,
                )
            )
    return docs


def run_pdftotext(pdf_path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-enc", "UTF-8", "-layout", str(pdf_path), "-"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed for {pdf_path}: {result.stderr.strip()}")
    return result.stdout


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "")
    # Remove page numbers (lines with just digits)
    text = re.sub(r"\n\d+\n", "\n", text)
    # Fix hyphenated line breaks
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing spaces
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def detect_language(text: str, fallback: str) -> str:
    snippet = text[:5000].lower()
    if detect:
        try:
            return detect(snippet)
        except Exception:
            pass
    french_markers = [" le ", " la ", " les ", " des ", " une ", " que ", " du "]
    english_markers = [" the ", " and ", " of ", " with ", " without ", " toward "]
    fr_score = sum(snippet.count(token) for token in french_markers)
    en_score = sum(snippet.count(token) for token in english_markers)
    if fr_score > en_score and fr_score > 0:
        return "fr"
    if en_score > fr_score and en_score > 0:
        return "en"
    return fallback


class Tokenizer:
    def __init__(self) -> None:
        self.encoding = None
        if tiktoken:
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.encoding = None

    def encode(self, text: str) -> List:
        if self.encoding:
            return self.encoding.encode(text)
        return text.split()

    def decode(self, tokens: List) -> str:
        if self.encoding:
            return self.encoding.decode(tokens)
        return " ".join(tokens)


def chunk_tokens(tokens: List[int], chunk_size: int, overlap_ratio: float) -> Iterable[tuple[int, int]]:
    if overlap_ratio >= 0.5:
        raise ValueError("Overlap ratio must be less than 0.5 to avoid infinite loops.")
    overlap = int(chunk_size * overlap_ratio)
    step = chunk_size - overlap
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        yield start, end
        if end == len(tokens):
            break
        start += step


def ensure_dirs(output_root: Path) -> None:
    for sub in ["raw_text", "clean_text", "chunks", "metadata"]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    docs = discover_documents(args.source_dirs)
    if args.max_docs:
        docs = docs[: args.max_docs]
    ensure_dirs(args.output_root)
    graph_input_dir = Path("graphrag_project") / "input" / "data"
    graph_input_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_root / "metadata" / "manifest.jsonl"
    chunks_path = args.output_root / "chunks" / "chunks.jsonl"

    tokenizer = Tokenizer()
    doc_count = 0
    chunk_count = 0
    with manifest_path.open("w", encoding="utf-8") as mf, chunks_path.open(
        "w", encoding="utf-8"
    ) as cf:
        for doc in docs:
            relative_name = f"{doc.doc_id}.txt"
            clean_text_path = args.output_root / "clean_text" / relative_name
            graph_text_path = graph_input_dir / relative_name

            if clean_text_path.exists() and not args.force:
                text = clean_text_path.read_text(encoding="utf-8")
            else:
                try:
                    raw_text = run_pdftotext(doc.path)
                except RuntimeError as err:
                    print(f"[WARN] {err}", file=sys.stderr)
                    continue
                text = clean_text(raw_text)
                clean_text_path.write_text(text, encoding="utf-8")
            graph_text_path.write_text(text, encoding="utf-8")

            language_detected = detect_language(text, doc.language)
            tokens = tokenizer.encode(text)
            manifest_entry = {
                "doc_id": doc.doc_id,
                "source_path": str(doc.path),
                "graph_text_path": str(graph_text_path),
                "language_hint": doc.language,
                "language_detected": language_detected,
                "collection": doc.collection,
                "doc_type": doc.doc_type,
                "num_chars": len(text),
                "num_tokens": len(tokens),
            }
            mf.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
            doc_count += 1

            for chunk_index, (start, end) in enumerate(
                chunk_tokens(tokens, args.chunk_tokens, args.chunk_overlap)
            ):
                chunk_id = f"{doc.doc_id}-{chunk_index:04d}"
                chunk_text = tokenizer.decode(tokens[start:end])
                chunk_entry = {
                    "id": chunk_id,
                    "doc_id": doc.doc_id,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "language": language_detected,
                    "source_path": str(doc.path),
                    "collection": doc.collection,
                    "doc_type": doc.doc_type,
                    "token_start": start,
                    "token_end": end,
                }
                cf.write(json.dumps(chunk_entry, ensure_ascii=False) + "\n")
                chunk_count += 1

    print(
        f"Processed {doc_count} documents -> "
        f"{chunk_count} chunks.\n"
        f"Manifest: {manifest_path}\nChunks: {chunks_path}\n"
        f"GraphRAG input dir: {graph_input_dir}"
    )


if __name__ == "__main__":
    main()
