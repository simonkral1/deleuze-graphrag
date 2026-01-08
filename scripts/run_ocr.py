#!/usr/bin/env python3
"""
OCR fallback for PDFs that failed pdftotext. Converts page-by-page with pdftoppm + tesseract.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List


def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def get_page_count(pdf_path: Path) -> int:
    info = run(["pdfinfo", str(pdf_path)])
    if info.returncode != 0:
        raise RuntimeError(f"pdfinfo failed for {pdf_path}: {info.stderr.strip()}")
    for line in info.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":")[1].strip())
    raise RuntimeError(f"Could not read page count for {pdf_path}")


def ocr_pdf(pdf_path: Path, out_txt: Path, lang: str = "eng", dpi: int = 200) -> None:
    page_count = get_page_count(pdf_path)
    print(f"[OCR] {pdf_path.name} ({page_count} pages) -> {out_txt.name} [{lang}]")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    text_parts: List[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for page in range(1, page_count + 1):
            base = Path(tmpdir) / "page"
            ppm_prefix = str(base)
            extract = run(
                [
                    "pdftoppm",
                    "-r",
                    str(dpi),
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    str(pdf_path),
                    ppm_prefix,
                ]
            )
            if extract.returncode != 0:
                print(f"[WARN] pdftoppm failed on page {page} of {pdf_path.name}: {extract.stderr.strip()}")
                continue
            images = sorted(Path(tmpdir).glob("page-*.ppm"))
            if not images:
                print(f"[WARN] No images generated for page {page} of {pdf_path.name}")
                continue
            img_path = images[0]
            ocr = run(["tesseract", str(img_path), "stdout", "-l", lang])
            if ocr.returncode != 0:
                print(f"[WARN] tesseract failed on page {page} of {pdf_path.name}: {ocr.stderr.strip()}")
                continue
            text_parts.append(ocr.stdout)
            for img in images:
                img.unlink(missing_ok=True)

    out_txt.write_text("\n".join(text_parts), encoding="utf-8")


def load_zero_token_docs(manifest_path: Path) -> List[dict]:
    docs = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        if obj.get("num_tokens", 1) == 0:
            docs.append(obj)
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR PDFs that failed pdftotext.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("deleuze_corpus/metadata/manifest.jsonl"),
        help="Path to manifest.jsonl with num_tokens info.",
    )
    parser.add_argument(
        "--doc-id",
        action="append",
        default=None,
        help="Specific doc_id(s) to OCR. If omitted, all zero-token docs are processed.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for pdftoppm.",
    )
    parser.add_argument(
        "--lang-en",
        default="eng",
        help="Language code for English documents.",
    )
    parser.add_argument(
        "--lang-fr",
        default="fra",
        help="Language code for French documents.",
    )
    args = parser.parse_args()

    manifest = args.manifest
    if not manifest.exists():
        print(f"Manifest not found: {manifest}", file=sys.stderr)
        sys.exit(1)

    zero_docs = load_zero_token_docs(manifest)
    if args.doc_id:
        zero_docs = [d for d in zero_docs if d["doc_id"] in args.doc_id]
    if not zero_docs:
        print("No zero-token docs found; nothing to OCR.")
        return

    for doc in zero_docs:
        pdf_path = Path(doc["source_path"])
        doc_id = doc["doc_id"]
        lang = args.lang_en if "french" not in doc.get("doc_type", "").lower() else args.lang_fr
        out_txt = Path("deleuze_corpus/clean_text") / f"{doc_id}.txt"
        ocr_pdf(pdf_path, out_txt, lang=lang, dpi=args.dpi)
        graph_txt = Path("graphrag_project/input/data") / f"{doc_id}.txt"
        graph_txt.write_text(out_txt.read_text(encoding="utf-8"), encoding="utf-8")

    print("OCR completed. Re-run scripts/prepare_corpus.py (without --force) to refresh manifests/chunks.")


if __name__ == "__main__":
    main()
