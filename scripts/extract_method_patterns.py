#!/usr/bin/env python3
"""
Extract methodological patterns from Deleuze corpus.
Classifies chunks by Deleuzian method type (concept creation, critique, example usage, etc.)
using a combination of regex heuristics and LLM classification.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


class MethodPattern(Enum):
    """Deleuzian methodological patterns."""
    CONCEPT_CREATION = "CONCEPT_CREATION"    # Defining/creating new concepts
    CRITIQUE = "CRITIQUE"                     # Dismantling other theories
    EXAMPLE_USAGE = "EXAMPLE_USAGE"           # Concrete examples that operationalize concepts
    PROBLEM_REFRAMING = "PROBLEM_REFRAMING"   # Restating questions in Deleuzian terms
    ARGUMENTATION = "ARGUMENTATION"           # Non-dialectical argument structures
    STYLISTIC = "STYLISTIC"                   # Paradox, disjunction, immanence patterns
    NONE = "NONE"                             # Just descriptive/explanatory content


@dataclass
class PatternClassification:
    """Result of pattern classification."""
    patterns: list[str]
    confidence: float
    key_phrase: Optional[str] = None
    method: str = "heuristic"  # "heuristic" or "llm"


# Regex heuristics for fast pre-filtering
PATTERN_HEURISTICS = {
    MethodPattern.CONCEPT_CREATION: [
        r"(?:we|let us)\s+call\s+this",
        r"(?:we|I)\s+term\s+",
        r"is\s+defined\s+(?:by|as)",
        r"what\s+is\s+\w+\?\s+\w+\s+is",
        r"(?:we|let us)\s+define",
        r"this\s+(?:is|will be)\s+(?:what we call|our concept of)",
        r"the\s+concept\s+of\s+\w+\s+(?:is|means|refers)",
    ],
    MethodPattern.CRITIQUE: [
        r"the\s+problem\s+with\s+",
        r"it\s+is\s+not\s+(?:a\s+)?(?:matter|question)\s+of",
        r"(?:opposed|contrary)\s+to\s+",
        r"against\s+(?:the|this|all)",
        r"(?:hegel|freud|lacan|oedipus|psychoanalysis)\s+",
        r"we\s+(?:reject|refuse|oppose)",
        r"the\s+error\s+(?:of|is|lies)",
        r"(?:transcendence|representation)\s+(?:is|as)\s+(?:the|a)\s+trap",
    ],
    MethodPattern.EXAMPLE_USAGE: [
        r"(?:for|as\s+an?)\s+example",
        r"(?:take|consider)\s+the\s+case\s+of",
        r"(?:let'?s|let us)\s+(?:take|consider)",
        r"(?:the|a)\s+(?:wasp|orchid|wolf|rat|horse|little hans)",
        r"(?:virginia woolf|kafka|proust|artaud|bacon|beckett)",
        r"(?:this|we)\s+(?:can be\s+)?see(?:n)?\s+in\s+(?:the|a)",
    ],
    MethodPattern.PROBLEM_REFRAMING: [
        r"the\s+question\s+is\s+not\s+",
        r"not\s+['\"]?what['\"]?\s+but\s+['\"]?how['\"]?",
        r"it'?s\s+not\s+(?:about|a matter of)\s+.{1,30}\s+but\s+",
        r"the\s+(?:real|true)\s+question\s+is",
        r"(?:we|one)\s+must\s+ask\s+(?:instead|rather)",
        r"the\s+wrong\s+question",
    ],
    MethodPattern.ARGUMENTATION: [
        r"(?:disjunctive|inclusive)\s+(?:synthesis|or)",
        r"(?:and)\s+\.{3}\s+(?:and)",
        r"n(?:\s+)?-(?:\s+)?1",
        r"(?:plateau|rhizome|assemblage|machine)",
        r"(?:neither|not)\s+.{1,20}\s+nor\s+",
        r"(?:both|at once)\s+.{1,30}\s+and\s+",
        r"multiplicities",
    ],
    MethodPattern.STYLISTIC: [
        r"(?:immanence|immanent)\s+(?:plane|field|to)",
        r"(?:either|or)\s+\.{3}\s+(?:either|or)",
        r"(?:and|or)\.{3}(?:and|or)",
        r"(?:becoming|devenir)[-\s](?:\w+)",
        r"(?:de|re)territoriali[sz]",
        r"(?:smooth|striated)\s+space",
        r"(?:molar|molecular)",
        r"(?:body\s+without\s+organs|bwo)",
        r"(?:line\s+of\s+flight|ligne\s+de\s+fuite)",
    ],
}


def classify_by_heuristics(text: str) -> PatternClassification:
    """
    Fast rule-based classification using regex patterns.
    Returns patterns found with confidence based on match count.
    """
    text_lower = text.lower()
    found_patterns: list[str] = []
    key_phrases: list[str] = []
    
    for pattern_type, regexes in PATTERN_HEURISTICS.items():
        for regex in regexes:
            match = re.search(regex, text_lower, re.IGNORECASE)
            if match:
                if pattern_type.value not in found_patterns:
                    found_patterns.append(pattern_type.value)
                    key_phrases.append(match.group(0)[:50])
                break
    
    if not found_patterns:
        return PatternClassification(
            patterns=[MethodPattern.NONE.value],
            confidence=0.3,
            key_phrase=None,
            method="heuristic"
        )
    
    # Higher confidence with more pattern matches
    confidence = min(0.7, 0.5 + 0.1 * len(found_patterns))
    
    return PatternClassification(
        patterns=found_patterns,
        confidence=confidence,
        key_phrase=key_phrases[0] if key_phrases else None,
        method="heuristic"
    )


def classify_by_llm(text: str, client: "anthropic.Anthropic") -> PatternClassification:
    """
    Use Claude Haiku for semantic classification of method patterns.
    """
    prompt = f"""You are analyzing a passage from Gilles Deleuze's philosophical work.
Identify which methodological pattern(s) this passage exemplifies:

1. CONCEPT_CREATION: Deleuze introduces or defines a new philosophical concept
2. CRITIQUE: Deleuze dismantles or critiques another philosophical position
3. EXAMPLE_USAGE: A concrete example that operationalizes an abstract concept
4. PROBLEM_REFRAMING: A question is restated or reframed in Deleuzian terms
5. ARGUMENTATION: Shows how Deleuze builds non-dialectical arguments (assemblages, plateaus)
6. STYLISTIC: Exemplifies paradox, disjunction, or immanence moves
7. NONE: Just explanatory/descriptive content without clear methodological significance

Passage:
{text[:1500]}

Respond with ONLY valid JSON (no markdown): {{"patterns": ["PATTERN1", "PATTERN2"], "confidence": 0.0-1.0, "key_phrase": "brief quote showing the pattern"}}"""

    try:
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        result_text = response.content[0].text.strip()
        
        # Parse JSON response
        # Handle potential markdown code blocks
        if result_text.startswith("```"):
            result_text = re.sub(r'^```\w*\n?', '', result_text)
            result_text = re.sub(r'\n?```$', '', result_text)
        
        result = json.loads(result_text)
        
        return PatternClassification(
            patterns=result.get("patterns", [MethodPattern.NONE.value]),
            confidence=result.get("confidence", 0.5),
            key_phrase=result.get("key_phrase"),
            method="llm"
        )
    except Exception as e:
        print(f"LLM classification error: {e}", file=sys.stderr)
        return PatternClassification(
            patterns=[MethodPattern.NONE.value],
            confidence=0.0,
            key_phrase=None,
            method="llm_error"
        )


def classify_chunk(
    text: str,
    use_llm: bool = False,
    llm_client: Optional["anthropic.Anthropic"] = None,
    min_heuristic_confidence: float = 0.6
) -> PatternClassification:
    """
    Classify a text chunk by Deleuzian method pattern.
    
    Strategy:
    1. Try heuristic classification first (fast, free)
    2. If confidence too low and use_llm=True, fall back to LLM
    """
    heuristic_result = classify_by_heuristics(text)
    
    # If heuristics confident enough, use them
    if heuristic_result.confidence >= min_heuristic_confidence:
        return heuristic_result
    
    # If LLM enabled and heuristics uncertain, use LLM
    if use_llm and llm_client is not None:
        return classify_by_llm(text, llm_client)
    
    return heuristic_result


def process_corpus(
    input_path: Path,
    output_path: Path,
    use_llm: bool = False,
    sample_size: Optional[int] = None,
    min_text_length: int = 100,
    progress_interval: int = 500
) -> pd.DataFrame:
    """
    Process the entire corpus, classifying each chunk by method pattern.
    """
    print(f"Loading corpus from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} chunks")
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        print(f"Sampled {len(df)} chunks")
    
    # Filter by minimum text length
    df = df[df["text"].str.len() >= min_text_length].copy()
    print(f"After filtering: {len(df)} chunks with >= {min_text_length} chars")
    
    # Initialize LLM client if needed
    llm_client = None
    if use_llm:
        if not HAS_ANTHROPIC:
            print("Warning: anthropic package not installed. Using heuristics only.")
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                llm_client = anthropic.Anthropic(api_key=api_key)
                print("LLM classification enabled with Claude Haiku")
            else:
                print("Warning: ANTHROPIC_API_KEY not set. Using heuristics only.")
    
    # Classify each chunk
    results = []
    for i, (idx, row) in enumerate(df.iterrows()):
        if i > 0 and i % progress_interval == 0:
            print(f"Processed {i}/{len(df)} chunks...")
        
        classification = classify_chunk(
            row["text"],
            use_llm=use_llm,
            llm_client=llm_client
        )
        
        results.append({
            "id": row["id"],
            "doc_id": row["doc_id"],
            "chunk_index": row.get("chunk_index"),
            "text": row["text"],
            "language": row.get("language"),
            "patterns": classification.patterns,
            "pattern_confidence": classification.confidence,
            "key_phrase": classification.key_phrase,
            "classification_method": classification.method,
        })
    
    result_df = pd.DataFrame(results)
    
    # Save results
    print(f"Saving {len(result_df)} classified chunks to {output_path}...")
    result_df.to_parquet(output_path, index=False)
    
    # Print statistics
    print("\n=== Classification Statistics ===")
    all_patterns = [p for patterns in result_df["patterns"] for p in patterns]
    pattern_counts = pd.Series(all_patterns).value_counts()
    print(pattern_counts)
    
    return result_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Deleuzian method patterns from corpus chunks."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("deleuze_corpus/vectors.parquet"),
        help="Input parquet file with corpus chunks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("deleuze_corpus/method_patterns.parquet"),
        help="Output parquet file for classified patterns",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use Claude Haiku for uncertain classifications (costs ~$1-2 for full corpus)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only process a sample of N chunks (for testing)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=100,
        help="Minimum text length to include",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_corpus(
        input_path=args.input,
        output_path=args.output,
        use_llm=args.use_llm,
        sample_size=args.sample,
        min_text_length=args.min_length,
    )
