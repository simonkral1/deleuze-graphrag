#!/usr/bin/env python3
"""
Method Pattern Tool: Search for philosophical moves by type.

Searches the corpus filtered by Deleuzian method patterns:
- CONCEPT_CREATION: Passages where Deleuze invents/defines concepts
- CRITIQUE: Dismantling other theories (Hegel, psychoanalysis, Oedipus)
- EXAMPLE_USAGE: Concrete examples as conceptual machines (wasp-orchid, Little Hans)
- PROBLEM_REFRAMING: "Not 'what is X?' but 'how does X work?'"
- ARGUMENTATION: Disjunctive synthesis, plateau structures
- STYLISTIC: Paradox resolution, immanence vs transcendence

Uses pre-computed embeddings from vectors.parquet (1536-dim, ada-002 compatible).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import openai
import pandas as pd


# Valid method pattern types
VALID_PATTERNS = {
    "CONCEPT_CREATION",
    "CRITIQUE",
    "EXAMPLE_USAGE",
    "PROBLEM_REFRAMING",
    "ARGUMENTATION",
    "STYLISTIC",
}

# Pattern descriptions for the agent
PATTERN_DESCRIPTIONS = {
    "CONCEPT_CREATION": "Passages where Deleuze invents or defines concepts (BwO, rhizome, assemblage). Look for 'we call this', 'is defined by', introducing new terminology.",
    "CRITIQUE": "Passages dismantling other theories—Hegel, psychoanalysis, Oedipus, representational thinking. Look for 'the problem with', 'it is not a matter of', polemical tone.",
    "EXAMPLE_USAGE": "Concrete examples used as conceptual machines—wasp and orchid, Little Hans's horse, Wolf-Man, Bacon's paintings. Examples don't illustrate; they ARE concepts in action.",
    "PROBLEM_REFRAMING": "Passages that reframe questions—'the question is not X but Y', 'not what but how', shifting from essence to function.",
    "ARGUMENTATION": "Structural argumentative moves—disjunctive synthesis (AND...AND), building plateaus, connecting heterogeneous elements.",
    "STYLISTIC": "Distinctive stylistic markers—paradox resolution, immanence vs transcendence, stuttering language, dashes and ellipses.",
}


class MethodPatternTool:
    """
    Search tool for finding passages by philosophical method type.

    Combines method_patterns.parquet (classifications) with vectors.parquet (embeddings)
    to enable filtered semantic search within specific philosophical move types.
    """

    def __init__(
        self,
        patterns_path: Path | str = "deleuze_corpus/method_patterns.parquet",
        vectors_path: Path | str = "deleuze_corpus/vectors.parquet",
        openai_client: Optional[openai.Client] = None,
        embed_model: str = "text-embedding-ada-002",  # Must match vectors.parquet
    ):
        self.patterns_path = Path(patterns_path)
        self.vectors_path = Path(vectors_path)

        # Initialize OpenAI client for query embedding
        if openai_client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY required for embeddings")
            self.openai = openai.Client(api_key=api_key)
        else:
            self.openai = openai_client
        self.embed_model = embed_model

        # Load dataframes
        self.patterns_df = pd.read_parquet(self.patterns_path)
        self.vectors_df = pd.read_parquet(self.vectors_path)

        # Verify alignment
        if len(self.patterns_df) != len(self.vectors_df):
            raise ValueError(
                f"Misaligned data: patterns has {len(self.patterns_df)} rows, "
                f"vectors has {len(self.vectors_df)} rows"
            )

        # Pre-stack all vectors for fast search
        self._all_vectors = np.stack(self.vectors_df["vector"].values).astype(np.float32)
        self._all_norms = np.linalg.norm(self._all_vectors, axis=1, keepdims=True)

        # Build pattern -> indices mapping for fast filtering
        self._pattern_indices: dict[str, np.ndarray] = {}
        for pattern in VALID_PATTERNS:
            mask = self.patterns_df["patterns"].apply(
                lambda x: pattern in x if isinstance(x, (list, set)) else False
            )
            self._pattern_indices[pattern] = np.where(mask)[0]

        # Stats
        self.total_chunks = len(self.patterns_df)
        self.pattern_counts = {p: len(idx) for p, idx in self._pattern_indices.items()}

    def _embed_query(self, text: str) -> np.ndarray:
        """Get embedding for query text."""
        text = text.replace("\n", " ")
        response = self.openai.embeddings.create(input=[text], model=self.embed_model)
        return np.array(response.data[0].embedding, dtype=np.float32)

    def _cosine_similarity(self, query_vec: np.ndarray, doc_vecs: np.ndarray, doc_norms: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and documents."""
        query_norm = np.linalg.norm(query_vec)
        scores = (doc_vecs @ query_vec) / (doc_norms.flatten() * query_norm + 1e-8)
        return scores

    def search(
        self,
        query: str,
        method_types: list[str] | str | None = None,
        n_results: int = 10,
        min_confidence: float = 0.0,
    ) -> list[dict]:
        """
        Search for passages matching query, optionally filtered by method type.

        Args:
            query: Semantic search query
            method_types: One or more method types to filter by (e.g., "CRITIQUE", ["CONCEPT_CREATION", "EXAMPLE_USAGE"])
                         If None, searches all chunks.
            n_results: Number of results to return
            min_confidence: Minimum pattern confidence score (0.0-1.0)

        Returns:
            List of matching passages with metadata
        """
        # Normalize method_types
        if method_types is None:
            indices = np.arange(len(self.patterns_df))
        elif isinstance(method_types, str):
            method_types = [method_types]
            indices = self._get_indices_for_patterns(method_types)
        else:
            indices = self._get_indices_for_patterns(method_types)

        if len(indices) == 0:
            return []

        # Apply confidence filter
        if min_confidence > 0:
            confidence_mask = self.patterns_df.iloc[indices]["pattern_confidence"] >= min_confidence
            indices = indices[confidence_mask.values]

        if len(indices) == 0:
            return []

        # Get vectors for filtered indices
        filtered_vectors = self._all_vectors[indices]
        filtered_norms = self._all_norms[indices]

        # Embed query and compute similarities
        query_vec = self._embed_query(query)
        scores = self._cosine_similarity(query_vec, filtered_vectors, filtered_norms)

        # Get top-k
        if len(scores) <= n_results:
            top_local_indices = np.argsort(scores)[::-1]
        else:
            top_local_indices = np.argpartition(scores, -n_results)[-n_results:]
            top_local_indices = top_local_indices[np.argsort(scores[top_local_indices])[::-1]]

        # Build results
        results = []
        for local_idx in top_local_indices:
            global_idx = indices[local_idx]
            pattern_row = self.patterns_df.iloc[global_idx]
            vector_row = self.vectors_df.iloc[global_idx]

            results.append({
                "id": pattern_row["id"],
                "text": vector_row["text"],
                "doc_id": pattern_row["doc_id"],
                "language": pattern_row["language"],
                "patterns": pattern_row["patterns"],
                "pattern_confidence": pattern_row["pattern_confidence"],
                "key_phrase": pattern_row.get("key_phrase", ""),
                "score": float(scores[local_idx]),
                "source_path": vector_row.get("source_path", ""),
                "collection": vector_row.get("collection", ""),
            })

        return results

    def _get_indices_for_patterns(self, method_types: list[str]) -> np.ndarray:
        """Get union of indices for multiple pattern types."""
        all_indices = set()
        for mt in method_types:
            mt_upper = mt.upper()
            if mt_upper in self._pattern_indices:
                all_indices.update(self._pattern_indices[mt_upper])
        return np.array(sorted(all_indices))

    def get_exemplars(self, method_types: list[str], limit: int = 2) -> list[dict]:
        """
        Get high-confidence exemplar passages for given method types.
        Useful for injecting into system prompt as style examples.
        """
        indices = self._get_indices_for_patterns(method_types)
        if len(indices) == 0:
            return []

        # Sort by confidence
        subset = self.patterns_df.iloc[indices].copy()
        subset["_global_idx"] = indices
        subset = subset.sort_values("pattern_confidence", ascending=False).head(limit)

        results = []
        for _, row in subset.iterrows():
            global_idx = row["_global_idx"]
            vector_row = self.vectors_df.iloc[global_idx]
            results.append({
                "text": vector_row["text"][:500],  # Truncate for prompt injection
                "patterns": row["patterns"],
                "confidence": row["pattern_confidence"],
                "doc_id": row["doc_id"],
            })

        return results

    def get_pattern_stats(self) -> dict:
        """Return statistics about pattern distribution."""
        return {
            "total_chunks": self.total_chunks,
            "pattern_counts": self.pattern_counts,
            "coverage": {p: c / self.total_chunks for p, c in self.pattern_counts.items()},
        }

    def get_tool_definition(self) -> dict:
        """Return Anthropic tool definition."""
        return {
            "name": "search_method_patterns",
            "description": f"""Search for passages that exemplify specific PHILOSOPHICAL MOVES.

This tool filters the corpus by HOW Deleuze argues, not just WHAT he says.

**METHOD TYPES** (use one or more):
- CONCEPT_CREATION: {PATTERN_DESCRIPTIONS['CONCEPT_CREATION']}
- CRITIQUE: {PATTERN_DESCRIPTIONS['CRITIQUE']}
- EXAMPLE_USAGE: {PATTERN_DESCRIPTIONS['EXAMPLE_USAGE']}
- PROBLEM_REFRAMING: {PATTERN_DESCRIPTIONS['PROBLEM_REFRAMING']}
- ARGUMENTATION: {PATTERN_DESCRIPTIONS['ARGUMENTATION']}
- STYLISTIC: {PATTERN_DESCRIPTIONS['STYLISTIC']}

**WHEN TO USE**:
- Answering "What is X?" → use CONCEPT_CREATION
- Answering "What's wrong with X?" → use CRITIQUE
- Need concrete examples → use EXAMPLE_USAGE
- Showing how to reframe a problem → use PROBLEM_REFRAMING
- Building an argument → use ARGUMENTATION
- Need stylistic exemplars → use STYLISTIC

**STRATEGY**: Combine with search_corpus for precision:
1. First use this tool to find HOW Deleuze handles similar questions
2. Then use search_corpus to find specific content

Returns passages with their method classifications and confidence scores.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic search query describing the passage you want"
                    },
                    "method_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(VALID_PATTERNS)},
                        "description": "One or more method types to filter by"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results (default 5)",
                        "default": 5
                    }
                },
                "required": ["query", "method_types"]
            }
        }


def create_method_pattern_tool(openai_client: Optional[openai.Client] = None) -> MethodPatternTool:
    """Factory function to create MethodPatternTool with default paths."""
    return MethodPatternTool(
        patterns_path="deleuze_corpus/method_patterns.parquet",
        vectors_path="deleuze_corpus/vectors.parquet",
        openai_client=openai_client,
        embed_model="text-embedding-ada-002",
    )


if __name__ == "__main__":
    # Quick test
    tool = create_method_pattern_tool()
    print("Pattern stats:", tool.get_pattern_stats())

    results = tool.search(
        query="the body without organs as a practice",
        method_types=["CONCEPT_CREATION"],
        n_results=3
    )

    print("\nSearch results:")
    for r in results:
        print(f"  [{r['patterns']}] {r['text'][:200]}...")
