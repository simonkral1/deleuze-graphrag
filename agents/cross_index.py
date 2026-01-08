#!/usr/bin/env python3
"""
Cross-Index: Bidirectional links between entities, relationships, and text chunks.

Enables:
- Get quotes that support a specific entity
- Get quotes that support a specific relationship
- Annotate search results with entities mentioned
- Graph-augmented retrieval (expand results via entity connections)

Uses GraphRAG's pre-computed linkages in:
- entities.parquet (text_unit_ids column)
- relationships.parquet (text_unit_ids column)
- text_units.parquet (entity_ids, relationship_ids columns)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from difflib import SequenceMatcher

import numpy as np
import pandas as pd


class EntityQuoteCrossIndex:
    """
    Bidirectional index between entities/relationships and text chunks.

    Provides fast lookups in both directions:
    - entity -> supporting quotes
    - quote -> entities mentioned
    - relationship -> supporting quotes
    """

    def __init__(
        self,
        entities_path: Path | str = "graphrag_project/output/entities.parquet",
        relationships_path: Path | str = "graphrag_project/output/relationships.parquet",
        text_units_path: Path | str = "graphrag_project/output/text_units.parquet",
    ):
        self.entities_path = Path(entities_path)
        self.relationships_path = Path(relationships_path)
        self.text_units_path = Path(text_units_path)

        # Load dataframes
        self.entities_df = pd.read_parquet(self.entities_path)
        self.relationships_df = pd.read_parquet(self.relationships_path)
        self.text_units_df = pd.read_parquet(self.text_units_path)

        # Build indices
        self._build_indices()

        # Build fuzzy matching index for entities
        self._build_entity_name_index()

    def _build_indices(self):
        """Build bidirectional lookup indices."""
        # Entity -> chunk IDs
        self.entity_to_chunks: dict[str, list[str]] = {}
        # Chunk ID -> entity names
        self.chunk_to_entities: dict[str, list[str]] = {}
        # Relationship (source, target) -> chunk IDs
        self.relationship_to_chunks: dict[tuple[str, str], list[str]] = {}
        # Entity ID -> entity name (for lookups)
        self.entity_id_to_name: dict[str, str] = {}
        # Entity name -> entity data
        self.entity_name_to_data: dict[str, dict] = {}

        # Build entity_to_chunks and entity lookups
        for _, row in self.entities_df.iterrows():
            entity_id = row["id"]
            entity_name = row["title"]
            text_unit_ids = row.get("text_unit_ids", [])

            self.entity_id_to_name[entity_id] = entity_name
            self.entity_name_to_data[entity_name] = {
                "id": entity_id,
                "title": entity_name,
                "type": row.get("type", ""),
                "description": row.get("description", ""),
                "degree": row.get("degree", 0),
            }

            if text_unit_ids is not None:
                # Handle numpy arrays
                if hasattr(text_unit_ids, "tolist"):
                    text_unit_ids = text_unit_ids.tolist()
                self.entity_to_chunks[entity_name] = list(text_unit_ids)

        # Build chunk_to_entities from text_units
        for _, row in self.text_units_df.iterrows():
            chunk_id = row["id"]
            entity_ids = row.get("entity_ids", [])

            if entity_ids is not None:
                if hasattr(entity_ids, "tolist"):
                    entity_ids = entity_ids.tolist()

                entity_names = []
                for eid in entity_ids:
                    if eid in self.entity_id_to_name:
                        entity_names.append(self.entity_id_to_name[eid])

                if entity_names:
                    self.chunk_to_entities[chunk_id] = entity_names

        # Build relationship_to_chunks
        for _, row in self.relationships_df.iterrows():
            source = row["source"]
            target = row["target"]
            text_unit_ids = row.get("text_unit_ids", [])

            if text_unit_ids is not None:
                if hasattr(text_unit_ids, "tolist"):
                    text_unit_ids = text_unit_ids.tolist()
                # Handle string representation of list
                if isinstance(text_unit_ids, str):
                    text_unit_ids = [text_unit_ids]
                self.relationship_to_chunks[(source, target)] = list(text_unit_ids)

        # Build chunk_id -> text lookup
        self.chunk_texts: dict[str, str] = dict(
            zip(self.text_units_df["id"], self.text_units_df["text"])
        )

    def _build_entity_name_index(self):
        """Build index for fuzzy entity name matching."""
        self.entity_names = list(self.entity_name_to_data.keys())
        self.entity_names_lower = {name.lower(): name for name in self.entity_names}

        # Build word-based index for faster fuzzy matching
        self.entity_words: dict[str, set[str]] = {}
        for name in self.entity_names:
            words = set(name.lower().split())
            for word in words:
                if word not in self.entity_words:
                    self.entity_words[word] = set()
                self.entity_words[word].add(name)

    def fuzzy_match_entity(self, query: str, threshold: float = 0.6) -> Optional[str]:
        """
        Find the best matching entity name for a query string.

        Uses multiple strategies:
        1. Exact match (case-insensitive)
        2. Substring match
        3. Word overlap
        4. Sequence matching (fuzzy)

        Returns the canonical entity name or None if no match found.
        """
        query_lower = query.lower().strip()

        # 1. Exact match (case-insensitive)
        if query_lower in self.entity_names_lower:
            return self.entity_names_lower[query_lower]

        # 2. Check if query is substring of any entity or vice versa
        for name in self.entity_names:
            name_lower = name.lower()
            if query_lower in name_lower or name_lower in query_lower:
                return name

        # 3. Word overlap scoring
        query_words = set(query_lower.split())
        candidates = set()
        for word in query_words:
            if word in self.entity_words:
                candidates.update(self.entity_words[word])

        if candidates:
            # Score by word overlap
            best_score = 0
            best_match = None
            for candidate in candidates:
                candidate_words = set(candidate.lower().split())
                overlap = len(query_words & candidate_words)
                score = overlap / max(len(query_words), len(candidate_words))
                if score > best_score:
                    best_score = score
                    best_match = candidate

            if best_score >= threshold:
                return best_match

        # 4. Sequence matching (slower, last resort)
        best_ratio = 0
        best_match = None
        for name in self.entity_names:
            ratio = SequenceMatcher(None, query_lower, name.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = name

        if best_ratio >= threshold:
            return best_match

        return None

    def get_quotes_for_entity(
        self,
        entity_name: str,
        top_k: int = 5,
        fuzzy: bool = True
    ) -> list[dict]:
        """
        Get text chunks where this entity is discussed.

        Args:
            entity_name: Name of the entity (fuzzy matched if not exact)
            top_k: Maximum number of quotes to return
            fuzzy: Whether to use fuzzy matching

        Returns:
            List of quote dicts with text and metadata
        """
        # Resolve entity name
        if fuzzy and entity_name not in self.entity_to_chunks:
            matched = self.fuzzy_match_entity(entity_name)
            if matched:
                entity_name = matched
            else:
                return []

        chunk_ids = self.entity_to_chunks.get(entity_name, [])
        if not chunk_ids:
            return []

        # Get texts for chunks
        results = []
        for chunk_id in chunk_ids[:top_k]:
            text = self.chunk_texts.get(chunk_id, "")
            if text:
                results.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "entity": entity_name,
                    "other_entities": [e for e in self.chunk_to_entities.get(chunk_id, []) if e != entity_name],
                })

        return results

    def get_quotes_for_relationship(
        self,
        source: str,
        target: str,
        fuzzy: bool = True
    ) -> list[dict]:
        """
        Get text chunks that support a relationship between two entities.

        Args:
            source: Source entity name
            target: Target entity name
            fuzzy: Whether to use fuzzy matching

        Returns:
            List of quote dicts with text and metadata
        """
        # Resolve entity names
        if fuzzy:
            source_match = self.fuzzy_match_entity(source)
            target_match = self.fuzzy_match_entity(target)
            if source_match:
                source = source_match
            if target_match:
                target = target_match

        # Try both directions
        chunk_ids = self.relationship_to_chunks.get((source, target), [])
        if not chunk_ids:
            chunk_ids = self.relationship_to_chunks.get((target, source), [])

        if not chunk_ids:
            return []

        # Get relationship description
        rel_desc = ""
        for _, row in self.relationships_df.iterrows():
            if (row["source"] == source and row["target"] == target) or \
               (row["source"] == target and row["target"] == source):
                rel_desc = row.get("description", "")
                break

        results = []
        for chunk_id in chunk_ids:
            text = self.chunk_texts.get(chunk_id, "")
            if text:
                results.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "source": source,
                    "target": target,
                    "relationship_description": rel_desc,
                })

        return results

    def get_entities_in_chunk(self, chunk_id: str) -> list[dict]:
        """
        Get all entities mentioned in a specific chunk.

        Returns:
            List of entity dicts with name, type, description
        """
        entity_names = self.chunk_to_entities.get(chunk_id, [])
        return [self.entity_name_to_data.get(name, {"title": name}) for name in entity_names]

    def annotate_search_results(self, results: list[dict]) -> list[dict]:
        """
        Annotate search results with entity information.

        Adds 'entities_mentioned' field to each result.
        """
        for result in results:
            chunk_id = result.get("id", result.get("chunk_id", ""))
            if chunk_id:
                entities = self.chunk_to_entities.get(chunk_id, [])
                result["entities_mentioned"] = entities
        return results

    def get_related_quotes(
        self,
        entity_name: str,
        relationship_index,  # GraphTraversalIndex
        expand_k: int = 2,
        quotes_per_entity: int = 2
    ) -> list[dict]:
        """
        Get quotes for entities related to the given entity.

        Uses graph traversal to find connected entities, then retrieves
        supporting quotes for each.

        Args:
            entity_name: Starting entity
            relationship_index: GraphTraversalIndex for traversal
            expand_k: Number of related entities to expand to
            quotes_per_entity: Quotes per related entity

        Returns:
            List of quotes from related entities
        """
        # Fuzzy match entity
        matched = self.fuzzy_match_entity(entity_name)
        if matched:
            entity_name = matched

        # Get related entities via graph
        related = relationship_index.get_connections(entity_name, top_k=expand_k)

        results = []
        for rel in related:
            related_entity = rel.get("concept", "")
            if related_entity:
                quotes = self.get_quotes_for_entity(related_entity, top_k=quotes_per_entity)
                for q in quotes:
                    q["relationship_from"] = entity_name
                    q["relationship_type"] = rel.get("description", "")
                results.extend(quotes)

        return results

    def get_entity_info(self, entity_name: str, fuzzy: bool = True) -> Optional[dict]:
        """Get full info for an entity by name."""
        if fuzzy and entity_name not in self.entity_name_to_data:
            matched = self.fuzzy_match_entity(entity_name)
            if matched:
                entity_name = matched
        return self.entity_name_to_data.get(entity_name)

    def search_entities(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Simple text search over entity names and descriptions.
        Returns matching entities.
        """
        query_lower = query.lower()
        scored = []

        for name, data in self.entity_name_to_data.items():
            score = 0
            # Name match
            if query_lower in name.lower():
                score += 2
            # Description match
            desc = data.get("description", "").lower()
            if query_lower in desc:
                score += 1

            if score > 0:
                scored.append((score, data))

        scored.sort(key=lambda x: (-x[0], -x[1].get("degree", 0)))
        return [item[1] for item in scored[:top_k]]

    def get_tool_definition(self) -> dict:
        """Return Anthropic tool definition for supporting quotes lookup."""
        return {
            "name": "get_supporting_quotes",
            "description": """Get exact textual evidence for a concept or relationship.

**USE AFTER** search_concepts or traverse_relationships to find the actual passages.

This tool bridges the graph (abstract structure) to the corpus (concrete text).

**STRATEGIES**:
1. Found an interesting entity via search_concepts? Get its supporting quotes.
2. Found a relationship via traverse_relationships? Get the passages that establish it.
3. Want to verify a claim about a concept? Get the primary sources.

**EXAMPLES**:
- "Get quotes about the Body without Organs" → entity="body without organs"
- "Show passages connecting rhizome and assemblage" → source="rhizome", target="assemblage"

Returns text chunks with:
- The actual passage
- Other entities mentioned in the same passage (co-occurrence)
- Relationship context if applicable

**NOTE**: Uses fuzzy matching—don't worry about exact capitalization.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Entity name to get quotes for (fuzzy matched)"
                    },
                    "source": {
                        "type": "string",
                        "description": "Source entity for relationship lookup"
                    },
                    "target": {
                        "type": "string",
                        "description": "Target entity for relationship lookup"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum quotes to return (default 5)",
                        "default": 5
                    }
                },
                "required": []
            }
        }

    def get_stats(self) -> dict:
        """Return statistics about the cross-index."""
        return {
            "total_entities": len(self.entity_name_to_data),
            "total_relationships": len(self.relationship_to_chunks),
            "total_text_units": len(self.text_units_df),
            "entities_with_quotes": len([e for e in self.entity_to_chunks if self.entity_to_chunks[e]]),
            "chunks_with_entities": len(self.chunk_to_entities),
        }


def create_cross_index() -> EntityQuoteCrossIndex:
    """Factory function to create EntityQuoteCrossIndex with default paths."""
    return EntityQuoteCrossIndex(
        entities_path="graphrag_project/output/entities.parquet",
        relationships_path="graphrag_project/output/relationships.parquet",
        text_units_path="graphrag_project/output/text_units.parquet",
    )


if __name__ == "__main__":
    # Quick test
    index = create_cross_index()
    print("Cross-index stats:", index.get_stats())

    # Test fuzzy matching
    test_queries = ["body without organs", "BWO", "rhizome", "war machine", "assemblage"]
    print("\nFuzzy matching tests:")
    for q in test_queries:
        match = index.fuzzy_match_entity(q)
        print(f"  '{q}' -> '{match}'")

    # Test quote lookup
    print("\nQuotes for 'rhizome':")
    quotes = index.get_quotes_for_entity("rhizome", top_k=2)
    for q in quotes:
        print(f"  {q['text'][:200]}...")
