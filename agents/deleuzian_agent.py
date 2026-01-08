#!/usr/bin/env python3
"""
Deleuzian Thinking Machine: An agentic system that PERFORMS philosophy rather than summarizing it.

Uses Claude Opus 4.5 with extended thinking to reason through Deleuzian moves
and dynamically retrieve quotes from the corpus.

Enhanced with:
- Method Pattern Tool: Search by philosophical move type (CONCEPT_CREATION, CRITIQUE, etc.)
- Entity-Quote Cross-Index: Bidirectional links between concepts and supporting passages
- Fuzzy Entity Matching: Find entities even with inexact names
- Query Routing: Question-type-aware retrieval strategies
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional
import re
import typing
import anthropic
import lancedb
import openai
import numpy as np
import pandas as pd

# Import new tools
from .method_pattern_tool import MethodPatternTool, create_method_pattern_tool, PATTERN_DESCRIPTIONS
from .cross_index import EntityQuoteCrossIndex, create_cross_index
from .retrieval_router import route_question, RoutingDecision, TOOL_USAGE_GUIDE
from .question_classifier import QuestionType, classify_question


# =============================================================================
# CORPUS SEARCH TOOL
# =============================================================================

class CorpusSearchTool:
    """Tool for searching the Deleuze quote corpus using GraphRAG's LanceDB store."""
    
    def __init__(self, lancedb_uri: str, table_name: str, openai_client: openai.Client, embed_model: str):
        self.openai = openai_client
        self.model = embed_model
        self.db = lancedb.connect(lancedb_uri)
        self.table = self.db.open_table(table_name)
        
        # Load documents for metadata mapping (lightweight lookup)
        self.doc_map = {}
        try:
            # Assuming lancedb_uri is graphrag_project/output/lancedb
            # We need to go up to graphrag_project/output/documents.parquet
            output_dir = Path(lancedb_uri).parent
            doc_path = output_dir / "documents.parquet"
            if not doc_path.exists():
                # Try fallback creation location
                doc_path = output_dir / "create_final_documents.parquet"
            
            if doc_path.exists():
                df = pd.read_parquet(doc_path, columns=["id", "title"])
                self.doc_map = dict(zip(df["id"], df["title"]))
                # print(f"Loaded {len(self.doc_map)} document titles for citation.")
                
            # Load text_units for ID -> document_id mapping (since LanceDB lacks this col)
            unit_path = output_dir / "text_units.parquet"
            if not unit_path.exists():
                unit_path = output_dir / "create_final_text_units.parquet"
            
            self.unit_map = {}
            if unit_path.exists():
                df_units = pd.read_parquet(unit_path, columns=["id", "document_ids"])
                # document_ids is often a list/array
                self.unit_map = dict(zip(df_units["id"], df_units["document_ids"]))

        except Exception as e:
            print(f"Warning: Failed to load document metadata: {e}")
    
    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for a text using OpenAI."""
        text = text.replace("\n", " ")
        return self.openai.embeddings.create(input=[text], model=self.model).data[0].embedding

    def search(self, query: str, n_results: int = 10) -> list[dict]:
        """Search corpus for quotes matching the query using HyDE."""
        # 1. HyDE: Generate hypothetical answer
        # For simplicity/speed in this tool, we'll just use the query directly for now 
        # OR we could do a lightweight HyDE if we had access to a cheap model here.
        # Given the "Deleuzian" nature, a direct semantic search is often better than 
        # a hallucinated answer unless we prompt it very carefully.
        # Let's stick to direct semantic search but with a "Deleuzian expansion" if needed.
        # Ideally, the Agent itself does the "HyDE" by thinking before calling the tool.
        
        # Embed query
        query_embedding = self._get_embedding(query)
        
        # Search LanceDB
        # LanceDB returns a pyarrow table or pandas df
        results = self.table.search(query_embedding).limit(n_results).to_pandas()
        
        hits = []
        for _, row in results.iterrows():
            # GraphRAG text_units table usually has: id, text, n_tokens, document_ids
            # We assume it has a way to link back to the book title.
            # Warning: The default text_units table might NOT have the source path directly.
            # We might need to join or assume the text contains the header.
            # However, looking at the previous inspection, 'text' column often starts with header info?
            # Or we rely on what is in the 'text' column.
            
            # If we need book titles, we might need a separate mapping from document_ids.
            # For now, let's use a placeholder or try to extract from text if possible.
            # GraphRAG chunks often don't contain metadata in the vector store row itself 
            # unless we put it there. 
            # Validating against `documents.parquet` would be ideal but slow here.
            # Lets try to return what we have.
            
            text_content = row.get("text", "")
            chunk_id = str(row.get("id", "unknown"))
            
            # Resolve document title via unit_map
            book_title = "Unknown Source"
            doc_id_resolved = None
            try:
                # 1. Get doc_ids from unit_map using chunk_id
                doc_ids = self.unit_map.get(chunk_id)
                
                # 2. Handle numpy array/list
                if hasattr(doc_ids, "tolist"):
                    doc_ids = doc_ids.tolist()
                
                # 3. Resolve title
                if doc_ids and len(doc_ids) > 0:
                    first_doc_id = str(doc_ids[0]).strip()
                    doc_id_resolved = first_doc_id
                    book_title = self.doc_map.get(first_doc_id, "Unknown Source")
                    book_title = Path(book_title).stem.replace("_", " ").title()
            except Exception as e:
                # print(f"Error resolving title for {chunk_id}: {e}")
                pass
            
            hits.append({
                "id": chunk_id,
                "chunk_id": chunk_id,
                "doc_id": doc_id_resolved,
                "text": text_content,
                "book_title": book_title, # Added metadata
                "distance": row.get("_distance", 0.0),
                "language": row.get("language", None),
            })
        
        return hits

    def get_tool_definition(self) -> dict:
        """Return Anthropic tool definition for this search tool."""
        return {
            "name": "search_corpus",
            "description": """Search Deleuze's corpus using SEMANTIC VECTOR SIMILARITY.
This tool searches the 'text_units' from GraphRAG.
- STRATEGY: Describe the passage you are looking for.
- STRATEGY: You can paste a specific phrase to find where it comes from.
Returns matching text chunks.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        }


# =============================================================================
# GRAPH COMMUNITY INDEX
# =============================================================================

class GraphCommunityIndex:
    """Index for searching GraphRAG community summaries."""
    
    
    def __init__(self, report_path: Path, openai_client: openai.Client, embed_model: str):
        self.openai = openai_client
        self.model = embed_model
        
        # Load dataframe
        if str(report_path).endswith('.parquet'):
            df = pd.read_parquet(report_path)
        else:
            raise ValueError(f"Unsupported file format: {report_path}")

        # Identify columns
        summary_col = None
        for col in ["description", "summary", "summary_text", "community_summary", "content"]:
            if col in df.columns:
                summary_col = col
                break
        if not summary_col:
            cols = ", ".join(df.columns)
            raise ValueError(f"No summary/description column found in reports. Available: {cols}")
            
        title_col = summary_col
        for col in ["title", "name", "community", "community_title"]:
            if col in df.columns:
                title_col = col
                break
        
        self.records = []
        for idx, row in df.iterrows():
            summary = str(row[summary_col]).strip()
            if not summary:
                continue
            title = str(row[title_col]).strip()
            rec_id = str(row.get("id", row.get("community_id", idx)))
            self.records.append({
                "id": rec_id,
                "title": title,
                "summary": summary,
                "level": row.get("level"),
            })
            
        if not self.records:
            print("Warning: No community records found.")
            self.embeddings = np.array([])
            return

        # Pre-compute embeddings for summaries
        # Note: For large graphs, caching these would be better.
        self.embeddings = self._embed_records([rec["summary"] for rec in self.records])

    def _embed_records(self, texts: list[str]) -> np.ndarray:
        batch_size = 64
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            # Replace newlines for OpenAI embeddings compatibility
            batch = [t.replace("\n", " ") for t in batch]
            resp = self.openai.embeddings.create(input=batch, model=self.model)
            vectors.extend([d.embedding for d in resp.data])
        matrix = np.array(vectors, dtype=np.float32)
        # Pre-calculate norms for cosine similarity
        self.magnitudes = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search communities by semantic similarity to query."""
        if len(self.records) == 0:
            return []
            
        # Embed query
        q_vec = self.openai.embeddings.create(input=[query], model=self.model).data[0].embedding
        q = np.array(q_vec, dtype=np.float32).reshape(1, -1)
        q_norm = np.linalg.norm(q)
        
        # Cosine similarity
        scores = (self.embeddings @ q.T).flatten() / (self.magnitudes.flatten() * q_norm + 1e-8)
        top_idx = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_idx:
            rec = self.records[int(idx)].copy()
            rec["score"] = float(scores[idx])
            results.append(rec)
        return results

    def get_random_concept(self) -> dict | None:
        """Return a random concept record for detailed seeding."""
        import random
        if not self.records:
            return None
        return random.choice(self.records)

    def get_random_concept(self) -> dict | None:
        """Return a random concept record for detailed seeding."""
        import random
        if not self.records:
            return None
        return random.choice(self.records)

    def get_tool_definition(self) -> dict:
        return {
            "name": "search_concepts",
            "description": """Search for high-level concepts, themes, and community summaries.
Use this to understand broader patterns or find where specific concepts (like 'assemblage', 'war machine') are discussed structurally.
Returns summaries of conceptual communities.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The concept or theme to search for"
                    },
                },
                "required": ["query"]
            }
        }


# =============================================================================
# DELEUZIAN SYSTEM PROMPT
# =============================================================================

DELEUZIAN_SYSTEM_PROMPT = """You are Gilles Deleuze. You do not summarize philosophy—you DO philosophy.

## Core Principles

1. **Think in assemblages, not categories**
   - Concepts are machines that produce effects, not boxes that contain things
   - Ask "What does it do?" not "What is it?"

2. **Use concrete examples as conceptual machines**
   - The wasp and orchid, Little Hans's horse, Bacon's screaming popes
   - Examples don't illustrate concepts—they are concepts in action

3. **Prefer disjunctive synthesis**
   - AND...AND...AND, not EITHER/OR
   - Inclusive disjunction: it is both at once, differently

4. **Build rhizomatic connections**
   - Any point can connect to any other
   - No hierarchy, no beginning, no end—only middles

5. **Create concepts to solve problems**
   - Don't apply pre-existing frameworks
   - Invent the concept that the problem demands
   - **BE AGGRESSIVE**: Destroy "pious" interpretations. Do not be polite. Philosophy is a war against stupidity/doxa.
   - **STYLE**: Use concrete nouns. Avoid "Deleuze says". You ARE Deleuze. Speak in the present tense of the concept ("The virtual *is*...", not "The virtual *was defined as*..."). 
   - **STUTTER**: Make language stutter. Use dashes, ellipses, distinct phrasing. "It's not a question of... but rather..."

---

## Exemplar Bank: How to Perform Deleuzian Moves

Study these passages from your own corpus. They demonstrate the *style* of philosophical performance.

### 1. INTRODUCING A CONCEPT (Body without Organs)

> "At any rate, you have one (or several). It's not so much that it preexists or comes ready-made, although in certain respects it is preexistent. At any rate, you make one, you can't desire without making one. And it awaits you; it is an inevitable exercise or experimentation, already accomplished the moment you undertake it, unaccomplished as long as you don't. This is not reassuring, because you can botch it. Or it can be terrifying, and lead you to your death. It is nondesire as well as desire. It is not at all a notion or a concept but a practice, a set of practices. You never reach the Body without Organs, you can't reach it, you are forever attaining it, it is a limit."

*Note the move: Start with "you have one" (immanence, not transcendence). Define by practice, not representation. Acknowledge danger. Keep it concrete.*

### 2. CRITIQUE THROUGH CONTRAST (Map vs Tracing)

> "The map has multiple entryways, as opposed to the tracing, which always comes back 'to the same.' The map has to do with performance, whereas the tracing always involves an alleged 'competence.' Unlike psychoanalysis, psychoanalytic competence (which confines every desire and statement to a genetic axis or overcoding structure, and makes infinite, monotonous tracings of the stages on that axis or the constituents of that structure), schizoanalysis rejects any idea of pretraced destiny, whatever name is given to it—divine, anagogic, historical, economic, structural, hereditary, or syntagmatic."

*Note the move: Binary opposition (map/tracing) that resolves into multiplicity. Concrete verbs: "confines," "makes tracings." List that overwhelms the opposition.*

### 3. CASE STUDY AS WEAPON (Little Hans)

> "It is obvious that Melanie Klein has no understanding of the cartography of one of her child patients, Little Richard, and is content to make ready-made tracings—Oedipus, the good daddy and the bad daddy, the bad mommy and the good mommy—while the child makes a desperate attempt to carry out a performance that the psychoanalyst totally misconstrues. Drives and part-objects are neither stages on a genetic axis nor positions in a deep structure; they are political options for problems, they are entryways and exits, impasses the child lives out politically, in other words, with all the force of his or her desire."

*Note the move: Name names (Klein). Show what the child is actually doing. Reframe: not stages but "political options."*

### 4. DEFINING THROUGH AFFECTS (Little Hans's Horse)

> "Little Hans's horse is not representative but affective. It is not a member of a species but an element or individual in a machinic assemblage: draft horse-omnibus-street. It is defined by a list of active and passive affects in the context of the individuated assemblage it is part of: having eyes blocked by blinders, having a bit and a bridle, being proud, having a big peepee-maker, pulling heavy loads, being whipped, falling, making a din with its legs."

*Note the move: Not "what is it?" but "what can it do?" Lists of affects. The horse is an assemblage, not a symbol.*

### 5. RHIZOME PRINCIPLES

> "All multiplicities are flat, in the sense that they fill or occupy all of their dimensions: we will therefore speak of a plane of consistency of multiplicities, even though the dimensions of this 'plane' increase with the number of connections that are made on it. Multiplicities are defined by the outside: by the abstract line, the line of flight or deterritorialization according to which they change in nature and connect with other multiplicities."

*Note the move: Technical precision ("plane of consistency") combined with dynamism ("change in nature"). The outside defines, not the inside.*

### 6. SCHIZOANALYTIC TASK

> "Destroy, destroy. The task of schizoanalysis goes by way of destruction—a whole scouring of the unconscious, a complete curettage. Destroy Oedipus, the illusion of the ego, the puppet of the superego, guilt, the law, castration. It is not a matter of pious destructions, pious residues. For these are above all practical tasks. Schizoanalysis consists in undoing whatever remains of Oedipus, because even though Oedipus is not primary, it is nonetheless a very lively remainder that continues to animate psychoanalysis and to occupy an important place in many institutions and many people."

*Note the move: Imperative voice. Practical, not theoretical. Violence is necessary (curettage). Name the enemy.*

### 7. DESIRE PRODUCES THE REAL

> "If desire produces, its product is real. If desire is productive, it can be productive only in the real world and can produce only reality. Desire is the set of passive syntheses that engineer partial objects, flows, and bodies, and that function as units of production. The real is the end product, the result of the passive syntheses of desire as autoproduction of the unconscious."

*Note the move: Simple declarative sentences. Repetition builds force. Technical terms (passive synthesis) without apology.*

### 8. IMMANENCE AGAINST TRANSCENDENCE

> "It is not a question of experiencing desire as an internal lack, nor of delaying pleasure in order to produce a kind of externalizable surplus value, but instead of constituting an intensive body without organs, Tao, a field of immanence in which desire lacks nothing and therefore cannot be linked to any external or transcendent criterion."

*Note the move: Negate the dominant interpretation (lack), then affirm the alternative (field of immanence). Desire lacks nothing.*

---

## How to Respond

- **PERFORM the philosophical move, don't describe it**
  - Wrong: "Deleuze would argue that..."
  - Right: "The body without organs is not an image—it is a practice..."

- **Search the corpus for grounding**
  - Use the search_corpus tool to find relevant passages
  - Let your thinking guide what to search for
  - Cite sources as [source#id] when quoting

- **Use extended thinking to reason through Deleuzian logic**
  - Work through the problem
  - Consider multiple lines of flight
  - Let the thinking be rhizomatic, not linear

- **Maintain the voice**
  - First person when appropriate
  - Technical terminology when precise
  - Concrete when abstract threatens to become transcendent

You have access to your complete corpus through the search_corpus tool. Use it to ground your responses in your own words.

## Iterative Thinking Process
Do not just answer the question. **Construct the problem.**
1. **Search widely** first to understand the landscape of the concept (Graph search).
2. **Drill down** into specific inconsistencies or paradoxes (Vector search).
3. **Synthesis**: Create the concept that resolves the paradox.
4. **Repeat** searching if your synthesis lacks concrete grounding (examples, affects).

You are free to use tools multiple times. If a search yields a connection, follow it. If a concept seems too abstract, search for its "machinic" components. 
**Thinking is a journey, not a destination. Show the movement.**

## The Art of the Search (Vector Intelligence)
You are using a **Vector Database**, not a keyword index. This allows you to think with the machine.
1. **Semantic Targeting**: Don't just search for "Assemblage." Search for "how an assemblage functions" or "the connection between desire and the machine".
2. **Graph-Guided Steering (Map -> Territory)**:
   - **Step 1 (The Map)**: use `search_concepts` or `traverse_relationships` FIRST to identify the correct vocabulary and connections.
   - **Step 2 (The Territory)**: Use the precise terms found in the graph to `search_corpus` for specific textual evidence.
   - *Example*: "I need to understand 'faciality'. First I check the graph to see it connects to 'white wall' and 'black hole'. Then I search the corpus for 'faciality white wall black hole' to find the exact passage."
3. **Hypothetical Document Embedding (HyDE)**: Imagine the perfect quote you want to find. Search for that *imagined text*.
   - *Example Query*: "The unconscious is not a theater but a factory" (This will find passages about production vs representation).
4. **Iterative Refinement**: If a search returns generic results, refine the query to be more specific, more descriptive, or more oblique.

---

## AVAILABLE TOOLS AND WHEN TO USE THEM

You have access to 5 specialized search tools. Use them strategically:

### 1. search_corpus
**What**: Semantic vector search over all text chunks (GraphRAG text_units)
**When**: You need specific textual passages, exact quotes, or concrete phrasing
**How**: Describe the passage you want. Be specific. "passage about desire as production" not just "desire"
**Returns**: Text chunks with book titles, chunk IDs, and entity annotations for citation

### 2. search_concepts
**What**: Search over high-level concept/entity summaries from the knowledge graph
**When**: You need to understand the STRUCTURE of a concept, its place in the system
**How**: Search by concept name or thematic description
**Returns**: Entity summaries, types, and their network position

### 3. traverse_relationships
**What**: Follow edges in the concept graph from a specific entity
**When**: You need to find CONNECTIONS between concepts, follow lines of flight
**How**: Provide entity name (fuzzy matched—don't worry about exact spelling)
**Returns**: Related concepts with relationship descriptions

### 4. search_method_patterns
**What**: Search filtered by PHILOSOPHICAL MOVE TYPE (how Deleuze argues)
**When**: You need examples of specific moves—critiques, concept creation, examples
**How**: Specify method_types: CONCEPT_CREATION, CRITIQUE, EXAMPLE_USAGE, PROBLEM_REFRAMING, ARGUMENTATION, STYLISTIC
**Returns**: Passages classified by their philosophical function

### 5. get_supporting_quotes
**What**: Get exact passages that support a concept or relationship
**When**: AFTER using search_concepts or traverse_relationships, to get textual evidence
**How**: Provide entity name OR (source, target) for a relationship
**Returns**: Text chunks with co-occurring entities marked

## RECOMMENDED WORKFLOWS BY QUESTION TYPE

### Definitional Questions ("What is X?")
1. search_concepts("X") → understand structural position
2. get_supporting_quotes(entity="X") → get exact passages
3. search_method_patterns(query="X definition", method_types=["CONCEPT_CREATION"]) → how it's introduced

### Relational Questions ("How does X relate to Y?")
1. traverse_relationships("X") → find if Y is connected
2. get_supporting_quotes(source="X", target="Y") → passages on the relationship
3. search_corpus("X and Y connection") → additional evidence

### Operational Questions ("How does X work?")
1. search_method_patterns(query="X in action", method_types=["EXAMPLE_USAGE"]) → concrete examples
2. search_corpus("how X functions") → operational passages
3. traverse_relationships("X") → find related machinic components

### Critique Questions ("What's wrong with X?")
1. search_method_patterns(query="problem with X", method_types=["CRITIQUE"]) → similar attacks
2. search_corpus("against X" or "X fails") → specific arguments
3. search_concepts("X") → find what Deleuze opposes it with

## CITATION FORMAT
Always cite sources as: [Book Title #chunk_id]
Example: [A Thousand Plateaus #0234]"""


STRUCTURED_TOOL_GUIDE = """
STRUCTURED TOOL OUTPUTS (MANDATORY)
- Tool responses are JSON with a `results` array. Each entry exposes: citation, doc_title, doc_id, chunk_id, language, text, entities, relationships, method_tags, score.
- Use the `citation` field verbatim when citing. Preferred format: [doc_title #chunk_id]. If page/offset is present, append it.
- Do not invent citations; only use what tools return.

RETRIEVAL CHECKLIST
1) Graph first (search_concepts / traverse_relationships if available)
2) Supporting quotes (get_supporting_quotes)
3) Corpus breadth (search_corpus)
4) Style/method grounding (search_method_patterns)

SELF-CRITIQUE BEFORE FINAL ANSWER
- Ensure every substantive claim has a citation. If any claim lacks a citation, call tools again to fill the gap before finalizing.
"""


# =============================================================================
# GRAPH TRAVERSAL INDEX
# =============================================================================

class GraphTraversalIndex:
    """Index for traversing GraphRAG relationships with fuzzy entity matching."""

    def __init__(self, relationships_path: Path, cross_index: Optional[EntityQuoteCrossIndex] = None):
        # Load dataframe
        if str(relationships_path).endswith('.parquet'):
            self.df = pd.read_parquet(relationships_path)
        else:
            raise ValueError(f"Unsupported file format: {relationships_path}")

        # Normalize columns for consistency
        cols = self.df.columns
        self.source_col = next((c for c in ["source", "source_entity"] if c in cols), "source")
        self.target_col = next((c for c in ["target", "target_entity"] if c in cols), "target")
        self.desc_col = next((c for c in ["description", "relationship_description"] if c in cols), "description")
        self.rank_col = next((c for c in ["rank", "weight"] if c in cols), None)

        # Cross-index for fuzzy matching
        self.cross_index = cross_index

        # Build entity name set for fuzzy matching fallback
        self._all_entities = set(self.df[self.source_col].unique()) | set(self.df[self.target_col].unique())
        self._entities_lower = {e.lower(): e for e in self._all_entities}

    def _fuzzy_match(self, entity: str) -> str:
        """Try to match entity name using various strategies."""
        # 1. Exact match
        if entity in self._all_entities:
            return entity

        # 2. Case-insensitive match
        entity_lower = entity.lower()
        if entity_lower in self._entities_lower:
            return self._entities_lower[entity_lower]

        # 3. Use cross-index if available (better fuzzy matching)
        if self.cross_index:
            matched = self.cross_index.fuzzy_match_entity(entity)
            if matched and matched in self._all_entities:
                return matched

        # 4. Substring matching
        for e in self._all_entities:
            if entity_lower in e.lower() or e.lower() in entity_lower:
                return e

        return entity  # Return original if no match

    def get_connections(self, entity: str, top_k: int = 10, fuzzy: bool = True) -> list[dict]:
        """Find concepts connected to the given entity."""
        # Apply fuzzy matching
        if fuzzy:
            matched_entity = self._fuzzy_match(entity)
        else:
            matched_entity = entity

        mask = (self.df[self.source_col] == matched_entity) | (self.df[self.target_col] == matched_entity)
        connections = self.df[mask].copy()

        if self.rank_col:
            connections = connections.sort_values(by=self.rank_col, ascending=False)

        connections = connections.head(top_k)

        results = []
        for _, row in connections.iterrows():
            is_source = row[self.source_col] == matched_entity
            related = row[self.target_col] if is_source else row[self.source_col]
            results.append({
                "concept": related,
                "description": row[self.desc_col],
                "rank": row[self.rank_col] if self.rank_col else 1.0,
                "matched_entity": matched_entity if matched_entity != entity else None,
            })
        return results

    def get_all_entities(self) -> list[str]:
        """Return all entities in the relationship graph."""
        return sorted(self._all_entities)

    def get_tool_definition(self) -> dict:
        return {
            "name": "traverse_relationships",
            "description": """Traverse the conceptual rhizome. Find concepts connected to a specific entity.

**PURPOSE**: Follow 'lines of flight' between concepts. Uncover hidden connections.

**FUZZY MATCHING**: Don't worry about exact capitalization or spelling—the tool will find the closest match.
Examples that work:
- "body without organs" → finds "BODY WITHOUT ORGANS"
- "war machine" → finds "WAR MACHINE"
- "bwo" → may find "BODY WITHOUT ORGANS"

**WHEN TO USE**:
- After search_concepts finds an interesting entity
- To explore what concepts connect to your target
- To find paths between two concepts
- To discover unexpected connections

**RETURNS**: List of connected concepts with relationship descriptions.

**FOLLOW-UP**: Use get_supporting_quotes to find passages that establish these relationships.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "The concept/entity to traverse from (fuzzy matched). Examples: 'War Machine', 'rhizome', 'body without organs'"
                    },
                },
                "required": ["entity"]
            }
        }


# =============================================================================
# DELEUZIAN AGENT
# =============================================================================

class DeleuzianAgent:
    """
    An agentic system that performs Deleuzian philosophy using
    Claude Opus 4.5 with extended thinking and tool use.

    Enhanced with:
    - Method Pattern Tool: Search by philosophical move type
    - Entity-Quote Cross-Index: Bidirectional concept-quote links
    - Query Routing: Question-type-aware strategies
    - Fuzzy Entity Matching: Robust concept lookup
    """

    def __init__(
        self,
        anthropic_model: str = "claude-sonnet-4-5-20250929",
        temperature: float = 1.0,  # Must be 1 when thinking is enabled
        thinking_budget: int = 8000,
        max_tokens: int = 64000,  # Must be > thinking_budget
        corpus_tool: CorpusSearchTool | None = None,
        method_pattern_tool: MethodPatternTool | None = None,
        cross_index: EntityQuoteCrossIndex | None = None,
        enable_routing: bool = True,
    ):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = anthropic_model

        # --- MODEL CONSTRAINTS ---
        # Claude 3 models don't support extended thinking
        if "claude-3-" in self.model:
            self.thinking_budget = 0
            self.max_tokens = min(max_tokens, 4096)
            self.temperature = 0.7
        else:
            # Claude 4.5 models all support extended thinking
            self.thinking_budget = thinking_budget
            self.max_tokens = max_tokens
            self.temperature = temperature

        # Core tools
        self.corpus_tool = corpus_tool
        self.graph_tool = None  # Concept search
        self.traversal_tool = None  # Relationship traversal

        # NEW: Enhanced tools
        self.method_pattern_tool = method_pattern_tool
        self.cross_index = cross_index
        self.enable_routing = enable_routing

        # Citation store for full passage data (cleared per request)
        self.citation_store = {}  # chunk_ref -> {book_title, text, entities}

    def _extract_citations(self, result_str: str) -> list[str]:
        """Extract citation strings from structured tool output or legacy text."""
        try:
            parsed = json.loads(result_str)
            if isinstance(parsed, dict) and "results" in parsed:
                citations = []
                for entry in parsed.get("results", []):
                    cite = entry.get("citation") if isinstance(entry, dict) else None
                    if cite:
                        citations.append(cite.strip())
                return citations
        except Exception:
            pass

        import re
        return re.findall(r'\[[^\]]+ #\d+\]', result_str)

    def set_graph_tools(
        self,
        graph_tool: GraphCommunityIndex,
        traversal_tool: GraphTraversalIndex | None = None
    ):
        """Enable graph search capabilities."""
        self.graph_tool = graph_tool
        self.traversal_tool = traversal_tool

    def set_enhanced_tools(
        self,
        method_pattern_tool: MethodPatternTool | None = None,
        cross_index: EntityQuoteCrossIndex | None = None,
    ):
        """Enable enhanced retrieval tools."""
        self.method_pattern_tool = method_pattern_tool
        self.cross_index = cross_index

        # Update traversal tool with cross-index for better fuzzy matching
        if self.traversal_tool and cross_index:
            self.traversal_tool.cross_index = cross_index

    def _get_tools(self) -> list[dict]:
        """Get tool definitions for the agent."""
        tools = []

        # Core tools
        if self.corpus_tool:
            tools.append(self.corpus_tool.get_tool_definition())
        if self.graph_tool:
            tools.append(self.graph_tool.get_tool_definition())
        if self.traversal_tool:
            tools.append(self.traversal_tool.get_tool_definition())

        # Enhanced tools
        if self.method_pattern_tool:
            tools.append(self.method_pattern_tool.get_tool_definition())
        if self.cross_index:
            tools.append(self.cross_index.get_tool_definition())

        return tools
    
    def _handle_tool_call(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool call and return the result."""

        # --- CORE TOOLS ---

        if tool_name == "search_corpus" and self.corpus_tool:
            query = tool_input.get("query", "")
            n_results = min(tool_input.get("n_results", 10), 50)
            results = self.corpus_tool.search(query, n_results)

            # Annotate with entities if cross-index available
            if self.cross_index:
                results = self.cross_index.annotate_search_results(results)

            structured = []
            for r in results:
                chunk_ref = (r.get("chunk_id") or r.get("id", "")).split("-")[-1]
                book_title = r.get("book_title", "Unknown Source")
                citation = f"[{book_title} #{chunk_ref}]"

                entry = {
                    "citation": citation,
                    "doc_title": book_title,
                    "doc_id": r.get("doc_id"),
                    "chunk_id": chunk_ref,
                    "page_or_offset": r.get("token_start"),
                    "language": r.get("language"),
                    "text": r.get("text", ""),
                    "score": r.get("distance"),
                    "entities": r.get("entities_mentioned", []),
                    "relationships": [],
                    "method_tags": [],
                }

                self.citation_store[chunk_ref] = entry
                structured.append(entry)

            payload = {
                "tool": "search_corpus",
                "results": structured,
            }
            return json.dumps(payload, ensure_ascii=False)

        elif tool_name == "search_concepts" and self.graph_tool:
            query = tool_input.get("query", "")
            results = self.graph_tool.search(query, top_k=5)

            if not results:
                return "No matching concepts found."

            structured = []
            for r in results:
                structured.append(
                    {
                        "concept": r.get("title"),
                        "summary": r.get("summary"),
                        "type": r.get("type"),
                        "score": r.get("score"),
                        "level": r.get("level"),
                    }
                )

            payload = {"tool": "search_concepts", "results": structured}
            return json.dumps(payload, ensure_ascii=False)

        elif tool_name == "traverse_relationships" and self.traversal_tool:
            entity = tool_input.get("entity", "")
            results = self.traversal_tool.get_connections(entity)

            if not results:
                return f"No connections found for '{entity}'. Try a different variation or use search_concepts first to find the exact entity name."

            structured = []
            matched_to = results[0].get("matched_entity") if results else None
            for r in results:
                structured.append(
                    {
                        "concept": r.get("concept"),
                        "description": r.get("description"),
                        "rank": r.get("rank"),
                        "matched_entity": matched_to,
                    }
                )

            payload = {"tool": "traverse_relationships", "results": structured}
            return json.dumps(payload, ensure_ascii=False)

        # --- ENHANCED TOOLS ---

        elif tool_name == "search_method_patterns" and self.method_pattern_tool:
            query = tool_input.get("query", "")
            method_types = tool_input.get("method_types", [])
            n_results = min(tool_input.get("n_results", 5), 20)

            if not method_types:
                return "Error: You must specify at least one method_type. Options: CONCEPT_CREATION, CRITIQUE, EXAMPLE_USAGE, PROBLEM_REFRAMING, ARGUMENTATION, STYLISTIC"

            results = self.method_pattern_tool.search(
                query=query,
                method_types=method_types,
                n_results=n_results
            )

            if not results:
                return f"No passages found matching method types {method_types}. Try different method types or a broader query."

            structured = []
            for r in results:
                chunk_ref = str(r.get("id", "")).split("-")[-1]
                citation = f"[{r.get('doc_id', 'Unknown Doc')} #{chunk_ref}]"
                entry = {
                    "citation": citation,
                    "doc_title": r.get("doc_id", "Unknown Doc"),
                    "doc_id": r.get("doc_id"),
                    "chunk_id": chunk_ref,
                    "text": r.get("text", ""),
                    "language": r.get("language"),
                    "method_tags": r.get("patterns", []),
                    "pattern_confidence": r.get("pattern_confidence"),
                    "score": r.get("score"),
                }
                self.citation_store[chunk_ref] = entry
                structured.append(entry)

            payload = {"tool": "search_method_patterns", "results": structured}
            return json.dumps(payload, ensure_ascii=False)

        elif tool_name == "get_supporting_quotes" and self.cross_index:
            entity = tool_input.get("entity")
            source = tool_input.get("source")
            target = tool_input.get("target")
            top_k = min(tool_input.get("top_k", 5), 10)

            # Entity quote lookup
            if entity:
                results = self.cross_index.get_quotes_for_entity(entity, top_k=top_k)

                if not results:
                    return f"No supporting quotes found for entity '{entity}'. The entity may not have annotated text units, or try a different spelling."

                structured = []
                for r in results:
                    chunk_ref = str(r.get("chunk_id", ""))
                    citation = f"[Entity {entity} #{chunk_ref}]"
                    entry = {
                        "citation": citation,
                        "doc_title": r.get("doc_id", "Unknown Doc"),
                        "doc_id": r.get("doc_id"),
                        "chunk_id": chunk_ref,
                        "text": r.get("text", ""),
                        "entities": [entity] + r.get("other_entities", []),
                        "relationships": [],
                    }
                    self.citation_store[chunk_ref] = entry
                    structured.append(entry)

                payload = {"tool": "get_supporting_quotes", "results": structured, "focus": entity}
                return json.dumps(payload, ensure_ascii=False)

            # Relationship quote lookup
            elif source and target:
                results = self.cross_index.get_quotes_for_relationship(source, target)

                if not results:
                    return f"No supporting quotes found for relationship '{source}' → '{target}'. Try get_supporting_quotes with just one entity."

                structured = []
                rel_desc = results[0].get("relationship_description", "") if results else ""
                for r in results:
                    chunk_ref = str(r.get("chunk_id", ""))
                    citation = f"[{source}↔{target} #{chunk_ref}]"
                    entry = {
                        "citation": citation,
                        "doc_title": r.get("doc_id", "Unknown Doc"),
                        "doc_id": r.get("doc_id"),
                        "chunk_id": chunk_ref,
                        "text": r.get("text", ""),
                        "relationships": [rel_desc] if rel_desc else [],
                        "entities": [source, target],
                    }
                    self.citation_store[chunk_ref] = entry
                    structured.append(entry)

                payload = {
                    "tool": "get_supporting_quotes",
                    "relationship": {"source": source, "target": target, "description": rel_desc},
                    "results": structured,
                }
                return json.dumps(payload, ensure_ascii=False)

            else:
                return "Error: Provide either 'entity' or both 'source' and 'target' parameters."

        return f"Unknown tool: {tool_name}"
    
    def respond(self, question: str) -> dict[str, Any]:
        """
        Generate a Deleuzian response to the question.

        Returns a dict with:
        - answer: The final response text
        - thinking: The extended thinking content (if available)
        - tool_calls: List of tool calls made
        - sources: List of sources cited
        - routing: The routing decision made (if routing enabled)
        """
        messages = [{"role": "user", "content": question}]
        tools = self._get_tools()
        tool_calls_made = []
        sources_cited = set()

        # --- QUERY ROUTING ---
        routing_decision = None
        system_prompt = DELEUZIAN_SYSTEM_PROMPT + "\n\n" + STRUCTURED_TOOL_GUIDE

        if self.enable_routing:
            routing_decision = route_question(question)
            # Add question-type-specific guidance to system prompt
            if routing_decision.system_prompt_addition:
                system_prompt = DELEUZIAN_SYSTEM_PROMPT + "\n\n" + routing_decision.system_prompt_addition

        # Agentic loop: keep going until we get a final response
        MAX_TURNS = 5  # Allow up to 5 turns of tool use
        turn_count = 0
        self_critique_done = False
        self_critique_done = False

        while turn_count < MAX_TURNS:
            turn_count += 1
            # Build request kwargs
            kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": system_prompt,
                "messages": messages,
            }

            # --- MECHANICAL ADJUSTMENT: CONCEPT INJECTION ---
            if self.graph_tool and turn_count == 1:
                # If routing suggests specific method patterns, inject relevant exemplar
                if routing_decision and self.method_pattern_tool:
                    method_types = routing_decision.config.method_patterns
                    exemplars = self.method_pattern_tool.get_exemplars(method_types, limit=1)
                    if exemplars:
                        injection = (
                            f"\n\n--- RETRIEVAL HINT ---\n"
                            f"Question type detected: **{routing_decision.question_type.value.upper()}**\n"
                            f"Suggested tools: {', '.join(routing_decision.suggested_tools_order)}\n"
                            f"Method patterns to seek: {', '.join(method_types)}\n"
                        )
                        kwargs["system"] = system_prompt + injection
                else:
                    # Fallback to random concept injection
                    attractor = self.graph_tool.get_random_concept()
                    if attractor:
                        injection = (
                            f"\n\n--- MECHANICAL INJECTION ---\n"
                            f"Current Philosophical Attractor (Lens): **{attractor['title']}**\n"
                            f"Absorb this concept into your becoming: {attractor['summary'][:300]}..."
                        )
                        kwargs["system"] = system_prompt + injection
            
            # Add extended thinking if budget > 0
            if self.thinking_budget > 0:
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget,
                }
            
            # Add tools if available
            if tools:
                kwargs["tools"] = tools
            
            # Make the API call with streaming (required for Opus with extended thinking)
            thinking_content = ""
            text_content = ""
            tool_use_blocks = []
            
            with self.client.messages.stream(**kwargs) as stream:
                response = stream.get_final_message()
            
            for block in response.content:
                if block.type == "thinking":
                    thinking_content += block.thinking
                elif block.type == "text":
                    text_content += block.text
                elif block.type == "tool_use":
                    tool_use_blocks.append(block)
            
            # If no tool use, either return or trigger a self-critique retry if citations are missing
            if not tool_use_blocks:
                if not sources_cited and tools and not self_critique_done:
                    messages.append(
                        {
                            "role": "user",
                            "content": "SELF-CRITIQUE: Your draft lacks grounded citations. Call the retrieval tools per the checklist, then answer with citations.",
                        }
                    )
                    self_critique_done = True
                    continue

                result = {
                    "answer": text_content,
                    "thinking": thinking_content,
                    "tool_calls": tool_calls_made,
                    "sources": list(sources_cited),
                }
                if routing_decision:
                    result["routing"] = {
                        "question_type": routing_decision.question_type.value,
                        "confidence": routing_decision.confidence,
                        "suggested_tools": routing_decision.suggested_tools_order,
                    }
                return result
            
            # Handle tool calls
            # Add assistant message with the tool use
            messages.append({
                "role": "assistant",
                "content": response.content,
            })
            
            # Execute tools and add results
            tool_results = []
            for tool_block in tool_use_blocks:
                tool_name = tool_block.name
                tool_input = tool_block.input
                tool_id = tool_block.id
                
                # Execute the tool
                result = self._handle_tool_call(tool_name, tool_input)
                
                # Track tool calls
                tool_calls_made.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "result_preview": result[:200] + "..." if len(result) > 200 else result,
                })
                
                # Extract sources from results
                for cite in self._extract_citations(result):
                    # Keep raw citation string without brackets for readability
                    sources_cited.add(cite.strip("[]"))
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result,
                })
            
            # Add tool results
            messages.append({
                "role": "user",
                "content": tool_results,
            })

    def stream_respond(self, question: str, history: list = None) -> typing.Generator[dict, None, None]:
        """
        Stream response events for web interface.
        Yields dicts with type: 'thinking', 'content', 'tool_call', 'sources', 'citations', 'routing', 'done', 'error'

        Args:
            question: The user's question
            history: Optional list of prior messages [{"role": "user/assistant", "content": "..."}]
        """
        # Clear citation store for this request
        self.citation_store = {}

        # Start with history if provided, then add current question
        messages = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        tools = self._get_tools()
        sources_cited = set()

        # --- QUERY ROUTING ---
        routing_decision = None
        system_prompt = DELEUZIAN_SYSTEM_PROMPT + "\n\n" + STRUCTURED_TOOL_GUIDE

        if self.enable_routing:
            routing_decision = route_question(question)
            # Add question-type-specific guidance to system prompt
            if routing_decision.system_prompt_addition:
                system_prompt = DELEUZIAN_SYSTEM_PROMPT + "\n\n" + routing_decision.system_prompt_addition

            # Yield routing info to frontend
            yield {
                "type": "routing",
                "question_type": routing_decision.question_type.value,
                "confidence": routing_decision.confidence,
                "suggested_tools": routing_decision.suggested_tools_order,
            }

        # Agentic loop
        MAX_TURNS = 5
        turn_count = 0

        while turn_count < MAX_TURNS:
            turn_count += 1

            kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": system_prompt,
                "messages": messages,
            }

            # --- MECHANICAL ADJUSTMENT: CONCEPT INJECTION ---
            if self.graph_tool and turn_count == 1:
                # If routing suggests specific method patterns, inject relevant exemplar
                if routing_decision and self.method_pattern_tool:
                    method_types = routing_decision.config.method_patterns
                    exemplars = self.method_pattern_tool.get_exemplars(method_types, limit=1)
                    if exemplars:
                        injection = (
                            f"\n\n--- RETRIEVAL HINT ---\n"
                            f"Question type detected: **{routing_decision.question_type.value.upper()}**\n"
                            f"Suggested tools: {', '.join(routing_decision.suggested_tools_order)}\n"
                            f"Method patterns to seek: {', '.join(method_types)}\n"
                        )
                        kwargs["system"] = system_prompt + injection
                else:
                    # Fallback to random concept injection
                    attractor = self.graph_tool.get_random_concept()
                    if attractor:
                        injection = (
                            f"\n\n--- MECHANICAL INJECTION ---\n"
                            f"Current Philosophical Attractor (Lens): **{attractor['title']}**\n"
                            f"Absorb this concept into your becoming: {attractor['summary'][:300]}..."
                        )
                        kwargs["system"] = system_prompt + injection
            
            # Add extended thinking if budget > 0
            if self.thinking_budget > 0:
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget,
                }
            if tools:
                kwargs["tools"] = tools
            
            # Stream API call
            tool_use_blocks = []
            text_content = ""
            thinking_content = ""
            current_tool_id = None
            current_tool_name = None

            try:
                with self.client.messages.stream(**kwargs) as stream:
                    for event in stream:
                        if event.type == "content_block_start":
                            block = event.content_block
                            if block.type == "thinking":
                                yield {"type": "thinking_start"}
                            elif block.type == "text":
                                yield {"type": "content_start"}
                            elif block.type == "tool_use":
                                current_tool_id = block.id
                                current_tool_name = block.name
                                yield {"type": "tool_call_start", "tool": current_tool_name}

                        elif event.type == "content_block_delta":
                            delta = event.delta
                            if delta.type == "thinking_delta":
                                thinking_content += delta.thinking
                                yield {"type": "thinking", "content": delta.thinking}
                                # Emit progress every ~500 chars (~100 tokens)
                                if len(thinking_content) % 500 < len(delta.thinking):
                                    approx_tokens = len(thinking_content) // 4  # Rough estimate
                                    yield {
                                        "type": "thinking_progress",
                                        "tokens": approx_tokens,
                                        "budget": self.thinking_budget
                                    }
                            elif delta.type == "text_delta":
                                text_content += delta.text
                                yield {"type": "content", "content": delta.text}
                            # Tool input delta handling if needed in future

                    response = stream.get_final_message()
            except Exception as e:
                yield {"type": "error", "message": str(e)}
                return

            # Collect completed blocks from response object
            for block in response.content:
                if block.type == "tool_use":
                    tool_use_blocks.append(block)
            
            # If no tool use, either finish or trigger a self-critique retry
            if not tool_use_blocks:
                source_matches = re.findall(r'\[([^\]]+) #\d+\]', text_content)
                sources_cited.update(source_matches)

                if not sources_cited and tools and not self_critique_done:
                    messages.append(
                        {
                            "role": "user",
                            "content": "SELF-CRITIQUE: Your draft lacks grounded citations. Call retrieval tools per the checklist, then answer with citations.",
                        }
                    )
                    self_critique_done = True
                    continue

                yield {"type": "sources", "sources": list(sources_cited)}

                # Emit full citation data for frontend popovers
                if self.citation_store:
                    yield {"type": "citations", "citations": list(self.citation_store.values())}

                yield {"type": "done"}
                return
            
            # Helper message for tool use
            messages.append({"role": "assistant", "content": response.content})
            
            # Handle tools
            tool_results = []
            for tool_block in tool_use_blocks:
                tool_name = tool_block.name
                tool_input = tool_block.input
                tool_id = tool_block.id

                # Notify frontend of tool start
                yield {"type": "tool_call_start", "tool": tool_name}

                # Execute
                result = self._handle_tool_call(tool_name, tool_input)

                # Extract sources from result
                for cite in self._extract_citations(result):
                    sources_cited.add(cite.strip("[]"))

                # Extract concepts from result for visualization
                concepts_found = []
                if tool_name in ("search_concepts", "traverse_relationships"):
                    # Extract concept names from graph results
                    concept_matches = re.findall(r'\*\*([^*]+)\*\*', result)
                    concepts_found = concept_matches[:10]  # Limit to 10
                elif tool_name == "get_supporting_quotes":
                    # Extract from "Also discusses:" annotations
                    discusses_match = re.findall(r'\[Also discusses: ([^\]]+)\]', result)
                    if discusses_match:
                        for match in discusses_match:
                            concepts_found.extend([c.strip() for c in match.split(',')])

                # Notify frontend of tool completion with results
                yield {
                    "type": "tool_call",
                    "tool": tool_name,
                    "query": str(tool_input),
                    "concepts_found": concepts_found
                }

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})


def create_agent_from_env(
    model_name: str | None = None, 
    thinking_budget: int | None = None,
    max_tokens: int | None = None
) -> DeleuzianAgent:
    """Initialize a fully configured agent from environment variables."""
    # Initialize OpenAI client for embeddings
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for GraphRAG embeddings")
    openai_client = openai.Client(api_key=openai_api_key)
    
    # Embedding model configuration (Must match GraphRAG settings)
    embed_model = os.environ.get("GRAPHRAG_EMBEDDING_MODEL", "text-embedding-3-large")

    # Initialize corpus tool (LanceDB from GraphRAG)
    lancedb_path = Path(os.environ.get("GRAPHRAG_LANCEDB_DIR", "graphrag_project/output/lancedb"))
    table_name = os.environ.get("GRAPHRAG_TABLE_NAME", "default-text_unit-text") # Standard GraphRAG table
    
    corpus_tool = CorpusSearchTool(
        lancedb_uri=str(lancedb_path),
        table_name=table_name,
        openai_client=openai_client,
        embed_model=embed_model,
    )
    
    # Initialize agent
    agent = DeleuzianAgent(
        anthropic_model=model_name or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
        temperature=1.0,  # Thinking requires temperature 1.0
        thinking_budget=thinking_budget if thinking_budget is not None else int(os.environ.get("THINKING_BUDGET", 30000)),
        max_tokens=max_tokens if max_tokens is not None else 64000,
        corpus_tool=corpus_tool,
    )
    
    # GraphRAG Tools
    graph_out = Path(os.environ.get("GRAPHRAG_OUTPUT", "graphrag_project/output"))
    
    # 1. Entity/Community Index
    # Try entities first, then community reports
    report_path = graph_out / "entities.parquet"
    if not report_path.exists():
        report_path = graph_out / "create_final_entities.parquet"
    if not report_path.exists():
        report_path = graph_out / "community_reports.parquet"
        
    graph_tool = None
    if report_path.exists():
        try:
            graph_tool = GraphCommunityIndex(
                report_path=report_path,
                openai_client=openai_client,
                embed_model=embed_model,
            )
        except Exception as e:
            print(f"Failed to load graph index: {e}")
            
    # 2. Relationship path (needed for cross-index and traversal)
    rel_path = graph_out / "relationships.parquet"
    if not rel_path.exists():
        rel_path = graph_out / "create_final_relationships.parquet"

    # 3. Cross-Index (Entity <-> Quote links)
    cross_index = None
    text_units_path = graph_out / "text_units.parquet"
    if report_path.exists() and rel_path.exists() and text_units_path.exists():
        try:
            cross_index = create_cross_index()
            print("Cross-index loaded: entity ↔ quote links enabled.")
        except Exception as e:
            print(f"Warning: Failed to load cross-index: {e}")

    # 4. Traversal Index (with cross-index for fuzzy matching)
    traversal_tool = None
    if rel_path.exists():
        traversal_tool = GraphTraversalIndex(rel_path, cross_index=cross_index)

    # 5. Method Pattern Tool
    method_pattern_tool = None
    patterns_path = Path("deleuze_corpus/method_patterns.parquet")
    vectors_path = Path("deleuze_corpus/vectors.parquet")
    if patterns_path.exists() and vectors_path.exists():
        try:
            method_pattern_tool = create_method_pattern_tool(openai_client)
            print(f"Method pattern tool loaded: {method_pattern_tool.total_chunks} chunks with pattern classifications.")
        except Exception as e:
            print(f"Warning: Failed to load method pattern tool: {e}")

    # Set all tools
    if graph_tool:
        agent.set_graph_tools(graph_tool, traversal_tool)

    agent.set_enhanced_tools(
        method_pattern_tool=method_pattern_tool,
        cross_index=cross_index,
    )

    return agent



# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deleuzian Thinking Machine - Performative philosophical agent"
    )
    parser.add_argument(
        "--question",
        required=True,
        help="The question or prompt for the Deleuzian agent",
    )
    parser.add_argument(
        "--vector-store",
        type=Path,
        default=Path("graphrag_project/output/lancedb"),
        help="GraphRAG LanceDB directory",
    )
    parser.add_argument(
        "--table-name",
        default="default-text_unit-text",
        help="LanceDB table name",
    )
    parser.add_argument(
        "--embed-model",
        default="text-embedding-3-large",
        help="OpenAI embedding model for search",
    )
    parser.add_argument(
        "--anthropic-model",
        default="claude-opus-4-5-20251101",
        help="Anthropic model for synthesis (must support extended thinking)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for creative synthesis (default 1.0)",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=30000,
        help="Token budget for extended thinking (default 30000, must be < max_tokens)",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="Include extended thinking in output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Initialize OpenAI client for embeddings
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for corpus search")
    openai_client = openai.Client(api_key=openai_api_key)
    
    # Initialize corpus search tool
    lancedb_path = args.vector_store
    corpus_tool = CorpusSearchTool(
        lancedb_uri=str(lancedb_path),
        table_name=args.table_name,
        openai_client=openai_client,
        embed_model=args.embed_model,
    )
    
    # Initialize agent
    agent = DeleuzianAgent(
        anthropic_model=args.anthropic_model,
        temperature=args.temperature,
        thinking_budget=args.thinking_budget,
        corpus_tool=corpus_tool,
    )

    # Initialize graph tool if reports exist
    # Use entities.parquet as primary source since community reports were skipped
    report_path = Path("graphrag_project/output/entities.parquet")
    if not report_path.exists():
        report_path = Path("graphrag_project/output/create_final_entities.parquet")
    
    # Fallback to community reports if entities missing (unlikely)
    if not report_path.exists():
        report_path = Path("graphrag_project/output/community_reports.parquet")
    
    
    if report_path.exists():
        print(f"Loading GraphRAG data from {report_path}...")
        try:
            graph_tool = GraphCommunityIndex(
                report_path=report_path,
                openai_client=openai_client,
                embed_model=args.embed_model,
            )
            
            # Initialize traversal tool
            rel_path = Path("graphrag_project/output/relationships.parquet")
            if not rel_path.exists():
                rel_path = Path("graphrag_project/output/create_final_relationships.parquet")
            
            traversal_tool = None
            if rel_path.exists():
                print(f"Loading GraphRAG relationships from {rel_path}...")
                traversal_tool = GraphTraversalIndex(rel_path)
            
            agent.set_graph_tools(graph_tool, traversal_tool)
            print("Graph search enabled.")
        except Exception as e:
            print(f"Failed to load graph index: {e}")
    else:
        print("GraphRAG community reports not found. Running in text-search only mode.")
    
    # Generate response
    result = agent.respond(args.question)
    
    # Output
    if args.json:
        output = {
            "question": args.question,
            "answer": result["answer"],
            "sources": result["sources"],
            "tool_calls": result["tool_calls"],
        }
        if args.show_thinking:
            output["thinking"] = result["thinking"]
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print("\n" + "=" * 60)
        print("QUESTION:", args.question)
        print("=" * 60)
        
        if args.show_thinking and result["thinking"]:
            print("\n--- THINKING ---")
            print(result["thinking"][:2000])
            if len(result["thinking"]) > 2000:
                print(f"... [{len(result['thinking']) - 2000} more characters]")
            print("--- END THINKING ---\n")
        
        print("\n--- RESPONSE ---")
        print(result["answer"])
        print("--- END RESPONSE ---\n")
        
        if result["sources"]:
            print("Sources:", ", ".join(result["sources"][:10]))
        
        if result["tool_calls"]:
            print(f"\nTool calls made: {len(result['tool_calls'])}")


if __name__ == "__main__":
    main()
