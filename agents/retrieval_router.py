#!/usr/bin/env python3
"""
Retrieval Router: Question-aware retrieval strategy selection.

Routes different question types to optimal retrieval strategies:
- DEFINITIONAL: Graph-first (concept definitions) + vector search
- OPERATIONAL: Method patterns (EXAMPLE_USAGE) + vector search
- RELATIONAL: Relationship traversal + cross-index quotes
- COMPARATIVE: Multi-entity search + diff
- CRITIQUE: Method patterns (CRITIQUE) + vector search
- CREATIVE: Random walk + high-temperature expansion
- GENEALOGICAL: Citation search + graph genealogy
- APPLIED: Example patterns + vector search

Each strategy returns context optimized for the question type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum

from .question_classifier import QuestionType, classify_question, get_method_patterns_for_type


@dataclass
class RetrievalConfig:
    """Configuration for a retrieval strategy."""
    primary_tool: str  # Which tool to use first
    secondary_tool: Optional[str]  # Fallback/supplement tool
    method_patterns: list[str]  # Method types to boost
    graph_first: bool  # Whether to search graph before corpus
    expand_via_graph: bool  # Whether to expand results via entity connections
    n_primary: int  # Number of primary results
    n_secondary: int  # Number of secondary results


# Routing matrix: question type -> retrieval configuration
ROUTING_MATRIX: dict[QuestionType, RetrievalConfig] = {
    QuestionType.DEFINITIONAL: RetrievalConfig(
        primary_tool="search_concepts",
        secondary_tool="search_corpus",
        method_patterns=["CONCEPT_CREATION", "STYLISTIC"],
        graph_first=True,
        expand_via_graph=True,
        n_primary=5,
        n_secondary=5,
    ),
    QuestionType.OPERATIONAL: RetrievalConfig(
        primary_tool="search_method_patterns",
        secondary_tool="search_corpus",
        method_patterns=["EXAMPLE_USAGE", "ARGUMENTATION"],
        graph_first=False,
        expand_via_graph=True,
        n_primary=5,
        n_secondary=5,
    ),
    QuestionType.RELATIONAL: RetrievalConfig(
        primary_tool="traverse_relationships",
        secondary_tool="get_supporting_quotes",
        method_patterns=["ARGUMENTATION", "CONCEPT_CREATION"],
        graph_first=True,
        expand_via_graph=True,
        n_primary=8,
        n_secondary=5,
    ),
    QuestionType.GENEALOGICAL: RetrievalConfig(
        primary_tool="search_corpus",
        secondary_tool="search_concepts",
        method_patterns=["CRITIQUE", "PROBLEM_REFRAMING"],
        graph_first=False,
        expand_via_graph=False,
        n_primary=8,
        n_secondary=3,
    ),
    QuestionType.COMPARATIVE: RetrievalConfig(
        primary_tool="search_concepts",
        secondary_tool="search_corpus",
        method_patterns=["CRITIQUE", "CONCEPT_CREATION"],
        graph_first=True,
        expand_via_graph=False,  # Handle comparison explicitly
        n_primary=5,
        n_secondary=5,
    ),
    QuestionType.APPLIED: RetrievalConfig(
        primary_tool="search_method_patterns",
        secondary_tool="search_corpus",
        method_patterns=["EXAMPLE_USAGE", "PROBLEM_REFRAMING"],
        graph_first=False,
        expand_via_graph=True,
        n_primary=5,
        n_secondary=5,
    ),
    QuestionType.CREATIVE: RetrievalConfig(
        primary_tool="search_concepts",
        secondary_tool="search_method_patterns",
        method_patterns=["PROBLEM_REFRAMING", "CONCEPT_CREATION", "STYLISTIC"],
        graph_first=True,
        expand_via_graph=True,
        n_primary=5,
        n_secondary=5,
    ),
    QuestionType.CRITIQUE: RetrievalConfig(
        primary_tool="search_method_patterns",
        secondary_tool="search_corpus",
        method_patterns=["CRITIQUE", "ARGUMENTATION"],
        graph_first=False,
        expand_via_graph=True,
        n_primary=5,
        n_secondary=5,
    ),
}


@dataclass
class RoutingDecision:
    """Result of routing a question."""
    question_type: QuestionType
    config: RetrievalConfig
    confidence: float
    suggested_tools_order: list[str]
    method_pattern_hint: Optional[str]
    system_prompt_addition: str


def route_question(question: str) -> RoutingDecision:
    """
    Analyze a question and determine optimal retrieval strategy.

    Returns a RoutingDecision with:
    - The classified question type
    - Retrieval configuration
    - Suggested tool order
    - Additional system prompt guidance
    """
    classification = classify_question(question)
    config = ROUTING_MATRIX[classification.question_type]

    # Build tool order
    tools_order = [config.primary_tool]
    if config.secondary_tool:
        tools_order.append(config.secondary_tool)

    # Add method pattern tool if relevant patterns
    if config.method_patterns and "search_method_patterns" not in tools_order:
        tools_order.append("search_method_patterns")

    # Add supporting quotes if graph-first
    if config.graph_first and "get_supporting_quotes" not in tools_order:
        tools_order.append("get_supporting_quotes")

    # Build method pattern hint
    method_hint = None
    if config.method_patterns:
        method_hint = f"Prefer passages with these philosophical moves: {', '.join(config.method_patterns)}"

    # Build system prompt addition
    prompt_addition = _build_prompt_addition(classification.question_type, config)

    return RoutingDecision(
        question_type=classification.question_type,
        config=config,
        confidence=classification.confidence,
        suggested_tools_order=tools_order,
        method_pattern_hint=method_hint,
        system_prompt_addition=prompt_addition,
    )


def _build_prompt_addition(q_type: QuestionType, config: RetrievalConfig) -> str:
    """Build question-type-specific guidance for the system prompt."""
    additions = {
        QuestionType.DEFINITIONAL: """
**QUESTION TYPE: DEFINITIONAL** ("What is X?")
STRATEGY:
1. First search_concepts to find the structural position of the concept
2. Then get_supporting_quotes for the entity to find exact definitional passages
3. Use search_method_patterns with CONCEPT_CREATION to find how Deleuze introduces this concept
STYLE: Define by function, not essence. "X is not... X is rather..."
""",
        QuestionType.OPERATIONAL: """
**QUESTION TYPE: OPERATIONAL** ("How does X work?")
STRATEGY:
1. Use search_method_patterns with EXAMPLE_USAGE to find concrete examples
2. Use search_corpus to find operational descriptions
3. Show the concept in action, not in abstraction
STYLE: Lists of affects, machinic operations. "X does this, produces that, connects with..."
""",
        QuestionType.RELATIONAL: """
**QUESTION TYPE: RELATIONAL** ("How does X relate to Y?")
STRATEGY:
1. Use traverse_relationships to find graph connections between concepts
2. Use get_supporting_quotes to find passages discussing the relationship
3. Map the rhizomatic connections—there may be multiple paths
STYLE: AND...AND...AND, not is/is not. Show the assemblage.
""",
        QuestionType.COMPARATIVE: """
**QUESTION TYPE: COMPARATIVE** ("What's the difference between X and Y?")
STRATEGY:
1. Search for each concept separately with search_concepts
2. Use search_method_patterns with CRITIQUE to find contrastive passages
3. Find where Deleuze explicitly opposes or distinguishes them
STYLE: Binary opposition that explodes into multiplicity. Not either/or but both differently.
""",
        QuestionType.GENEALOGICAL: """
**QUESTION TYPE: GENEALOGICAL** ("Where does X come from?")
STRATEGY:
1. Use search_corpus to find historical/developmental passages
2. Use search_method_patterns with PROBLEM_REFRAMING to find how Deleuze reconceives origins
3. Look for critique of prior thinkers (Hegel, Freud, etc.)
STYLE: Not linear origin but emergence from a field of forces.
""",
        QuestionType.APPLIED: """
**QUESTION TYPE: APPLIED** ("How does X apply to Y?")
STRATEGY:
1. Use search_method_patterns with EXAMPLE_USAGE to find analogous applications
2. Use search_corpus to find related case studies
3. Create new connections rhizomatically
STYLE: The example IS the concept. Don't illustrate—perform.
""",
        QuestionType.CREATIVE: """
**QUESTION TYPE: CREATIVE** ("What would Deleuze say about X?")
STRATEGY:
1. Use search_concepts to find related conceptual territories
2. Use search_method_patterns with PROBLEM_REFRAMING and STYLISTIC
3. Create NEW connections—don't just apply old answers
STYLE: Invent the concept the problem demands. Be aggressive, surprising.
""",
        QuestionType.CRITIQUE: """
**QUESTION TYPE: CRITIQUE** ("What's wrong with X?")
STRATEGY:
1. Use search_method_patterns with CRITIQUE to find similar attacks
2. Use search_corpus to find specific arguments against the target
3. Look for how Deleuze destroys pious interpretations
STYLE: "Destroy, destroy." Name the enemy. Show what's at stake.
""",
    }

    return additions.get(q_type, "")


def get_routing_summary() -> str:
    """Return a summary of the routing matrix for documentation."""
    lines = ["# Question Type -> Retrieval Strategy Mapping\n"]
    for q_type, config in ROUTING_MATRIX.items():
        lines.append(f"## {q_type.value.upper()}")
        lines.append(f"- Primary: {config.primary_tool}")
        lines.append(f"- Secondary: {config.secondary_tool}")
        lines.append(f"- Method Patterns: {config.method_patterns}")
        lines.append(f"- Graph First: {config.graph_first}")
        lines.append("")
    return "\n".join(lines)


# Tool usage guidance for the agent
TOOL_USAGE_GUIDE = """
## AVAILABLE TOOLS AND WHEN TO USE THEM

### 1. search_corpus
**What**: Semantic vector search over all text chunks (GraphRAG text_units)
**When**: You need specific textual passages, exact quotes, or concrete phrasing
**How**: Describe the passage you want. Be specific. "passage about desire as production" not just "desire"
**Returns**: Text chunks with book titles and chunk IDs for citation

### 2. search_concepts
**What**: Search over high-level concept/entity summaries from the knowledge graph
**When**: You need to understand the STRUCTURE of a concept, its place in the system
**How**: Search by concept name or thematic description
**Returns**: Entity summaries, types, and their network position

### 3. traverse_relationships
**What**: Follow edges in the concept graph from a specific entity
**When**: You need to find CONNECTIONS between concepts, follow lines of flight
**How**: Provide exact entity name (fuzzy matched). Get back connected concepts.
**Returns**: Related concepts with relationship descriptions

### 4. search_method_patterns
**What**: Search filtered by PHILOSOPHICAL MOVE TYPE (how Deleuze argues)
**When**: You need examples of specific moves—critiques, concept creation, examples
**How**: Specify method_types: CONCEPT_CREATION, CRITIQUE, EXAMPLE_USAGE, PROBLEM_REFRAMING, ARGUMENTATION, STYLISTIC
**Returns**: Passages classified by their philosophical function

### 5. get_supporting_quotes
**What**: Get exact passages that support a concept or relationship
**When**: AFTER using search_concepts or traverse_relationships, to get the textual evidence
**How**: Provide entity name OR (source, target) for a relationship
**Returns**: Text chunks with co-occurring entities marked

## RECOMMENDED WORKFLOWS

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
Example: [A Thousand Plateaus #0234]
"""


if __name__ == "__main__":
    # Test routing
    test_questions = [
        "What is the Body without Organs?",
        "How does schizoanalysis work?",
        "What's the relationship between desire and production?",
        "How does rhizome differ from tree?",
        "What would Deleuze say about AI?",
        "What's wrong with psychoanalysis?",
        "Where does the concept of becoming come from?",
        "How does the war machine apply to capitalism?",
    ]

    for q in test_questions:
        decision = route_question(q)
        print(f"Q: {q}")
        print(f"   Type: {decision.question_type.value}")
        print(f"   Tools: {decision.suggested_tools_order}")
        print(f"   Methods: {decision.config.method_patterns}")
        print()
