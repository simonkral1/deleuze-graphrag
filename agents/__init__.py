"""
Deleuzian Thinking Machine - Agent modules

This package contains the agentic components for the Deleuzian philosophy system:

- deleuzian_agent: Main agent with Claude Opus 4.5 + extended thinking
- method_pattern_tool: Search by philosophical move type (CONCEPT_CREATION, CRITIQUE, etc.)
- cross_index: Bidirectional entity ↔ quote links
- retrieval_router: Question-type-aware retrieval strategies
- question_classifier: Classify questions by type (definitional, operational, relational, etc.)
- nomad_agent: Legacy agent (simpler, GPT-4o-mini synthesis)
"""

from .deleuzian_agent import (
    DeleuzianAgent,
    CorpusSearchTool,
    GraphCommunityIndex,
    GraphTraversalIndex,
    create_agent_from_env,
    DELEUZIAN_SYSTEM_PROMPT,
)

from .method_pattern_tool import (
    MethodPatternTool,
    create_method_pattern_tool,
    VALID_PATTERNS,
    PATTERN_DESCRIPTIONS,
)

from .cross_index import (
    EntityQuoteCrossIndex,
    create_cross_index,
)

from .retrieval_router import (
    route_question,
    RoutingDecision,
    RetrievalConfig,
    ROUTING_MATRIX,
    TOOL_USAGE_GUIDE,
)

from .question_classifier import (
    QuestionType,
    QuestionClassification,
    classify_question,
    get_method_patterns_for_type,
)

__all__ = [
    # Main agent
    "DeleuzianAgent",
    "create_agent_from_env",
    "DELEUZIAN_SYSTEM_PROMPT",
    # Core tools
    "CorpusSearchTool",
    "GraphCommunityIndex",
    "GraphTraversalIndex",
    # Enhanced tools
    "MethodPatternTool",
    "create_method_pattern_tool",
    "VALID_PATTERNS",
    "PATTERN_DESCRIPTIONS",
    "EntityQuoteCrossIndex",
    "create_cross_index",
    # Routing
    "route_question",
    "RoutingDecision",
    "RetrievalConfig",
    "ROUTING_MATRIX",
    "TOOL_USAGE_GUIDE",
    # Classification
    "QuestionType",
    "QuestionClassification",
    "classify_question",
    "get_method_patterns_for_type",
]
