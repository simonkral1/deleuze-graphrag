#!/usr/bin/env python3
"""
Question classifier for routing Deleuzian queries to appropriate method patterns.
Classifies questions by type to enable targeted retrieval and response synthesis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class QuestionType(Enum):
    """Types of philosophical questions with Deleuzian handling strategies."""
    DEFINITIONAL = "definitional"      # "What is X?"
    OPERATIONAL = "operational"        # "How does X work?"
    RELATIONAL = "relational"          # "How does X relate to Y?"
    GENEALOGICAL = "genealogical"      # "Where does X come from?"
    COMPARATIVE = "comparative"        # "What's the difference between X and Y?"
    APPLIED = "applied"                # "How does X apply to [case]?"
    CREATIVE = "creative"              # "What would Deleuze say about [new topic]?"
    CRITIQUE = "critique"              # "What's wrong with X?" or "Why reject X?"


@dataclass
class QuestionClassification:
    """Result of question classification."""
    question_type: QuestionType
    confidence: float
    requires_creativity: bool
    matched_pattern: Optional[str] = None


# Rule-based patterns for fast classification
QUESTION_PATTERNS = {
    QuestionType.DEFINITIONAL: [
        r"^what\s+is\s+(the\s+)?",
        r"^define\s+",
        r"^what\s+does\s+.+\s+mean",
        r"^what\s+are\s+(the\s+)?",
        r"^explain\s+(the\s+concept\s+of\s+)?",
    ],
    QuestionType.OPERATIONAL: [
        r"^how\s+does\s+.+\s+work",
        r"^how\s+does\s+.+\s+function",
        r"^how\s+does\s+.+\s+operate",
        r"^what\s+does\s+.+\s+do\b",
        r"^how\s+is\s+.+\s+(used|applied|practiced)",
    ],
    QuestionType.RELATIONAL: [
        r"^how\s+does\s+.+\s+relate\s+to",
        r"^what\s+is\s+the\s+(relationship|connection)\s+between",
        r"^how\s+are\s+.+\s+(and|with)\s+.+\s+(connected|related|linked)",
        r".+\s+and\s+.+\s+relationship",
    ],
    QuestionType.GENEALOGICAL: [
        r"^where\s+does\s+.+\s+come\s+from",
        r"^what\s+is\s+the\s+(origin|source|history)\s+of",
        r"^how\s+did\s+.+\s+(develop|emerge|evolve)",
        r"^what\s+led\s+to",
    ],
    QuestionType.COMPARATIVE: [
        r"^what('s|\s+is)\s+the\s+difference\s+between",
        r"^how\s+does\s+.+\s+differ\s+from",
        r"^compare\s+",
        r"^.+\s+vs\.?\s+",
        r"^how\s+is\s+.+\s+different\s+from",
        r"^contrast\s+",
    ],
    QuestionType.APPLIED: [
        r"^how\s+(can|would|does)\s+.+\s+apply\s+to",
        r"^what\s+would\s+.+\s+look\s+like\s+in",
        r"^how\s+to\s+use\s+.+\s+for",
        r"^apply\s+.+\s+to",
    ],
    QuestionType.CREATIVE: [
        r"^what\s+would\s+deleuze\s+say\s+about",
        r"^what\s+would\s+deleuze\s+think\s+about",
        r"^how\s+would\s+deleuze\s+(approach|analyze|view)",
        r"^from\s+a\s+deleuz(ian|e)\s+(perspective|view)",
        r"^what\s+is\s+a\s+deleuzian\s+(take|view|analysis)\s+on",
    ],
    QuestionType.CRITIQUE: [
        r"^what('s|\s+is)\s+wrong\s+with",
        r"^why\s+(does|did)\s+deleuze\s+(reject|criticize|oppose)",
        r"^what\s+are\s+the\s+problems\s+with",
        r"^critique\s+of\s+",
        r"^why\s+not\s+",
    ],
}

# Question types that require more creative synthesis (less retrieval-dependent)
CREATIVE_TYPES = {
    QuestionType.CREATIVE,
    QuestionType.APPLIED,
}


def classify_question(question: str) -> QuestionClassification:
    """
    Classify a question by type using rule-based pattern matching.
    
    Args:
        question: The question to classify
        
    Returns:
        QuestionClassification with type, confidence, and metadata
    """
    question_lower = question.lower().strip()
    
    # Try to match against known patterns
    for q_type, patterns in QUESTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, question_lower, re.IGNORECASE):
                return QuestionClassification(
                    question_type=q_type,
                    confidence=0.85,
                    requires_creativity=q_type in CREATIVE_TYPES,
                    matched_pattern=pattern,
                )
    
    # Fallback heuristics for unmatched questions
    if "?" not in question:
        # Statements or imperatives - treat as creative/applied
        return QuestionClassification(
            question_type=QuestionType.CREATIVE,
            confidence=0.5,
            requires_creativity=True,
            matched_pattern=None,
        )
    
    # Default to definitional for simple "what" questions
    if question_lower.startswith("what"):
        return QuestionClassification(
            question_type=QuestionType.DEFINITIONAL,
            confidence=0.6,
            requires_creativity=False,
            matched_pattern=None,
        )
    
    # Default to operational for "how" questions
    if question_lower.startswith("how"):
        return QuestionClassification(
            question_type=QuestionType.OPERATIONAL,
            confidence=0.6,
            requires_creativity=False,
            matched_pattern=None,
        )
    
    # Last resort: treat as creative (requires more synthesis)
    return QuestionClassification(
        question_type=QuestionType.CREATIVE,
        confidence=0.4,
        requires_creativity=True,
        matched_pattern=None,
    )


def get_method_patterns_for_type(question_type: QuestionType) -> list[str]:
    """
    Map question types to relevant method pattern categories.
    
    Returns list of method pattern types that should be retrieved
    for this question type.
    """
    mapping = {
        QuestionType.DEFINITIONAL: ["CONCEPT_CREATION", "STYLISTIC"],
        QuestionType.OPERATIONAL: ["EXAMPLE_USAGE", "ARGUMENTATION"],
        QuestionType.RELATIONAL: ["ARGUMENTATION", "CONCEPT_CREATION"],
        QuestionType.GENEALOGICAL: ["CRITIQUE", "PROBLEM_REFRAMING"],
        QuestionType.COMPARATIVE: ["CRITIQUE", "CONCEPT_CREATION"],
        QuestionType.APPLIED: ["EXAMPLE_USAGE", "PROBLEM_REFRAMING"],
        QuestionType.CREATIVE: ["PROBLEM_REFRAMING", "CONCEPT_CREATION", "STYLISTIC"],
        QuestionType.CRITIQUE: ["CRITIQUE", "ARGUMENTATION"],
    }
    return mapping.get(question_type, ["CONCEPT_CREATION"])


if __name__ == "__main__":
    # Quick test
    test_questions = [
        "What is the Body without Organs?",
        "How does schizoanalysis work?",
        "What's the relationship between desire and production?",
        "How does Deleuzian difference differ from Hegelian contradiction?",
        "What would Deleuze say about large language models?",
        "Where does the concept of the rhizome come from?",
        "How does the concept of becoming apply to AI?",
        "What's wrong with psychoanalysis according to Deleuze?",
    ]
    
    for q in test_questions:
        result = classify_question(q)
        print(f"Q: {q}")
        print(f"   Type: {result.question_type.value}, Confidence: {result.confidence:.2f}")
        print(f"   Creative: {result.requires_creativity}, Pattern: {result.matched_pattern}")
        print(f"   Methods: {get_method_patterns_for_type(result.question_type)}")
        print()
