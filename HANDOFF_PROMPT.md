# Handoff Prompt: Implementing Triple-Embedding Architecture for Deleuzian Thinking Machine

## Context
You are continuing work on transforming a Deleuze corpus RAG system from a passive "knowledge base" into an active "Deleuzian thinking machine" that performs philosophy rather than summarizing it.

## Current State

### What Exists (Working)
1. **Tier 1 - Conceptual Map Embeddings (GraphRAG)**: ✅
   - Location: `graphrag_project/output/community_reports.parquet`
   - Model: OpenAI text-embedding-3-large (3072-dim)
   - Purpose: Graph-level concept networks

2. **Tier 2 - Quote-Level Embeddings (Chroma)**: ✅
   - Location: `deleuze_corpus/vectors.parquet` (11,549 chunks, 1536-dim vectors)
   - Also in: `vector_store/` (Chroma database with collection "deleuze_quotes")
   - Model: Cohere embed-v4.0
   - Content: 479 documents from major Deleuze works (English + French)
   - Top works: Rhizome (594 chunks), Mille Plateaux (314), A Thousand Plateaus (297), Anti-Oedipus (191), Difference and Repetition (179)

3. **Current Agent Architecture**:
   - File: `agents/nomad_agent.py`
   - Uses: OpenAI GPT-4o-mini for synthesis (temperature 0.2)
   - Flow: Question → Retrieve communities + quotes → Synthesize answer
   - Problem: Claude/LLM is passive synthesizer, not active philosopher

4. **Configuration**:
   - File: `graphrag_project/settings.yaml`
   - Current synthesis model: gpt-5.2-2025-12-11 (OpenAI)
   - Embedding model: text-embedding-3-large (3072-dim)

### What's Missing (Your Task)
**Tier 3 - Method Embeddings**: ⭐ This is the critical addition

## Objective
Implement a three-tier embedding system where:
- **Tier 1** (existing): Provides structural scaffolding (concept maps)
- **Tier 2** (existing): Provides exact citations for grounding
- **Tier 3** (NEW): Provides Deleuze's methodological patterns for HOW to think

Then rebuild the agent to use Claude Opus 4.5 with extended thinking to PERFORM Deleuzian philosophy.

## Implementation Plan

### Phase 1: Extract Method Patterns from Corpus

#### 1.1 Create Method Pattern Extractor
**File to create**: `scripts/extract_method_patterns.py`

**Purpose**: Identify and categorize passages that show Deleuze's METHOD, not just content.

**Categories to extract**:
```python
method_patterns = {
    "concept_creation": [
        # Passages where Deleuze CREATES concepts (not discusses them)
        # Examples: Introduction of BwO in Anti-Oedipus, defining the rhizome
        # Markers: "We call this...", "Let us call...", "X is defined by..."
    ],
    "critique_patterns": [
        # How he dismantles other theories (Hegel, psychoanalysis, etc.)
        # Markers: "The problem with X is not...", "It's not a matter of..."
    ],
    "example_usage": [
        # Concrete examples that operationalize concepts
        # Famous ones: wasp-orchid, little Hans, Virginia Woolf becoming-imperceptible
        # Pattern: Concept → concrete example → conceptual machine
    ],
    "problem_posing": [
        # How he reframes questions
        # Pattern: "Not 'what is X?' but 'how does X work?'"
        # "The question is not... but rather..."
    ],
    "argumentation_moves": [
        # Disjunctive synthesis, assemblage-building, plateau structures
        # How arguments unfold non-dialectically
    ],
    "stylistic_signatures": [
        # Paradox resolution, immanence vs transcendence shifts
        # "It is AND/OR not EITHER/OR" type constructions
    ]
}
```

**Implementation approach**:
1. Load `deleuze_corpus/vectors.parquet`
2. Use LLM to classify each chunk by pattern type (could use Claude Haiku for cost-efficiency)
3. Extract metadata: `pattern_type`, `method_category`, `philosophical_move`
4. Save to: `deleuze_corpus/method_patterns.parquet`

**Sample prompt for classification**:
```python
classification_prompt = """You are analyzing a passage from Deleuze's work.
Identify which methodological pattern(s) it exemplifies:

1. CONCEPT_CREATION: Deleuze introduces/defines a new concept
2. CRITIQUE: Deleuze dismantles another philosophical position
3. EXAMPLE_USAGE: Concrete example operationalizes abstract concept
4. PROBLEM_REFRAMING: Question is restated in Deleuzian terms
5. ARGUMENTATION: Shows how Deleuze builds non-dialectical arguments
6. STYLISTIC: Exemplifies paradox, disjunction, or immanence moves
7. NONE: Just explanatory/descriptive content

Passage:
{text}

Respond with JSON: {{"patterns": ["PATTERN1", "PATTERN2"], "confidence": 0.0-1.0, "key_phrase": "..."}}
"""
```

#### 1.2 Build Pattern Recognition Heuristics
Add regex/keyword-based shortcuts to reduce LLM calls:

**Concept creation markers**:
- "We call this", "Let us call", "We term", "is defined by", "What is X? X is"

**Critique markers**:
- "The problem with", "It is not", "not a matter of", "opposed to", "against"

**Example markers**:
- "For example", "Take the case of", "Consider", "Let's take"

**Problem-reframing markers**:
- "The question is not", "not 'what' but 'how'", "It's not about... but"

### Phase 2: Embed Method Patterns (Tier 3)

#### 2.1 Create Method Embeddings
**File to create**: `scripts/create_method_embeddings.py`

**Purpose**: Embed method patterns with special preprocessing to capture HOW not just WHAT.

**Model choice**:
- Option A: OpenAI text-embedding-3-large (consistency with Tier 1)
- Option B: Voyage AI voyage-3 (specialized for semantic nuance)

**Preprocessing strategy**:
```python
def preprocess_method_pattern(text: str, pattern_type: str) -> str:
    """Augment text with methodological context"""
    prefixes = {
        "concept_creation": "Deleuze creates a concept: ",
        "critique": "Deleuze critiques: ",
        "example_usage": "Deleuze uses this example: ",
        "problem_posing": "Deleuze reframes the question: ",
    }
    return f"{prefixes.get(pattern_type, '')}{text}"
```

**Storage**: Save to new Chroma collection `deleuze_methods` or add metadata to existing collection.

#### 2.2 Create Method Index
**File to create**: `agents/method_index.py`

```python
class MethodPatternIndex:
    """Index for retrieving Deleuze's methodological patterns"""

    def __init__(self, persist_dir: Path, collection: str = "deleuze_methods"):
        # Initialize Chroma client with method patterns
        pass

    def search_by_question_type(self, question: str, question_type: str, top_k: int = 3) -> List[dict]:
        """Retrieve method patterns relevant to question type"""
        # Use hybrid: question similarity + pattern type filter
        pass
```

### Phase 3: Question Classification System

#### 3.1 Create Question Classifier
**File to create**: `agents/question_classifier.py`

**Question types**:
```python
QUESTION_TYPES = {
    "definitional": "What is X?",
    "relational": "How does X relate to Y?",
    "operational": "How does X work?",
    "genealogical": "Where does X come from?",
    "comparative": "What's the difference between X and Y?",
    "applied": "How does X apply to [concrete case]?",
    "creative": "What would Deleuze say about [new topic]?"
}
```

**Implementation**:
```python
def classify_question(question: str) -> dict:
    """
    Returns: {
        "type": "operational",
        "confidence": 0.85,
        "requires_creativity": False  # vs retrieval-only
    }
    """
    # Use Claude Haiku for fast classification
    # Or rule-based: "what is" → definitional, "how does" → operational
```

### Phase 4: Enhanced Nomad Agent

#### 4.1 Modify `agents/nomad_agent.py`

**Key changes**:

1. **Add Anthropic client**:
```python
from anthropic import Anthropic

class ResponseSynthesizer:
    def __init__(self, model: str = "claude-opus-4-5-20251101"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY required")
        self.client = Anthropic(api_key=api_key)
        self.model = model
```

2. **Add method retrieval to workflow**:
```python
def main():
    # ... existing code ...

    # NEW: Classify question
    question_type = classify_question(args.question)

    # NEW: Retrieve method patterns
    method_index = MethodPatternIndex(args.vector_store)
    method_patterns = method_index.search_by_question_type(
        args.question,
        question_type["type"],
        top_k=3
    )

    # Enhanced synthesis
    synthesizer = ResponseSynthesizer(model=args.anthropic_model)
    answer = synthesizer.generate(
        args.question,
        communities,
        quotes,
        method_patterns,  # NEW
        question_type     # NEW
    )
```

3. **Rebuild system prompt**:
```python
def build_enhanced_prompt(self, question: str, communities: List[dict],
                         quotes: List[dict], method_patterns: List[dict],
                         question_type: dict) -> List[dict]:

    method_examples = "\n\n".join([
        f"**Method Example {i+1}** ({p['metadata']['pattern_type']}):\n{p['text']}"
        for i, p in enumerate(method_patterns)
    ])

    community_context = "\n".join([
        f"- [{c['title']}]: {c['summary']}"
        for c in communities
    ])

    quote_sources = "\n".join([
        f"- [{q['metadata']['doc_id']}#{q['id']}]: {q['text']}"
        for q in quotes
    ])

    system_prompt = f"""You are Gilles Deleuze. You do not summarize philosophy—you DO philosophy.

This is a {question_type['type']} question. Your methodological approach:

{method_examples}

Core Deleuzian Principles:
- Think in assemblages, not categories
- Use concrete examples as conceptual machines
- Avoid dialectical synthesis; prefer disjunctive synthesis (AND/OR not EITHER/OR)
- Build rhizomatic connections, not arborescent hierarchies
- Create concepts to solve problems, don't apply pre-existing frameworks
- Ask "How does it work?" not "What is it?"

When answering:
1. REFRAME the question if it's posed in non-Deleuzian terms
2. BUILD concepts through assemblages and examples
3. AVOID summary language ("Deleuze argues that...")
4. PERFORM the philosophical move, don't describe it
5. Ground responses in your own words (cite sources as [doc_id#chunk_id])

Conceptual assemblages available to you:
{community_context}

Your own words to draw from:
{quote_sources}

Now respond AS Deleuze, not ABOUT Deleuze."""

    user_prompt = f"""Question: {question}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
```

4. **Use extended thinking mode**:
```python
def generate(self, question: str, communities, quotes, method_patterns, question_type) -> str:
    messages = self.build_enhanced_prompt(
        question, communities, quotes, method_patterns, question_type
    )

    response = self.client.messages.create(
        model=self.model,
        max_tokens=4096,
        temperature=0.7,  # Higher for creative concept-building
        thinking={
            "type": "enabled",
            "budget_tokens": 10000  # Let Claude think through Deleuzian logic
        },
        messages=messages
    )

    # Extract answer (thinking is separate from content)
    return response.content[0].text
```

#### 4.2 Add CLI arguments
```python
parser.add_argument(
    "--anthropic-model",
    default="claude-opus-4-5-20251101",
    help="Anthropic model for synthesis (supports extended thinking)"
)
parser.add_argument(
    "--synthesis-temperature",
    type=float,
    default=0.7,
    help="Temperature for creative philosophical synthesis"
)
```

### Phase 5: Few-Shot Examples

#### 5.1 Extract Q&A Patterns from Seminars
**File to create**: `scripts/extract_qa_examples.py`

**Purpose**: Find actual Deleuze Q&A exchanges from seminar transcripts.

**Strategy**:
1. Search `deleuze_corpus/vectors.parquet` for seminar documents
2. Identify Q&A patterns (look for "Question:", "Q:", "Student:", etc.)
3. Extract question-answer pairs
4. Embed these as templates

**Storage**: `deleuze_corpus/qa_examples.json`

```json
{
  "definitional": [
    {
      "question": "What is the relationship between desire and production?",
      "answer": "[Actual Deleuze response from Anti-Oedipus]",
      "source": "anti-oedipus#chunk_142"
    }
  ],
  "operational": [
    {
      "question": "How does the Body without Organs function?",
      "answer": "[Response from seminars]",
      "source": "seminar-1980-05-20#chunk_23"
    }
  ]
}
```

#### 5.2 Integrate into System Prompt
Add to `build_enhanced_prompt`:

```python
# Load Q&A examples
qa_examples = load_qa_examples(question_type['type'])
few_shot_section = "\n\n".join([
    f"Q: {ex['question']}\nA: {ex['answer'][:500]}..."
    for ex in qa_examples[:2]  # Show 2 examples
])

system_prompt += f"""

Example of your argumentative style in {question_type['type']} questions:

{few_shot_section}

Now handle the current question in this mode..."""
```

### Phase 6: Configuration Updates

#### 6.1 Update `graphrag_project/settings.yaml`
```yaml
models:
  # Keep existing for GraphRAG indexing
  default_chat_model:
    type: openai_chat
    api_key: ${OPENAI_API_KEY}
    model: gpt-5.2-2025-12-11
    temperature: 0.2
    max_tokens: 4096

  # NEW: Synthesis model for Nomad agent
  synthesis_model:
    type: anthropic_chat
    api_key: ${ANTHROPIC_API_KEY}
    model: claude-opus-4-5-20251101
    temperature: 0.7
    max_tokens: 4096
    thinking:
      enabled: true
      budget_tokens: 10000

  # NEW: Method embedding model (Tier 3)
  method_embedding_model:
    type: openai_embedding
    api_key: ${OPENAI_API_KEY}
    model: text-embedding-3-large
    dimensions: 3072

# NEW: Method pattern configuration
method_patterns:
  enabled: true
  collection: deleuze_methods
  categories:
    - concept_creation
    - critique_patterns
    - example_usage
    - problem_posing
    - argumentation_moves
    - stylistic_signatures
```

### Phase 7: Testing

#### 7.1 Test Questions by Type
Create `tests/test_enhanced_nomad.py`:

```python
test_questions = {
    "definitional": "What is the Body without Organs?",
    "operational": "How does schizoanalysis work?",
    "relational": "What's the relationship between desire and production?",
    "comparative": "How does Deleuzian difference differ from Hegelian contradiction?",
    "creative": "What would Deleuze say about large language models?"
}

# Run each through enhanced pipeline
# Compare old (passive) vs new (active) responses
```

#### 7.2 Evaluation Criteria
- Does response PERFORM philosophy vs DESCRIBE it?
- Are Deleuzian concepts created/deployed actively?
- Is the argumentation rhizomatic vs linear?
- Are citations grounded in actual quotes?
- Does extended thinking show Deleuzian reasoning process?

## File Structure Overview

```
deleuze2/
├── agents/
│   ├── nomad_agent.py          [MODIFY: Add method retrieval, switch to Claude]
│   ├── method_index.py         [CREATE: Method pattern index]
│   └── question_classifier.py  [CREATE: Question type classifier]
├── scripts/
│   ├── extract_method_patterns.py    [CREATE: Extract Tier 3 patterns]
│   ├── create_method_embeddings.py   [CREATE: Embed method patterns]
│   └── extract_qa_examples.py        [CREATE: Extract seminar Q&As]
├── deleuze_corpus/
│   ├── vectors.parquet         [EXISTS: 11,549 chunks, Tier 2]
│   ├── method_patterns.parquet [CREATE: Classified method patterns]
│   └── qa_examples.json        [CREATE: Few-shot examples]
├── vector_store/               [EXISTS: Chroma DB with deleuze_quotes]
│   └── deleuze_methods/        [CREATE: New collection for Tier 3]
├── graphrag_project/
│   ├── settings.yaml           [MODIFY: Add synthesis_model config]
│   └── output/
│       └── community_reports.parquet  [EXISTS: Tier 1]
└── tests/
    └── test_enhanced_nomad.py  [CREATE: Test suite]
```

## Priority Order

1. **Start here**: Create `scripts/extract_method_patterns.py` to classify existing chunks
2. Embed method patterns into new Chroma collection
3. Create `agents/question_classifier.py` (can be simple rule-based initially)
4. Modify `agents/nomad_agent.py` to integrate all three tiers
5. Switch to Claude Opus 4.5 with extended thinking
6. Extract few-shot Q&A examples
7. Test and compare outputs

## Environment Requirements

**API Keys needed**:
- `OPENAI_API_KEY` (existing, for embeddings)
- `COHERE_API_KEY` (existing, for Tier 2 embeddings)
- `ANTHROPIC_API_KEY` (NEW, for Claude Opus 4.5 synthesis)

**Dependencies to check** (likely already installed):
```bash
pip install anthropic>=0.40.0  # For Claude Opus 4.5 + extended thinking
pip install chromadb cohere openai pandas numpy
```

## Success Metrics

**Before (Current)**:
- "Deleuze argues that the Body without Organs is..."
- Temperature: 0.2 (conservative)
- Model: GPT-4o-mini (passive synthesizer)

**After (Target)**:
- "The Body without Organs is not an image or metaphor—it's a limit, a practice..."
- Temperature: 0.7 (creative)
- Model: Claude Opus 4.5 with 10k thinking tokens
- System actively PERFORMS Deleuzian moves (disjunctive synthesis, problem-reframing, etc.)

## Key Insight

The transformation is from:
- **Retrieval → Synthesis** (passive knowledge base)

To:
- **Method Retrieval → Conceptual Performance** (active thinking machine)

Claude shouldn't cite Deleuze—it should DO what Deleuze does.

---

## Quick Start Commands

```bash
# 1. Create method pattern extractor
touch scripts/extract_method_patterns.py

# 2. Run extraction (will take time with 11k chunks)
python scripts/extract_method_patterns.py \
  --input deleuze_corpus/vectors.parquet \
  --output deleuze_corpus/method_patterns.parquet

# 3. Create method embeddings
python scripts/create_method_embeddings.py \
  --patterns deleuze_corpus/method_patterns.parquet \
  --output vector_store/deleuze_methods

# 4. Test enhanced agent
python agents/nomad_agent.py \
  --question "How does schizoanalysis work?" \
  --anthropic-model claude-opus-4-5-20251101 \
  --synthesis-temperature 0.7

# 5. Compare outputs
python tests/test_enhanced_nomad.py
```

## Questions to Resolve

1. **Method embedding model**: Stick with text-embedding-3-large for consistency, or try Voyage-3 for semantic nuance?
2. **Question classifier**: Start with rule-based or immediately use Claude Haiku?
3. **Few-shot examples**: How many per question type? (Suggest 2-3)
4. **Method pattern storage**: Separate Chroma collection or metadata in existing collection?

## Notes

- The corpus has both English and French texts—method patterns should ideally be bilingual
- Extended thinking mode is critical—it lets Claude work through Deleuzian logic before responding
- Temperature 0.7 is intentionally higher to enable creative concept-building
- The system should feel like having a philosophical conversation with Deleuze, not querying a database about him

---

**You are now ready to implement the Deleuzian thinking machine. Start with Phase 1 (method extraction) and work sequentially through the phases. Good luck!**
