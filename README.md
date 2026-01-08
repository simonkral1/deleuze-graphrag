# Deleuze GraphRAG Pipeline

This repo contains the assets needed to turn the Deleuze corpus (books + seminars) into a rhizomatic
GraphRAG project powered by Cohere `embed-v4.0`. The flow has four stages:

1. **Corpus preparation** – convert all PDFs into cleaned text files plus Cohere-ready chunks.
2. **Prompt tuning** – build a bilingual extraction prompt that respects Deleuzian ontology.
3. **GraphRAG indexing** – ingest the cleaned corpus with Cohere embeddings.
4. **Nomad agent** – route questions between the GraphRAG concept map and the quote-level vector index.

## 1. Prepare the corpus

Install the Python dependencies (requires `pdftotext` from poppler) and run the extractor:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_corpus.py \
  --source-dirs pdf_books seminars \
  --output-root deleuze_corpus \
  --chunk-tokens 1200 \
  --chunk-overlap 0.15
```

Artifacts:

- `graphrag_project/input/data/*.txt` — cleaned documents for GraphRAG ingestion.
- `deleuze_corpus/metadata/manifest.jsonl` — metadata for each document.
- `deleuze_corpus/chunks/chunks.jsonl` — chunked payload for Cohere embeddings / vector DB.

## 2. Prompt tuning (auto schema discovery)

```
export COHERE_API_KEY=...
export HYPERBOLIC_API_KEY=...
graphrag prompt-tune \
  --root ./graphrag_project \
  --domain "Deleuzian Philosophy and Schizoanalysis" \
  --selection-method random \
  --limit 50 \
  --language English
```

Review the generated prompt at `graphrag_project/prompts/entity_extraction.txt` and keep the bilingual
instructions above to ensure FR/EN alignment. Add any fixed labels (Concept, Persona, Assemblage) if the
auto-tuner drifts.

## 3. GraphRAG ingest (with OpenAI embeddings + DeepSeek chat)

`graphrag_project/settings.yaml` now points to OpenAI embeddings for GraphRAG internals and DeepSeek V3 (Hyperbolic) for summaries/prompt tuning:

```yaml
embeddings:
  llm:
    type: openai_embedding
    api_key: ${OPENAI_API_KEY}
    api_base: https://api.openai.com/v1
    model: text-embedding-3-large
    dimensions: 3072
models:
  default_chat_model:
    type: openai_chat
    api_key: ${HYPERBOLIC_API_KEY}
    api_base: https://api.hyperbolic.xyz/v1
    model: deepseek-ai/DeepSeek-V3
    temperature: 0.2
```

Ingest:

```
export OPENAI_API_KEY=...
export HYPERBOLIC_API_KEY=...
graphrag index --root ./graphrag_project
```

The pipeline writes graph + community parquet files under `graphrag_project/output`.

## 4. Build the quote-level vector store

Use the Cohere chunks JSONL to populate a vector DB (Chroma). This ensures we can cite exact passages:

```bash
export COHERE_API_KEY=...
python scripts/build_quote_index.py \
  --chunks-path deleuze_corpus/chunks/chunks.jsonl \
  --persist-dir vector_store \
  --collection deleuze_quotes \
  --batch-size 64
```

This will create a persistent Chroma DB at `./vector_store` containing the embeddings from Cohere
`embed-v4.0`.

## 5. Nomad agent

`agents/nomad_agent.py` wires together:

- Graph queries (reading the GraphRAG community parquet exports).
- Quote retrieval (vector DB powered by Cohere `embed-v4.0`).

Run:

```
COHERE_API_KEY=... \
OPENAI_API_KEY=... \
python agents/nomad_agent.py --question "How do the wasp and the orchid illustrate deterritorialization?"
```

The agent decides whether to call the GraphRAG map, the quote finder, or both.

### Qdrant option for the quote index

If you prefer Qdrant instead of Chroma, set `QDRANT_URL` and `QDRANT_API_KEY` in `.env` and adapt the quote index builder to write/read from your Qdrant collection.
