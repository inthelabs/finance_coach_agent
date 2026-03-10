# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Build vector DB from synthetic data (run once, or to reset)
python ingest.py

# Launch the Streamlit app
streamlit run app.py
```

The app auto-generates synthetic data and rebuilds the vector DB on startup when `RESET_DB = True` in `app.py`. Toggle that flag to `False` to skip regeneration on subsequent runs.

## Environment

Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_google_api_key_here
```

## Architecture

This is a RAG pipeline that answers natural language financial queries using pre-aggregated transaction data stored in ChromaDB.

**Two-phase pipeline:**
1. **Ingestion** (`ingest.py` → `utils.py`): Generates synthetic transactions, aggregates them into weekly/monthly/quarterly text chunks, embeds via `all-MiniLM-L6-v2` (SentenceTransformers), and persists to ChromaDB at `./chroma/`. Also saves `./chroma/index_meta.json` with the latest transaction date and available period lists.
2. **Query** (`rag.py`): Embeds the user query, retrieves relevant chunks, and streams a response via `gemini-2.0-flash`.

**Retrieval strategy (in priority order, `rag.py:retrive()`):**
1. **Exact window** — "last N months/quarters" → fetch those specific periods by metadata filter
2. **Exact period** — named month/quarter → fetch that period by metadata filter
3. **Semantic search** — fallback embedding similarity search filtered by inferred `chunk_type`

Granularity (`weekly`/`monthly`/`quarterly`) is inferred from keywords in the query. Defaults to `monthly`.

**ChromaDB schema:** Single collection `financial_data` with `chunk_type` and `period` metadata fields on every document. Period format: `YYYY-MM` for monthly, `YYYYQ#` for quarterly, pandas `W` period string for weekly.

**Key files:**
- `utils.py` — data generation, chunking logic, embedding, ChromaDB writes, `index_meta.json` management
- `rag.py` — retrieval logic, period extraction regexes, LangChain chain, streaming
- `app.py` — Streamlit UI, session state for collection/embedding model, streaming display
- `ingest.py` — thin CLI wrapper around `utils.ingest_data()`

**`RESET_DB` flag in `app.py`:** When `True`, regenerates synthetic data (100k transactions, 15 years) and rebuilds the vector DB on every app startup. Set to `False` after the initial build to avoid the ~minute-long startup delay.

## Deployment

- Platform: Streamlit Cloud
- Repo: https://github.com/inthelabs/finance_coach_agent
- Branch: main
- Entry point: app.py

## Git Branch Naming
```
feature/<description>     # New capabilities
fix/<description>         # Bug fixes
refactor/<description>    # Restructuring, no new behaviour
eval/<description>        # Evaluation and testing
```

### Workflow
```bash
git checkout main && git pull origin main
git checkout -b feature/your-feature-name
# make changes, test locally
git add 
git commit -m "Step X: Clear description of what changed and why"
git push origin feature/your-feature-name
# Only merge to main after testing
```
