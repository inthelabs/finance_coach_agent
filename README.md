# Financial Coach: RAG-Powered Business Financial Advisor

A conversational AI financial coach built with **LangChain**, **ChromaDB**, and **Google Gemini**, powered by a **RAG pipeline** that ingests (synthetic) business transaction data and answers natural language financial queries with grounded, context-aware advice.

---

## Project Structure
```
financial-coach/
├── app.py                  # Streamlit UI — main application
├── rag.py                  # Retrieval chain + LLM generation
├── utils.py                # Core utilities: chunking, embedding, vector store
├── ingest.py               # CLI script for data ingestion
├── .env                    # Environment variables (API keys) — not committed
├── requirements.txt        # Python dependencies
└── README.md
```

---

## How It Works

This project demonstrates a full **RAG (Retrieval-Augmented Generation)** pipeline applied to business financial data:
```
Ingestion Pipeline (runs once):
  CSV Transactions → Chunking Strategy (Weekly/Monthly/Quarterly)
  → SentenceTransformer Embeddings → ChromaDB (Persistent)

Query Pipeline (runs per query):
  User Question → Embed Query → Semantic Search → Top-K Chunks
  → Augmented Prompt → Google Gemini → Financial Advice
```

### Chunking Strategy
Rather than embedding individual transactions, data is pre-aggregated into three chunk types stored in a single ChromaDB collection with `chunk_type` metadata:

| Chunk Type | Fields | Best For |
|------------|--------|----------|
| `weekly` | income, expenses, P&L, WoW delta, top 3 categories | Granular spending queries |
| `monthly` | income, expenses, P&L, employee spend, 3-month trend | Profitability & cashflow |
| `quarterly` | income, expenses, P&L, QoQ delta, best month | Strategic performance review |

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/financial-coach.git
cd financial-coach
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_google_api_key_here
```
Get a free API key at [aistudio.google.com](https://aistudio.google.com)

---

## Usage

### Step 1 — Generate synthetic data + build vector DB (run once)
```bash
python ingest.py
```
This generates realistic synthetic business transactions (Shopify income + business expenses across 2 years), chunks them, embeds them, and persists to ChromaDB locally.

### Step 2 — Launch the app
```bash
streamlit run app.py
```

### Example queries
- *"What was my most profitable month?"*
- *"Can I afford to hire a new employee?"*
- *"How did Q3 compare to Q2?"*
- *"What are my biggest expense categories?"*
- *"What is my 3-month cashflow trend?"*

---

## Architecture Decisions

**Why pre-aggregated chunks instead of raw transactions?**
LLMs are not calculators. Pre-computing weekly/monthly/quarterly summaries means the retriever pulls precise financial summaries rather than raw rows, and the LLM focuses on reasoning and advice rather than arithmetic.

**Why ChromaDB with persistent storage?**
Embeddings are computed once and saved to disk. The app loads the existing collection on startup rather than re-embedding on every run.

**Why one collection with metadata over multiple collections?**
Simplicity for an MVP. `chunk_type` metadata allows filtering if needed, while keeping the retrieval logic in one place.

---

## Extending This Project

- Add metadata filtering to route queries to the right chunk type
- Connect to a real bank API (Plaid, TrueLayer) instead of synthetic data
- Add conversation memory so the advisor maintains context across questions
- Swap Google Gemini for a local model via Ollama for fully offline use