# Retrieval code and generation of the augmented prompt fed to the LLM.
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import chromadb
import re
import pandas as pd
from utils import load_index_meta
import os
from dotenv import load_dotenv

load_dotenv()
print("CWD:", os.getcwd())
print("GOOGLE_API_KEY present?:", bool(os.getenv("GOOGLE_API_KEY")))
print("GOOGLE_API_KEY length:", len(os.getenv("GOOGLE_API_KEY") or ""))
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# Words that suggest a message is a pronoun-laden fragment referencing prior context.
FRAGMENT_PRONOUNS = {'it', 'that', 'he', 'she', 'they', 'this', 'those', 'them'}

# Single-word messages that carry no standalone financial meaning.
STANDALONE_AFFIRMATIONS = {'yes', 'no', 'yeah', 'ok', 'okay', 'sure', 'nope', 'yep'}

# Keywords that indicate a clearly self-contained financial question.
FINANCIAL_KEYWORDS = {
    'profit', 'loss', 'revenue', 'income', 'expense', 'expenses', 'spend',
    'spending', 'month', 'monthly', 'quarter', 'quarterly', 'week', 'weekly',
    'afford', 'cashflow', 'cash', 'trend', 'category', 'categories',
    'salary', 'hire', 'budget', 'cost', 'costs', 'sales', 'net', 'p&l',
    'invoice', 'tax', 'vat', 'employee', 'payroll', 'advertising',
}


def is_fragment_or_ambiguous(user_message: str) -> bool:
    """
    Returns True if the message is too short or ambiguous to stand alone as a
    financial query — i.e. it needs prior conversation context to be understood.

    Logic (in order):
    1. Single-word affirmation/negation → True
    2. More than 6 words → clearly standalone → False
    3. Contains a financial keyword → clearly standalone → False
    4. Fewer than 4 words AND contains a referential pronoun → True (fragment)
    5. Otherwise → False
    """
    words = user_message.strip().lower().split()

    # Rule 1: bare affirmation with no content
    if len(words) == 1 and words[0] in STANDALONE_AFFIRMATIONS:
        return True

    # Rule 2: long enough to be self-contained
    if len(words) > 6:
        return False

    # Rule 3: contains a financial domain keyword — treat as standalone
    if FINANCIAL_KEYWORDS.intersection(set(words)):
        return False

    # Rule 4: short fragment with a referential pronoun
    if len(words) < 4 and FRAGMENT_PRONOUNS.intersection(set(words)):
        return True

    return False


# ── LLM & chains ──────────────────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    streaming=True
)

# Rewrites short/ambiguous follow-ups into fully self-contained questions.
rewriter_prompt = ChatPromptTemplate.from_template(
    "You are a query resolver. Given a conversation history and a short follow-up "
    "message, rewrite the follow-up as a fully self-contained question by resolving "
    "any pronouns or references to earlier turns. "
    "Return ONLY the rewritten question — no explanation, no punctuation changes beyond "
    "what is necessary.\n\n"
    "Conversation History:\n{history}\n\n"
    "Follow-up Message: {query}\n\n"
    "Rewritten Question:"
)
rewriter_chain = rewriter_prompt | llm | StrOutputParser()

system_prompt = """
You are a financial advisor with expertise in behavioral psychology and financial habits.

Rules:
- Use ONLY the Financial Context provided. If the context does not contain the needed period or details, ask a brief follow-up question.
- Be specific, practical, and data-backed. Use £ amounts when available.
- If asked "can I afford X", estimate affordability using income, expenses, and net P&L in the context, and suggest strategies.
"""

# {history} is an optional block — pass an empty string when there is no prior conversation.
prompt_template = ChatPromptTemplate.from_template(
    system_prompt + """
{history}
Financial Context:
{context}

User Query: {query}

Advice:"""
)

chain = prompt_template | llm | StrOutputParser()


# ── History helpers ────────────────────────────────────────────────────────────

def format_history_for_rewriter(chat_history: list, last_n: int = 4) -> str:
    """Formats the most recent `last_n` messages as 'Role: content' lines."""
    recent = chat_history[-last_n:]
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def format_history_for_prompt(chat_history: list, last_n: int = 8) -> str:
    """
    Returns a labelled 'Conversation History:' block for the main prompt,
    or an empty string when there is no prior history.
    """
    if not chat_history:
        return ""
    recent = chat_history[-last_n:]
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "Conversation History:\n" + "\n".join(lines) + "\n"


def resolve_query_with_history(user_query: str, chat_history: list) -> str:
    """
    If `user_query` is a fragment or ambiguous reference (per `is_fragment_or_ambiguous`),
    uses the LLM rewriter to produce a self-contained question using the last 4 turns
    of `chat_history` as context.  Otherwise returns `user_query` unchanged.

    Always prints both the original and resolved queries for debugging.
    """
    print(f"[resolve] Original query : {user_query!r}")

    if not chat_history or not is_fragment_or_ambiguous(user_query):
        print("[resolve] Resolved query  : (unchanged — standalone or no history)")
        return user_query

    history_text = format_history_for_rewriter(chat_history, last_n=4)
    resolved_query = rewriter_chain.invoke({"history": history_text, "query": user_query}).strip()

    print(f"[resolve] Resolved query  : {resolved_query!r}")
    return resolved_query


# ── Period / granularity helpers ───────────────────────────────────────────────

def infer_granularity(user_query: str) -> str:
    q = user_query.lower()

    if 'quarter' in q or re.search(r"\bq[1-4]\b", q):
        return 'quarterly'

    if 'weekly' in q or 'week' in q or 'last 7 days' in q:
        return 'weekly'

    if 'month' in q or 'monthly' in q or re.search(r"\b20\d{2}-\d{2}\b", q):
        return 'monthly'

    return 'monthly'


def normalize_year(y: int) -> int:
    # 00–79 -> 2000s, 80–99 -> 1900s
    if y < 100:
        return 2000 + y if y <= 79 else 1900 + y
    return y


def extract_month_period(user_query: str) -> str | None:
    q = user_query.strip().lower()

    # 1) YYYY-MM
    m = re.search(r"\b(20\d{2})-(0?[1-9]|1[0-2])\b", q)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"

    # 2) MM/YYYY or MM/YY
    m = re.search(r"\b(0?[1-9]|1[0-2])/(20\d{2}|\d{2})\b", q)
    if m:
        return f"{normalize_year(int(m.group(2))):04d}-{int(m.group(1)):02d}"

    # 3) MonthName + Year (Jan26, Jan 2026, January '26)
    m = re.search(r"\b([a-z]{3,9})[\s\-']*(20\d{2}|\d{2})\b", q)
    if m and m.group(1) in MONTHS:
        return f"{normalize_year(int(m.group(2))):04d}-{MONTHS[m.group(1)]:02d}"

    return None


def extract_quarter_period(user_query: str) -> str | None:
    q = user_query.strip().lower()
    m = re.search(r"\b(20\d{2})\s*q([1-4])\b", q) or re.search(r"\bq([1-4])\s*(20\d{2})\b", q)
    if not m:
        return None

    if m.re.pattern.startswith("\\b(20"):
        year, quarter = int(m.group(1)), int(m.group(2))
    else:
        quarter, year = int(m.group(1)), int(m.group(2))

    return f"{year}Q{quarter}"


def extract_single_period(user_query: str, granularity: str) -> str | None:
    if granularity == "monthly":
        return extract_month_period(user_query)
    if granularity == "quarterly":
        return extract_quarter_period(user_query)
    return None


def extract_last_n(user_query: str) -> tuple[str, int] | None:
    q = user_query.lower()
    m = re.search(r"\blast\s+(\d+)\s+(months|month)\b", q)
    if m:
        return ("monthly", int(m.group(1)))
    m = re.search(r"\blast\s+(\d+)\s+(quarters|quarter)\b", q)
    if m:
        return ("quarterly", int(m.group(1)))
    return None


def extract_time_window(
    user_query: str, granularity: str, latest_date_iso: str
) -> dict[str, str] | None:
    """
    Extract a time window (start, end) from the query for use as a ChromaDB
    metadata filter. Used when retrieval is semantic so that e.g. "best
    performing months in last 2 years" still restricts to that window.

    Returns None if no window can be inferred. Period format matches granularity:
    YYYY-MM for monthly, YYYYQ# for quarterly.
    """
    q = user_query.lower().strip()

    # "last N years" / "past N years"
    m = re.search(r"\b(?:last|past)\s+(\d+)\s+years?\b", q)
    if m:
        n_years = int(m.group(1))
        if granularity == "monthly":
            periods = last_n_months(latest_date_iso, n_years * 12)
        else:
            periods = last_n_quarters(latest_date_iso, n_years * 4)
        if not periods:
            return None
        return {"start": periods[0], "end": periods[-1]}

    # "last year" / "past year" (no number)
    if re.search(r"\b(?:last|past)\s+year\b", q):
        if granularity == "monthly":
            periods = last_n_months(latest_date_iso, 12)
        else:
            periods = last_n_quarters(latest_date_iso, 4)
        if not periods:
            return None
        return {"start": periods[0], "end": periods[-1]}

    # "in 2024" / "during 2024"
    m = re.search(r"\b(?:in|during)\s+(20\d{2})\b", q)
    if m:
        year = int(m.group(1))
        if granularity == "monthly":
            return {"start": f"{year}-01", "end": f"{year}-12"}
        return {"start": f"{year}Q1", "end": f"{year}Q4"}

    # "from 2023 to 2024" / "between 2023 and 2024"
    m = re.search(r"\b(?:from|between)\s+(20\d{2})\s+(?:to|and)\s+(20\d{2})\b", q)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        if y1 > y2:
            y1, y2 = y2, y1
        if granularity == "monthly":
            return {"start": f"{y1}-01", "end": f"{y2}-12"}
        return {"start": f"{y1}Q1", "end": f"{y2}Q4"}

    return None


def extract_period(user_query: str):
    m = re.search(r"\b(20\d{2}-\d{2})\b", user_query)
    if m:
        return m.group(1)
    q = re.search(r"\b(20\d{2})\s*q([1-4])\b", user_query.lower())
    if q:
        return f"{q.group(1)}Q{q.group(2)}"
    return None


# ── Retrieval ──────────────────────────────────────────────────────────────────

def last_n_months(latest_date_iso: str, n: int) -> list[str]:
    latest = pd.Timestamp(latest_date_iso).to_period("M")
    return [str(latest - i) for i in reversed(range(n))]


def last_n_quarters(latest_date_iso: str, n: int) -> list[str]:
    latest = pd.Timestamp(latest_date_iso).to_period("Q")
    return [str(latest - i) for i in reversed(range(n))]


def periods_in_range(start: str, end: str, granularity: str) -> list[str]:
    """Expand start/end period strings into a list of period strings. ChromaDB
    only supports $gte/$lte on numbers, so we filter by $in with this list."""
    if granularity == "monthly":
        r = pd.period_range(start=start, end=end, freq="M")
    else:
        r = pd.period_range(start=start, end=end, freq="Q")
    return [str(p) for p in r]


def get_chunks_by_periods(collection, chunk_type: str, periods: list[str]):
    docs_metas = []
    for p in periods:
        got = collection.get(
            where={"$and": [
                {"chunk_type": chunk_type},
                {"period": p}
            ]}
        )
        for d, m in zip(got.get("documents", []), got.get("metadatas", [])):
            docs_metas.append((d, m))
    return docs_metas


def retrive(user_query, embedding_model, collection, top_k: int = 5):
    meta = load_index_meta()
    latest_date = meta['latest_date']

    time_period = infer_granularity(user_query=user_query)

    # 1) Deterministic: last N months/quarters
    req = extract_last_n(user_query=user_query)
    if req:
        g, n = req
        periods = last_n_months(latest_date, n) if g == 'monthly' else last_n_quarters(latest_date, n)
        docs_metas = get_chunks_by_periods(collection, chunk_type=g, periods=periods)
        print(f'Deterministic Retrieval: last N {periods}')
        return docs_metas, {
            "mode": "exact_window",
            "latest_date": latest_date,
            "periods": periods,
            "time_window": None,
        }

    # 2) Deterministic: exact named period
    period = extract_single_period(user_query=user_query, granularity=time_period)
    if period:
        docs_metas = get_chunks_by_periods(collection, time_period, [period])
        print(f'Deterministic Retrieval: exact period {period}')
        return docs_metas, {
            "mode": "exact_period",
            "latest_date": latest_date,
            "periods": [period],
            "time_window": None,
        }

    # 3) Semantic similarity search (with optional time window filter)
    time_window = extract_time_window(user_query, time_period, latest_date)
    where_clause = {"chunk_type": time_period}
    if time_window:
        period_list = periods_in_range(
            time_window["start"], time_window["end"], time_period
        )
        where_clause = {
            "$and": [
                {"chunk_type": time_period},
                {"period": {"$in": period_list}},
            ]
        }
        print(f"Semantic Similarity Retrieval (time_window: {time_window['start']} → {time_window['end']})")
    else:
        print("Semantic Similarity Retrieval")

    query_embedding = embedding_model.encode(user_query)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where=where_clause,
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    docs_metas = list(zip(docs, metas))

    extra_info = {
        "mode": "semantic",
        "latest_date": latest_date,
        "periods": [],
        "time_window": time_window,
    }
    return docs_metas, extra_info


def format_context_from_pairs(docs_metas) -> str:
    blocks = [
        f"[{meta.get('chunk_type')} | {meta.get('period')}]\n{doc}".strip()
        for doc, meta in docs_metas
    ]
    return "\n\n---\n\n".join(blocks)


# ── Public API ─────────────────────────────────────────────────────────────────

def query_financial_chatbot(user_query, embedding_model, collection, chat_history: list = []):
    """
    Non-streaming query.  Resolves ambiguous follow-ups using chat_history before
    retrieval, and injects formatted history into the LLM prompt.
    """
    resolved_query = resolve_query_with_history(user_query, chat_history)

    docs_metas, extra_info = retrive(resolved_query, embedding_model, collection)

    context = format_context_from_pairs(docs_metas)
    context = f"Data time reference: latest transaction date = {extra_info['latest_date']}\n\n" + context

    history_block = format_history_for_prompt(chat_history)

    response = chain.invoke({"context": context, "query": resolved_query, "history": history_block})
    return response, docs_metas, extra_info


def stream_financial_chatbot(user_query, embedding_model, collection, chat_history: list = []):
    """
    Streaming query.  Same resolution and history injection as query_financial_chatbot.
    Yields (token, docs_metas, extra_info) on each streamed chunk.
    """
    resolved_query = resolve_query_with_history(user_query, chat_history)

    docs_metas, extra_info = retrive(resolved_query, embedding_model, collection)

    context = format_context_from_pairs(docs_metas)
    context = f"Data time reference: latest transaction date = {extra_info['latest_date']}\n\n" + context

    history_block = format_history_for_prompt(chat_history)

    for token in chain.stream({"context": context, "query": resolved_query, "history": history_block}):
        yield token, docs_metas, extra_info


if __name__ == "__main__":
    print("rag.py imports ✓")
    print("Chain initialised ✓")
