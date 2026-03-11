from __future__ import annotations

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

# Phrases that explicitly refer back to prior context even if the message is long.
REFERENTIAL_PHRASES = {
    "this period",
    "that period",
    "during this period",
    "during that period",
    "in this period",
    "in that period",
    "this month",
    "that month",
    "this quarter",
    "that quarter",
    "this year",
    "that year",
    "these months",
    "those months",
    "these quarters",
    "those quarters",
}

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
    msg = " ".join(words)

    # Rule 1: bare affirmation with no content
    if len(words) == 1 and words[0] in STANDALONE_AFFIRMATIONS:
        return True

    # Rule 2: explicit referential phrasing (even if long)
    if any(p in msg for p in REFERENTIAL_PHRASES):
        return True

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


def extract_year_reference(user_query: str) -> int | None:
    m = re.search(r"\b(20\d{2})\b", user_query)
    if not m:
        return None
    return int(m.group(1))


def extract_explicit_month_periods(user_query: str) -> list[str]:
    """
    Extract explicit month periods from queries like:
    - "pnl for nov and dec in 2025" -> ["2025-11", "2025-12"]
    - "revenue in January 2026" -> ["2026-01"]
    - "jan to mar 2024" -> ["2024-01", "2024-02", "2024-03"]
    Also supports explicit YYYY-MM strings.
    """
    q = user_query.lower()
    periods: list[str] = []

    # Explicit YYYY-MM occurrences
    for y, m in re.findall(r"\b(20\d{2})-(0?[1-9]|1[0-2])\b", q):
        periods.append(f"{int(y):04d}-{int(m):02d}")

    # Month range: "jan to mar 2024"
    range_match = re.search(
        r"\b([a-z]{3,9})\s*(?:to|\-)\s*([a-z]{3,9})\s*(?:in\s*)?(20\d{2})\b",
        q,
    )
    if range_match and range_match.group(1) in MONTHS and range_match.group(2) in MONTHS:
        y = int(range_match.group(3))
        start = f"{y:04d}-{MONTHS[range_match.group(1)]:02d}"
        end = f"{y:04d}-{MONTHS[range_match.group(2)]:02d}"
        if start <= end:
            return [str(p) for p in pd.period_range(start=start, end=end, freq="M")]

    year = extract_year_reference(q)
    if year is not None:
        # Multiple month mentions with a year: "nov and dec 2025"
        month_tokens = [tok for tok in re.findall(r"\b([a-z]{3,9})\b", q) if tok in MONTHS]
        for tok in month_tokens:
            periods.append(f"{year:04d}-{MONTHS[tok]:02d}")

    # Single month+year patterns (Jan 2026, Jan26, etc.)
    single = extract_month_period(q)
    if single:
        periods.append(single)

    return sorted(set(periods))


def extract_relative_time_window(user_query: str, granularity: str, latest_date_iso: str) -> dict[str, str] | None:
    """
    Handles relative windows not covered by extract_time_window(), e.g.
    this year / last year / year to date / this quarter / last quarter.
    """
    q = user_query.lower().strip()
    latest_ts = pd.Timestamp(latest_date_iso)
    latest_m = latest_ts.to_period("M")
    latest_q = latest_ts.to_period("Q")

    if re.search(r"\b(year to date|ytd)\b", q) or re.search(r"\bthis year\b", q):
        if granularity == "monthly":
            start = f"{latest_ts.year:04d}-01"
            end = str(latest_m)
        else:
            start = f"{latest_ts.year:04d}Q1"
            end = str(latest_q)
        return {"start": start, "end": end}

    if re.search(r"\blast year\b", q):
        y = latest_ts.year - 1
        if granularity == "monthly":
            return {"start": f"{y:04d}-01", "end": f"{y:04d}-12"}
        return {"start": f"{y:04d}Q1", "end": f"{y:04d}Q4"}

    # "compare this quarter to last quarter" (or mention both) -> include both quarters
    if ("this quarter" in q) and ("last quarter" in q):
        prev_q = latest_q - 1
        if granularity == "quarterly":
            return {"start": str(prev_q), "end": str(latest_q)}
        start = prev_q.asfreq("M", how="start")
        end = latest_q.asfreq("M", how="end")
        return {"start": str(start), "end": str(end)}

    if re.search(r"\bthis quarter\b", q):
        if granularity == "quarterly":
            return {"start": str(latest_q), "end": str(latest_q)}
        start = latest_q.asfreq("M", how="start")
        end = latest_q.asfreq("M", how="end")
        return {"start": str(start), "end": str(end)}

    if re.search(r"\blast quarter\b", q):
        prev_q = latest_q - 1
        if granularity == "quarterly":
            return {"start": str(prev_q), "end": str(prev_q)}
        start = prev_q.asfreq("M", how="start")
        end = prev_q.asfreq("M", how="end")
        return {"start": str(start), "end": str(end)}

    if re.search(r"\bthis month\b", q) and granularity == "monthly":
        return {"start": str(latest_m), "end": str(latest_m)}

    if re.search(r"\blast month\b", q) and granularity == "monthly":
        prev_m = latest_m - 1
        return {"start": str(prev_m), "end": str(prev_m)}

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


RANKING_KEYWORDS = {
    "best", "worst", "highest", "lowest", "top", "bottom", "strongest", "weakest",
    "most", "least",
}
EXPLANATION_KEYWORDS = {
    "why", "pattern", "patterns", "explain", "explanation", "drivers", "driver",
    "going on", "before", "after",
}
COMPARISON_KEYWORDS = {
    "growth", "change", "increase", "decline", "difference", "compare", "compared", "versus", "vs",
}

BREAKDOWN_KEYWORDS = {
    "by month", "monthly breakdown", "month by month", "by quarter", "quarterly breakdown",
}


def classify_intent(user_query: str) -> str:
    q = user_query.lower()
    if any(k in q for k in EXPLANATION_KEYWORDS):
        return "semantic_exploration"
    if any(k in q for k in RANKING_KEYWORDS):
        return "analytical_ranking"
    return "exact_lookup"


def metric_for_ranking(user_query: str) -> str:
    q = user_query.lower()
    if any(k in q for k in ["sales", "revenue", "income"]):
        return "total_income"
    if "margin" in q:
        return "margin"
    return "net_pl"


def compute_income_growth(docs_metas: list[tuple[str, dict]]) -> dict | None:
    if len(docs_metas) < 2:
        return None
    sorted_pairs = sorted(docs_metas, key=lambda dm: (dm[1].get("period") or ""))
    first_meta = sorted_pairs[0][1]
    last_meta = sorted_pairs[-1][1]
    a = first_meta.get("total_income")
    b = last_meta.get("total_income")
    if a is None or b is None or a == 0:
        return None
    pct = ((b - a) / a) * 100
    return {
        "from_period": first_meta.get("period"),
        "to_period": last_meta.get("period"),
        "income_from": a,
        "income_to": b,
        "income_growth_pct": pct,
    }


def _safe_pct_change(a: float, b: float) -> float | None:
    if a is None or a == 0:
        return None
    return ((b - a) / a) * 100


def detect_answer_granularity_for_comparison(user_query: str) -> tuple[str, bool]:
    """
    Returns (answer_granularity, expand_to_subperiods).
    - If the user asks for a breakdown ("by month"), expand_to_subperiods=True.
    - If comparing years only, answer_granularity='yearly'.
    - If comparing quarters, answer_granularity='quarterly'.
    - Otherwise default to 'monthly'.
    """
    q = user_query.lower()
    expand = any(k in q for k in BREAKDOWN_KEYWORDS)
    if "by month" in q or "month by month" in q or "monthly breakdown" in q:
        return ("monthly", True)
    if "by quarter" in q or "quarterly breakdown" in q:
        return ("quarterly", True)
    if re.search(r"\bbetween\s+20\d{2}\s+and\s+20\d{2}\b", q) or re.search(r"\bbetween\s+20\d{2}\s+to\s+20\d{2}\b", q):
        return ("yearly", expand)
    if re.search(r"\bbetween\s+q[1-4]\s+and\s+q[1-4]\s+20\d{2}\b", q) or re.search(r"\bbetween\s+(20\d{2})\s*q[1-4]\s+and\s+(20\d{2})\s*q[1-4]\b", q):
        return ("quarterly", expand)
    if re.search(r"\bbetween\s+([a-z]{3,9})\s+and\s+([a-z]{3,9})\s+20\d{2}\b", q):
        return ("monthly", expand)
    return ("monthly", expand)


def extract_between_years(user_query: str) -> tuple[int, int] | None:
    q = user_query.lower()
    m = re.search(r"\bbetween\s+(20\d{2})\s+(?:and|to)\s+(20\d{2})\b", q)
    if not m:
        return None
    y1, y2 = int(m.group(1)), int(m.group(2))
    if y1 > y2:
        y1, y2 = y2, y1
    return (y1, y2)


def extract_between_quarters_same_year(user_query: str) -> tuple[str, str] | None:
    """
    "between Q1 and Q2 2025" -> ("2025Q1","2025Q2")
    """
    q = user_query.lower()
    m = re.search(r"\bbetween\s+q([1-4])\s+and\s+q([1-4])\s+(20\d{2})\b", q)
    if not m:
        return None
    q1, q2, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return (f"{y}Q{q1}", f"{y}Q{q2}")


def quarter_to_months(q: str) -> list[str]:
    """Convert YYYYQ# to list of YYYY-MM months in that quarter."""
    p = pd.Period(q, freq="Q")
    start = p.asfreq("M", how="start")
    end = p.asfreq("M", how="end")
    return [str(x) for x in pd.period_range(start=start, end=end, freq="M")]


def aggregate_metric_over_periods(docs_metas: list[tuple[str, dict]], metric: str) -> float:
    total = 0.0
    for _, m in docs_metas:
        v = m.get(metric)
        if v is None:
            continue
        total += float(v)
    return total


def summarize_for_context(docs_metas: list[tuple[str, dict]], max_items: int = 6) -> list[tuple[str, dict]]:
    """
    Avoid dumping dozens of chunks into the prompt. Keep a small number of
    sources for evidence / citations.
    """
    if len(docs_metas) <= max_items:
        return docs_metas
    # Prefer earliest + latest + a few around the middle
    sorted_pairs = sorted(docs_metas, key=lambda dm: (dm[1].get("period") or ""))
    head = sorted_pairs[:2]
    tail = sorted_pairs[-2:]
    middle = sorted_pairs[len(sorted_pairs)//2 : len(sorted_pairs)//2 + max(0, max_items - 4)]
    return head + middle + tail


def retrive(user_query, embedding_model, collection, top_k: int = 5):
    meta = load_index_meta()
    latest_date = meta['latest_date']
    available_months = meta.get("available_months", [])
    available_quarters = meta.get("available_quarters", [])

    time_period = infer_granularity(user_query=user_query)
    intent = classify_intent(user_query)
    is_comparison = any(k in user_query.lower() for k in COMPARISON_KEYWORDS)
    answer_granularity, expand_to_subperiods = detect_answer_granularity_for_comparison(user_query) if is_comparison else (time_period, False)

    # 1) Exact lookup path: explicit periods (including multi-month)
    explicit_periods: list[str] = []
    if time_period == "monthly":
        explicit_periods = extract_explicit_month_periods(user_query)
        if available_months:
            avail = set(available_months)
            explicit_periods = [p for p in explicit_periods if p in avail]
    elif time_period == "quarterly":
        # If the query is a comparison between two quarters, prefer extracting both.
        qs = extract_between_quarters_same_year(user_query) if is_comparison else None
        if qs:
            explicit_periods = [qs[0], qs[1]]
        else:
            qp = extract_quarter_period(user_query)
            if qp:
                explicit_periods = [qp]
        if available_quarters:
            avail = set(available_quarters)
            explicit_periods = [p for p in explicit_periods if p in avail]

    if explicit_periods:
        docs_metas = get_chunks_by_periods(collection, chunk_type=time_period, periods=explicit_periods)
        computed = None
        if any(k in user_query.lower() for k in COMPARISON_KEYWORDS):
            computed = compute_income_growth(docs_metas)
        return docs_metas, {
            "mode": "exact_lookup",
            "intent": "exact_lookup",
            "latest_date": latest_date,
            "granularity": time_period,
            "periods": explicit_periods,
            "time_window": None,
            "computed": computed,
            "answer_granularity": answer_granularity,
            "expand_to_subperiods": expand_to_subperiods,
        }

    # 2) Exact lookup path: relative windows like "last 4 months" (kept deterministic)
    req = extract_last_n(user_query=user_query)
    if req:
        g, n = req
        periods = last_n_months(latest_date, n) if g == 'monthly' else last_n_quarters(latest_date, n)
        docs_metas = get_chunks_by_periods(collection, chunk_type=g, periods=periods)
        computed = None
        if any(k in user_query.lower() for k in COMPARISON_KEYWORDS):
            computed = compute_income_growth(docs_metas)
        # If this is a ranking query (e.g. "highest sales"), rank within the window deterministically.
        if intent == "analytical_ranking":
            metric = metric_for_ranking(user_query)
            ranked: list[tuple[float, str, dict]] = []
            for doc, m in docs_metas:
                if metric == "margin":
                    inc = m.get("total_income") or 0
                    npv = m.get("net_pl") or 0
                    val = (npv / inc) if inc else None
                else:
                    val = m.get(metric)
                if val is None:
                    continue
                ranked.append((float(val), doc, m))
            reverse = not any(w in user_query.lower() for w in ["worst", "lowest", "weakest"])
            ranked.sort(key=lambda x: x[0], reverse=reverse)
            top = [(d, m) for _, d, m in ranked[:top_k]]
            return top, {
                "mode": "analytical_ranking",
                "intent": "analytical_ranking",
                "latest_date": latest_date,
                "granularity": g,
                "periods": [m.get("period") for _, m in top],
                "time_window": {"start": periods[0], "end": periods[-1]} if periods else None,
                "ranking_metric": metric,
                "computed": None,
                "answer_granularity": answer_granularity,
                "expand_to_subperiods": expand_to_subperiods,
            }

        return docs_metas, {
            "mode": "exact_window",
            "intent": "exact_lookup",
            "latest_date": latest_date,
            "granularity": g,
            "periods": periods,
            "time_window": {"start": periods[0], "end": periods[-1]} if periods else None,
            "computed": computed,
            "answer_granularity": answer_granularity,
            "expand_to_subperiods": expand_to_subperiods,
        }

    # 3) Exact lookup path: exact single named period
    period = extract_single_period(user_query=user_query, granularity=time_period)
    if period:
        docs_metas = get_chunks_by_periods(collection, time_period, [period])
        return docs_metas, {
            "mode": "exact_lookup",
            "intent": "exact_lookup",
            "latest_date": latest_date,
            "granularity": time_period,
            "periods": [period],
            "time_window": None,
            "computed": None,
            "answer_granularity": answer_granularity,
            "expand_to_subperiods": expand_to_subperiods,
        }

    # 4) Time window extraction (used for ranking, semantic filtering, and exact compare windows)
    time_window = extract_time_window(user_query, time_period, latest_date) or extract_relative_time_window(
        user_query, time_period, latest_date
    )

    # 4aa) Comparison queries with broad windows: do NOT expand unless breakdown requested.
    # Example: "growth between 2024 and 2025" -> yearly comparison (aggregate), not 24 monthly chunks.
    if is_comparison and answer_granularity == "yearly":
        years = extract_between_years(user_query)
        if years:
            y1, y2 = years
            # Use monthly chunks as the aggregation substrate (we only store weekly/monthly/quarterly)
            months_y1 = [p for p in pd.period_range(start=f"{y1}-01", end=f"{y1}-12", freq="M").astype(str).tolist()]
            months_y2 = [p for p in pd.period_range(start=f"{y2}-01", end=f"{y2}-12", freq="M").astype(str).tolist()]
            if available_months:
                avail = set(available_months)
                months_y1 = [p for p in months_y1 if p in avail]
                months_y2 = [p for p in months_y2 if p in avail]
            docs1 = get_chunks_by_periods(collection, chunk_type="monthly", periods=months_y1)
            docs2 = get_chunks_by_periods(collection, chunk_type="monthly", periods=months_y2)
            inc1 = aggregate_metric_over_periods(docs1, "total_income")
            inc2 = aggregate_metric_over_periods(docs2, "total_income")
            growth = _safe_pct_change(inc1, inc2)
            computed = {
                "comparison": "yearly_income_growth",
                "from_year": y1,
                "to_year": y2,
                "income_from": inc1,
                "income_to": inc2,
                "income_growth_pct": growth,
            }
            # Only include subperiod sources in context if the user explicitly asked for breakdown.
            sources = (docs1 + docs2) if expand_to_subperiods else summarize_for_context(docs1 + docs2, max_items=6)
            return sources, {
                "mode": "exact_lookup",
                "intent": "exact_lookup",
                "latest_date": latest_date,
                "granularity": "monthly",
                "periods": [],
                "time_window": {"start": f"{y1}-01", "end": f"{y2}-12"},
                "computed": computed,
                "answer_granularity": "yearly",
                "expand_to_subperiods": expand_to_subperiods,
            }

    if is_comparison and answer_granularity == "monthly":
        years = extract_between_years(user_query)
        if years:
            y1, y2 = years
            # monthly breakdown: compare same month across years (YoY by month)
            breakdown = []
            for m in range(1, 13):
                p1 = f"{y1:04d}-{m:02d}"
                p2 = f"{y2:04d}-{m:02d}"
                if available_months:
                    avail = set(available_months)
                    if p1 not in avail or p2 not in avail:
                        continue
                d1 = get_chunks_by_periods(collection, chunk_type="monthly", periods=[p1])
                d2 = get_chunks_by_periods(collection, chunk_type="monthly", periods=[p2])
                if not d1 or not d2:
                    continue
                inc1 = d1[0][1].get("total_income")
                inc2 = d2[0][1].get("total_income")
                if inc1 is None or inc2 is None:
                    continue
                breakdown.append(
                    {
                        "month": f"{m:02d}",
                        "from_period": p1,
                        "to_period": p2,
                        "income_from": float(inc1),
                        "income_to": float(inc2),
                        "income_growth_pct": _safe_pct_change(float(inc1), float(inc2)),
                    }
                )
            computed = {
                "comparison": "monthly_yoy_income_growth",
                "from_year": y1,
                "to_year": y2,
                "breakdown": breakdown,
            }
            # For breakdown requests, include all month pairs in context; otherwise keep it tight
            sources = []
            if expand_to_subperiods:
                # include all referenced months as evidence
                periods = [b["from_period"] for b in breakdown] + [b["to_period"] for b in breakdown]
                sources = get_chunks_by_periods(collection, chunk_type="monthly", periods=periods)
            else:
                periods = []
                for b in breakdown[:2] + breakdown[-2:]:
                    periods.extend([b["from_period"], b["to_period"]])
                sources = get_chunks_by_periods(collection, chunk_type="monthly", periods=periods)
            return sources, {
                "mode": "exact_lookup",
                "intent": "exact_lookup",
                "latest_date": latest_date,
                "granularity": "monthly",
                "periods": [],
                "time_window": {"start": f"{y1}-01", "end": f"{y2}-12"},
                "computed": computed,
                "answer_granularity": "monthly",
                "expand_to_subperiods": expand_to_subperiods,
            }

    if is_comparison and answer_granularity == "quarterly":
        qs = extract_between_quarters_same_year(user_query)
        if qs:
            q1, q2 = qs
            docs = get_chunks_by_periods(collection, chunk_type="quarterly", periods=[q1, q2])
            # Fall back to aggregating monthly chunks if quarterly chunks are missing
            if len(docs) < 2:
                months1 = quarter_to_months(q1)
                months2 = quarter_to_months(q2)
                if available_months:
                    avail = set(available_months)
                    months1 = [p for p in months1 if p in avail]
                    months2 = [p for p in months2 if p in avail]
                d1 = get_chunks_by_periods(collection, chunk_type="monthly", periods=months1)
                d2 = get_chunks_by_periods(collection, chunk_type="monthly", periods=months2)
                inc1 = aggregate_metric_over_periods(d1, "total_income")
                inc2 = aggregate_metric_over_periods(d2, "total_income")
                computed = {
                    "comparison": "quarterly_income_growth",
                    "from_quarter": q1,
                    "to_quarter": q2,
                    "income_from": inc1,
                    "income_to": inc2,
                    "income_growth_pct": _safe_pct_change(inc1, inc2),
                }
                sources = summarize_for_context(d1 + d2, max_items=6)
                return sources, {
                    "mode": "exact_lookup",
                    "intent": "exact_lookup",
                    "latest_date": latest_date,
                    "granularity": "monthly",
                    "periods": [],
                    "time_window": {"start": months1[0], "end": months2[-1]} if months1 and months2 else None,
                    "computed": computed,
                    "answer_granularity": "quarterly",
                    "expand_to_subperiods": expand_to_subperiods,
                }

            inc1 = docs[0][1].get("total_income") if len(docs) > 0 else None
            inc2 = docs[1][1].get("total_income") if len(docs) > 1 else None
            computed = None
            if inc1 is not None and inc2 is not None:
                computed = {
                    "comparison": "quarterly_income_growth",
                    "from_quarter": q1,
                    "to_quarter": q2,
                    "income_from": inc1,
                    "income_to": inc2,
                    "income_growth_pct": _safe_pct_change(float(inc1), float(inc2)),
                }
            return docs, {
                "mode": "exact_lookup",
                "intent": "exact_lookup",
                "latest_date": latest_date,
                "granularity": "quarterly",
                "periods": [q1, q2],
                "time_window": None,
                "computed": computed,
                "answer_granularity": "quarterly",
                "expand_to_subperiods": expand_to_subperiods,
            }

    # 4a) Exact lookup path: relative compare windows (e.g. "compare this quarter to last quarter")
    if intent == "exact_lookup" and time_window:
        periods = periods_in_range(time_window["start"], time_window["end"], time_period)
        if time_period == "monthly" and available_months:
            avail = set(available_months)
            periods = [p for p in periods if p in avail]
        if time_period == "quarterly" and available_quarters:
            avail = set(available_quarters)
            periods = [p for p in periods if p in avail]

        docs_metas = get_chunks_by_periods(collection, chunk_type=time_period, periods=periods)
        computed = None
        if any(k in user_query.lower() for k in COMPARISON_KEYWORDS):
            computed = compute_income_growth(docs_metas)
        return docs_metas, {
            "mode": "exact_window",
            "intent": "exact_lookup",
            "latest_date": latest_date,
            "granularity": time_period,
            "periods": periods,
            "time_window": time_window,
            "computed": computed,
            "answer_granularity": answer_granularity,
            "expand_to_subperiods": expand_to_subperiods,
        }

    # 4b) Analytical/ranking path: time window + deterministic ranking (no semantic)
    if intent == "analytical_ranking" and time_window:
        candidate_periods = periods_in_range(time_window["start"], time_window["end"], time_period)
        if time_period == "monthly" and available_months:
            avail = set(available_months)
            candidate_periods = [p for p in candidate_periods if p in avail]
        if time_period == "quarterly" and available_quarters:
            avail = set(available_quarters)
            candidate_periods = [p for p in candidate_periods if p in avail]

        docs_metas = get_chunks_by_periods(collection, chunk_type=time_period, periods=candidate_periods)
        metric = metric_for_ranking(user_query)
        ranked: list[tuple[float, str, dict]] = []
        for doc, m in docs_metas:
            if metric == "margin":
                inc = m.get("total_income") or 0
                npv = m.get("net_pl") or 0
                val = (npv / inc) if inc else None
            else:
                val = m.get(metric)
            if val is None:
                continue
            ranked.append((float(val), doc, m))
        reverse = not any(w in user_query.lower() for w in ["worst", "lowest", "weakest"])
        ranked.sort(key=lambda x: x[0], reverse=reverse)
        top = [(d, m) for _, d, m in ranked[:top_k]]
        return top, {
            "mode": "analytical_ranking",
            "intent": "analytical_ranking",
            "latest_date": latest_date,
            "granularity": time_period,
            "periods": [m.get("period") for _, m in top],
            "time_window": time_window,
            "ranking_metric": metric,
            "computed": None,
            "answer_granularity": answer_granularity,
            "expand_to_subperiods": expand_to_subperiods,
        }

    # 5) Semantic exploration path (optionally time-window filtered)
    where_clause = {"chunk_type": time_period}
    if time_window:
        period_list = periods_in_range(time_window["start"], time_window["end"], time_period)
        where_clause = {"$and": [{"chunk_type": time_period}, {"period": {"$in": period_list}}]}

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
        "mode": "semantic_exploration",
        "intent": "semantic_exploration",
        "latest_date": latest_date,
        "granularity": time_period,
        "periods": [],
        "time_window": time_window,
        "computed": None,
        "answer_granularity": time_period,
        "expand_to_subperiods": False,
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
    header = f"Data time reference: latest transaction date = {extra_info['latest_date']}\n"
    header += (
        f"Routing: mode={extra_info.get('mode')}, granularity={extra_info.get('granularity')}, "
        f"answer_granularity={extra_info.get('answer_granularity')}, expand_to_subperiods={extra_info.get('expand_to_subperiods')}, "
        f"periods={extra_info.get('periods', [])}, time_window={extra_info.get('time_window')}\n"
    )
    if extra_info.get("computed"):
        c = extra_info["computed"]
        if c.get("comparison") in ("yearly_income_growth", "quarterly_income_growth"):
            from_label = c.get("from_year") or c.get("from_quarter")
            to_label = c.get("to_year") or c.get("to_quarter")
            pct = c.get("income_growth_pct")
            pct_txt = f"{pct:.1f}%" if pct is not None else "N/A"
            header += (
                f"Computed: income growth {from_label} → {to_label} = {pct_txt} "
                f"(£{c.get('income_from', 0):,.0f} → £{c.get('income_to', 0):,.0f})\n"
            )
        else:
            header += (
                f"Computed: income growth from {c.get('from_period')} to {c.get('to_period')} "
                f"= {c.get('income_growth_pct'):.1f}% "
                f"(£{c.get('income_from', 0):,.0f} → £{c.get('income_to', 0):,.0f})\n"
            )
    context = header + "\n" + context

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
    header = f"Data time reference: latest transaction date = {extra_info['latest_date']}\n"
    header += (
        f"Routing: mode={extra_info.get('mode')}, granularity={extra_info.get('granularity')}, "
        f"answer_granularity={extra_info.get('answer_granularity')}, expand_to_subperiods={extra_info.get('expand_to_subperiods')}, "
        f"periods={extra_info.get('periods', [])}, time_window={extra_info.get('time_window')}\n"
    )
    if extra_info.get("computed"):
        c = extra_info["computed"]
        if c.get("comparison") in ("yearly_income_growth", "quarterly_income_growth"):
            from_label = c.get("from_year") or c.get("from_quarter")
            to_label = c.get("to_year") or c.get("to_quarter")
            pct = c.get("income_growth_pct")
            pct_txt = f"{pct:.1f}%" if pct is not None else "N/A"
            header += (
                f"Computed: income growth {from_label} → {to_label} = {pct_txt} "
                f"(£{c.get('income_from', 0):,.0f} → £{c.get('income_to', 0):,.0f})\n"
            )
        else:
            header += (
                f"Computed: income growth from {c.get('from_period')} to {c.get('to_period')} "
                f"= {c.get('income_growth_pct'):.1f}% "
                f"(£{c.get('income_from', 0):,.0f} → £{c.get('income_to', 0):,.0f})\n"
            )
    context = header + "\n" + context

    history_block = format_history_for_prompt(chat_history)

    for token in chain.stream({"context": context, "query": resolved_query, "history": history_block}):
        yield token, docs_metas, extra_info


if __name__ == "__main__":
    print("rag.py imports ✓")
    print("Chain initialised ✓")
