#all the retrival code and generation of the augmented prompt that is then feed to LLM
from langchain_core.output_parsers import StrOutputParser
#from langchain_ollama import OllamaLLM
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

# system_prompt = """
# You are a financial advisor with expertise in behavioral psychology and financial habits.
# Analyze spending patterns and give personalized, actionable advice.
# If asked "can I afford X", analyze their data and suggest strategies.

# """

system_prompt = """
You are a financial advisor with expertise in behavioral psychology and financial habits.

Rules:
- Use ONLY the Financial Context provided. If the context does not contain the needed period or details, ask a brief follow-up question.
- Be specific, practical, and data-backed. Use £ amounts when available.
- If asked "can I afford X", estimate affordability using income, expenses, and net P&L in the context, and suggest strategies.
"""
prompt_template = ChatPromptTemplate.from_template(
system_prompt + """

Financial Context:
{context}

User Query: {query}

Advice:"""
)

llm = ChatGoogleGenerativeAI(
model="gemini-2.0-flash",
google_api_key=GOOGLE_API_KEY,
streaming=True
)

chain = prompt_template | llm | StrOutputParser()


def infer_granularity(user_query: str) -> str:
    q = user_query.lower()
    
    if 'quarter' in q or re.search(r"\bq[1-4]\b",q):
        return 'quarterly'
    
    if 'weekly' in q or 'week' in q or 'last 7 days' in q:
        return 'weekly'
    
    # monthly intent (default)
    # month names or "month"/"monthly" or YYYY-MM
    if 'month' in q or 'monthly' in q or re.search(r"\b20\d{2}-\d{2}\b",q):
        return 'monthly'
    
    #otherwise default to monthly as it is usually the safest. 
    return 'monthly'

import re

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
        year = int(m.group(1))
        month = int(m.group(2))
        return f"{year:04d}-{month:02d}"

    # 2) MM/YYYY or MM/YY (also handles 1/26)
    m = re.search(r"\b(0?[1-9]|1[0-2])/(20\d{2}|\d{2})\b", q)
    if m:
        month = int(m.group(1))
        year = normalize_year(int(m.group(2)))
        return f"{year:04d}-{month:02d}"

    # 3) MonthName + Year (Jan26, Jan 2026, January '26)
    m = re.search(r"\b([a-z]{3,9})[\s\-']*(20\d{2}|\d{2})\b", q)
    if m:
        mon_txt = m.group(1)
        if mon_txt in MONTHS:
            month = MONTHS[mon_txt]
            year = normalize_year(int(m.group(2)))
            return f"{year:04d}-{month:02d}"

    return None

def extract_quarter_period(user_query: str) -> str | None:
    q = user_query.strip().lower()
    m = re.search(r"\b(20\d{2})\s*q([1-4])\b", q) or re.search(r"\bq([1-4])\s*(20\d{2})\b", q)
    if not m:
        return None

    if m.re.pattern.startswith("\\b(20"):
        year = int(m.group(1)); quarter = int(m.group(2))
    else:
        quarter = int(m.group(1)); year = int(m.group(2))

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

def extract_period(user_query: str):
    m = re.search(r"\b(20\d{2}-\d{2})\b", user_query)
    if m:
        return m.group(1)
    q = re.search(r"\b(20\d{2})\s*q([1-4])\b", user_query.lower())
    if q:
        return f"{q.group(1)}Q{q.group(2)}"
    return None

def format_context(results) -> str:
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    #print(f"Similarity Search: {docs}")
    #print(f"Similarity Search Metas: {metas}")
    
    blocks = []
    for doc, meta in zip(docs, metas):
        period = meta.get("period", "unknown")
        chunk_type = meta.get("chunk_type", "unknown")
        blocks.append(f"[{chunk_type} | {period}]\n{doc}".strip())
        
    #from datetime import datetime
    #blocks += f"For time reference, today's date and time is: {datetime.now()}"
    
    return "\n\n---\n\n".join(blocks)


def last_n_months(latest_date_iso: str, n: int) -> list[str]:
    latest = pd.Timestamp(latest_date_iso).to_period("M")
    return [str(latest - i) for i in reversed(range(n))]

def last_n_quarters(latest_date_iso: str, n: int) -> list[str]:
    latest = pd.Timestamp(latest_date_iso).to_period("Q")
    return [str(latest - i) for i in reversed(range(n))]

def get_chunks_by_periods(collection, chunk_type: str, periods: list[str]):
    docs_metas = []
    for p in periods:
        got = collection.get(
            where={"$and": [
                {"chunk_type": chunk_type},
                {"period": p}
            ]}
        )
        docs = got.get("documents", [])
        metas = got.get("metadatas", [])
        for d, m in zip(docs, metas):
            docs_metas.append((d, m))
    return docs_metas

def retrive(user_query,embedding_model, collection, top_k: int=5):
 
    #load meta data so we can retrive the latest data
    meta = load_index_meta()
    latest_date = meta['latest_date']
    
    #before we use similarity search lets first infer the time period of the users request
    time_period = infer_granularity(user_query=user_query)
    
    #1) deterministic: last N months/quarters
    req = extract_last_n(user_query=user_query) #returns ('monthly',6) or ('quarterly,3) or None.
    if req:
        g,n = req
        if g == 'monthly':
            periods = last_n_months(latest_date_iso=latest_date,n=n)
        else:
            periods = last_n_quarters(latest_date_iso=latest_date,n=n)
        
        docs_metas = get_chunks_by_periods(collection=collection,chunk_type=g,periods=periods)
        print(f'Deterministic Retrival: last N {periods}')
        return docs_metas, {"mode": "exact_window", "latest_date": latest_date, "periods": periods}

    #2) Deterministic: get an extract period
    period = extract_single_period(user_query=user_query,granularity=time_period)
    if period:
        docs_metas = get_chunks_by_periods(collection, time_period, [period])
        print(f'Deterministic Retrival: get exact period {period}')
        
        return docs_metas, {"mode": "exact_period", "latest_date": latest_date, "periods": [period]}

    #3) Similarity Search using Embeddings
    query_embedding = embedding_model.encode(user_query)
    #get the top 5 similar chunks from Chroma
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where={"chunk_type": time_period}
    )
    
    # if we've gotten here then it is using Semantic search. Convert into (doc, meta) pairs
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    docs_metas = list(zip(docs, metas))
    print(f'Similarity Search Retrival')
    
    return docs_metas, {"mode": "semantic", "latest_date": latest_date}

def format_context_from_pairs(docs_metas):
    blocks = []
    for doc, meta in docs_metas:
        blocks.append(f"[{meta.get('chunk_type')} | {meta.get('period')}]\n{doc}".strip())
    return "\n\n---\n\n".join(blocks)

def query_financial_chatbot(user_query,embedding_model,collection):

    results, extra_info = retrive(user_query,embedding_model,collection)
    
    #build the conext string that will be then passed to the llm
    # context = ""
    # for doc in results['documents'][0]:
    #     context += doc +"\n---\n"
    context = format_context_from_pairs(docs_metas=results)
    context = f"Data time reference: latest transaction date = {extra_info['latest_date']}\n\n" + context

    response = chain.invoke({"context":context, "query":user_query})

    return response, results, extra_info

def stream_financial_chatbot(user_query, embedding_model, collection):
    docs_metas, extra_info = retrive(user_query, embedding_model, collection)

    context = format_context_from_pairs(docs_metas=docs_metas)
    context = f"Data time reference: latest transaction date = {extra_info['latest_date']}\n\n" + context

    # stream tokens from the chain
    for chunk in chain.stream({"context": context, "query": user_query}):
        yield chunk, docs_metas, extra_info
        
if __name__ == "__main__":    
    print("rag.py imports ✓")
    print("Chain initialised ✓")