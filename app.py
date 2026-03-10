import streamlit as st
from ingest import main as run_ingest
from utils import ingest_data, generate_synthetic_transactions
from rag import query_financial_chatbot, stream_financial_chatbot
import chromadb

RESET_DB = True
# In app.py, before loading collection
import os
if not os.path.exists('synthetic_transactions.csv') or RESET_DB:
    csv_path = 'synthetic_transactions.csv'
    generate_synthetic_transactions(
        num_transactions=100000,
        years=15,
        output_file=csv_path,
        combine=False  # no real data to combine with
    )


# Page config
st.set_page_config(page_title="Financial Coach", layout="wide")

st.title("💰 Financial Coach")
st.markdown("Ask me about your business finances")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "collection" not in st.session_state:
    with st.spinner("Loading financial data..."):
        collection, embedding_model = ingest_data(reset_db=RESET_DB)
        st.session_state.collection = collection
        st.session_state.embedding_model = embedding_model

collection = st.session_state.collection
embedding_model = st.session_state.embedding_model

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "citations" in message:
            with st.expander("📊 Data Sources"):
                for i, citation in enumerate(message["citations"], 1):
                    st.markdown(f"**Source {i}:** {citation}")

# Chat input
user_input = st.chat_input("Ask about your finances...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response from RAG system (non-streaming — used to populate session state)
    with st.spinner("Analyzing your data..."):
        response, results, extra_info = query_financial_chatbot(
            user_input,
            embedding_model,
            collection,
            chat_history=st.session_state.messages
        )

    # Extract citations from retrieved chunks
    citations = [
        f"[{meta.get('chunk_type')} | {meta.get('period')}]\n{doc}"
        for (doc, meta) in results
    ] if results else []

    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "citations": citations
    })

    # Display assistant message with streaming
    with st.chat_message("assistant"):
        st.caption(f"Mode: {extra_info['mode']}, Periods: {extra_info.get('periods', [])}")
        placeholder = st.empty()
        full_text = ""
        final_results = None
        final_extra = None

        for token, docs_metas, extra_info in stream_financial_chatbot(
            user_input,
            embedding_model,
            collection,
            chat_history=st.session_state.messages
        ):
            full_text += token
            placeholder.markdown(full_text)
            final_results = docs_metas
            final_extra = extra_info

        # show caption after stream finishes
        if final_extra:
            st.caption(f"Mode: {final_extra['mode']}, Periods: {final_extra.get('periods', [])}")

        # citations after stream finishes
        citations = [
            f"[{meta.get('chunk_type')} | {meta.get('period')}]\n{doc}"
            for (doc, meta) in (final_results or [])
        ]

        with st.expander("📊 Data Sources"):
            for i, c in enumerate(citations, 1):
                st.markdown(f"**Source {i}:**\n{c}")
