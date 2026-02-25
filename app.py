import streamlit as st
from ingest import main as run_ingest
from utils import ingest_data
from rag import query_financial_chatbot
import chromadb

# In app.py, before loading collection
import os
if not os.path.exists('synthetic_transactions.csv'):
    generate_synthetic_transactions(
        num_transactions=10000,
        years=2,
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
        collection, embedding_model = ingest_data()
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
    
    # Get response from RAG system
    with st.spinner("Analyzing your data..."):
        response, results = query_financial_chatbot(
            user_input, 
            embedding_model, 
            collection
        )
    
    # Extract citations from retrieved chunks
    citations = results['documents'][0] if results['documents'] else []
    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "citations": citations
    })
    
    # Display assistant message with citations
    with st.chat_message("assistant"):
        st.markdown(response)
        with st.expander("📊 Data Sources"):
            for i, citation in enumerate(citations, 1):
                st.markdown(f"**Source {i}:**\n{citation}")