# corpus_stats.py - Run this to get your presentation stats
import chromadb
import json
import os

# Load index meta
try:
    with open('./chroma/index_meta.json', 'r') as f:
        meta = json.load(f)
    print(f"\n📅 Latest date in dataset: {meta.get('latest_date')}")
    print(f"📅 Earliest date: {meta.get('earliest_date', 'not recorded')}")
except:
    print("index_meta.json not found")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma")

# Get collection
try:
    collection = client.get_collection("financial_data")
    
    total = collection.count()
    print(f"\n📦 TOTAL CHUNKS IN CHROMADB: {total}")
    
    # Count by chunk type
    for chunk_type in ['weekly', 'monthly', 'quarterly']:
        results = collection.get(where={"chunk_type": chunk_type})
        count = len(results['ids'])
        print(f"   {chunk_type}: {count} chunks")
    
    # Sample a chunk to see token estimate
    sample = collection.get(limit=1)
    if sample['documents']:
        sample_text = sample['documents'][0]
        words = len(sample_text.split())
        tokens_approx = int(words * 1.3)
        print(f"\n📝 Sample chunk word count: {words}")
        print(f"📝 Estimated tokens per chunk: ~{tokens_approx}")
        print(f"📝 Estimated TOTAL tokens: ~{total * tokens_approx:,}")
        print(f"\n--- SAMPLE CHUNK ---")
        print(sample_text[:500])
        
except Exception as e:
    print(f"Error: {e}")

# Transaction count from CSV
try:
    import pandas as pd
    df = pd.read_csv('synthetic_transactions.csv', on_bad_lines='skip')
    print(f"\n💳 Total transactions in CSV: {len(df):,}")
    print(f"📅 Date range: {df['date'].min()} → {df['date'].max()}")
except Exception as e:
    print(f"CSV error: {e}")