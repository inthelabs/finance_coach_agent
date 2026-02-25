#all the retrival code and generation of the augmented prompt that is then feed to LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import chromadb

import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')


system_prompt = """
You are a financial advisor with expertise in behavioral psychology and financial habits.
Analyze spending patterns and give personalized, actionable advice.
If asked "can I afford X", analyze their data and suggest strategies.

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
google_api_key=GOOGLE_API_KEY
)

chain = prompt_template | llm | StrOutputParser()
    
def retrive(user_query,embedding_model, collection, top_k: int=5):
    query_embedding = embedding_model.encode(user_query)
    #get the top 5 similar chunks from Chroma
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    return results

def query_financial_chatbot(user_query,embedding_model,collection):

    results = retrive(user_query,embedding_model,collection)
    
    #build the conext string that will be then passed to the llm
    context = ""
    for doc in results['documents'][0]:
        context += doc +"\n---\n"

    response = chain.invoke({"context":context, "query":user_query})

    return response, results


if __name__ == "__main__":    
    print("rag.py imports ✓")
    print("Chain initialised ✓")