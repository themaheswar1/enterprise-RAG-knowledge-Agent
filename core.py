"""
core.py — Shared agent logic
"""

import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st

load_dotenv()

VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "Mahesh_docs"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL      = "llama-3.3-70b-versatile"
TOP_K           = 5
MAX_TOKENS      = 1024


def load_components():
    embedder    = SentenceTransformer(EMBED_MODEL)
    client      = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    collection  = client.get_collection(name=COLLECTION_NAME)
    
    try:
        import streamlit as st
        api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY","")
    except Exception:
        api_key = os.getenv("GROQ_API_KEY")    
    
    groq_client = Groq(api_key=api_key)

    return embedder, collection, groq_client


def retrieve(question, embedder, collection):
    question_vector = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text":     results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return chunks


def build_prompt(question, chunks):
    context_blocks = []
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        source = (
            f"[Source {i+1}] "
            f"File: {meta['filename']} | "
            f"Page: {meta['page_num']} | "
            f"Lines: {meta['line_start']}–{meta['line_end']}"
        )
        context_blocks.append(f"{source}\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_blocks)

    return f"""You are an intelligent assistant for Mahesh/'s Solutions.
Answer employee questions using ONLY the company documents provided.

Rules:
1. Answer using ONLY the information in the provided context.
2. If the answer is not in the context, Say: I couldn't find this is company policy.
3. Always end your answer with a citations section listening which sources you have used.
4. Be concise and Best professional.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def generate_answer(prompt, groq_client):
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful enterprise knowledge assistant."},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.1,
    )
    return response.choices[0].message.content