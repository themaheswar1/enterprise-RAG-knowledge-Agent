# Prototyping our agent

import os
import streamlit as st
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

#------ Configuraton ------

VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "Mahesh_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5
MAX_TOKENS = 1024

#---- Page config - Must be first streamlit call

st.set_page_config(
    page_title="Mahesh/'s Solutions Enterprise Agent",
    page_icon=" 😎 ",
    layout="wide",
)

# ----- Load Components -- cached so they load once

@st.cache_resource
def load_components():
    """
    Load once & reuse across all interactions
    """
    embedder = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return embedder, collection, groq_client

# --- Core Agent Functions

def retrieve(question, embedder, collection):
    question_vector = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings = [question_vector],
        n_results = TOP_K,
        include = ["documents","metadatas","distances"]
    )
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text":     results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return chunks
    
def build_promt(question, chunks):
        context_blocks = []
        for i, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            source=(
                f"[Source {i+1}] "
                f"File: {meta['filename']} | "
                f"Page: {meta['page_num']} | "
                f"Lines: {meta['line_start']}–{meta['line_end']}"
            )
            context_blocks.append(f" {source}\n {chunk['text']}")
        context = "\n\n\n -- \n\n".join(context_blocks)

        return f"""You are an intelligent assistant for NovaTech Solutions.
Answer employee questions using ONLY the company documents provided.

Rules:
1. Answer using ONLY the information in the context below.
2. If the answer is not in the context, say: "I couldn't find this in the available documents."
3. Always cite which source(s) you used at the end of your answer.
4. Be concise and professional.

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
    
# ---------- UI

def main():
     with st.sidebar:
          st.title("Enterprise Agent")
          st.markdown("Ask anything about Mahesh/'s Solutions company policies")
          st.divider()

          st.markdown("**Documents loaded:**")
          docs = [
            "📄 Employee Leave Policy",
            "📄 Expense Reimbursement",
            "📄 Client Escalation Process",
            "📄 Onboarding Guide",
            "📄 Remote Work Policy",
            "📄 Performance Review",
            "📄 Data Security Guidelines",
            "📄 Project Handover Template",
            "📄 Vendor Management Policy",
            "📄 Q3 Meeting Notes",
        
        ]
          for doc in docs:
            st.markdown(f"&nbsp;&nbsp;{doc}", unsafe_allow_html=True)

          st.divider()
          st.caption("Powered by Groq + ChromaDB + all-MiniLM-L6-v2")

     # Main Window
     st.title("Mahesh/'s Solutions Enterprise Agent")
     st.divider()

     with st.spinner("Loading Agent.."):
          embedder, collection, groq_client = load_components()

     # Chat history
     if "messages" not in st.session_state:
          st.session_state.messages = []

     # Display previous messages
     for msg in st.session_state.messages:
          with st.chat_message(msg["role"]):
               st.markdown(msg["content"])
               if msg["role"] == "assistant" and "chunks" in msg:
                    with st.expander("View Sources"):
                         for i, chunk in enumerate(msg["chunks"]):
                              meta = chunk["metadata"]
                              similarity = round(1-chunk["distance"],3)
                              st.markdown(
                                   f"**[{i+1}]** `{meta['filename']}` | "
                                   f"Page **{meta['page_num']}** | "
                                   f"Lines **{meta['line_start']}–{meta['line_end']}** | "
                                   f"Similarity: `{similarity}`"
                                   
                              )
                              with st.expander(f"View Chunk text [{i+1}]"):
                                   st.text(chunk["text"])
     # --- Chat input
     if question := st.chat_input("Ask a question about Mahesh/'s Tech policies.."):   
          st.session_state.messages.append({"role":"user", "content": question})
          with st.chat_message("user"):
               st.markdown(question)

          # Generating ans
          with st.chat_message("assistant"):
               with st.spinner("Searching Documents and Generating answer..."):
                    chunks = retrieve(question, embedder, collection)
                    prompt = build_promt(question, chunks)
                    answer = generate_answer(prompt, groq_client)

               st.markdown(answer)

               # Showing meta
               with st.expander(" View sources here"):
                    for i, chunk in enumerate(chunks):
                         meta = chunk["metadata"]
                         similarity = round(1-chunk["distance"],3)

                         st.markdown(
                             
                             f"Page **{meta['page_num']}** | "
                             f"Lines **{meta['line_start']}–{meta['line_end']}** | "
                             f"Similarity: `{similarity}`"
                         ) 
                         with st.expander(f"View chunk text [{i+1}]"):
                              st.text(chunk["text"])

               # Save to histoy
               st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "chunks": chunks,
               })   

if __name__ == "__main__":
     main()                           
                        
                          
                                               
                              
         
         
        

        

          

         
    