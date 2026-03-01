# Prototyping our agent

import os
from emb_ingest import ingest

if not os.path.exists("vectorstore"):
     ingest()

import streamlit as st
from core import load_components, retrieve, build_prompt, generate_answer


#---- Page config - Must be first streamlit call

st.set_page_config(
    page_title="Mahesh/'s Solutions Enterprise Agent",
    page_icon=" 😎 ",
    layout="wide",
)

# ---- Client Page

def main():
     with st.sidebar:
          st.title("Enterprise Agent")
          st.markdown("Ask anything about Mahesh\'s Solutions company policies")
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
     st.title("Mahesh\'s Solutions Enterprise Agent")
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
                    prompt = build_prompt(question, chunks)
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
                        
                          
                                               
                              
         
         
        

        

          

         
    