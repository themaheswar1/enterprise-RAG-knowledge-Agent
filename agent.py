"""
This is an Enterprise Knowledge agent.

"""

import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

#----- Configuration ------

VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "Mahesh_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5  # no:of chunks to retrieve per query
MAX_TOKENS = 1024


# Set up

def load_components():

    print("---- Loading Embedding model ----")
    embedder = SentenceTransformer(EMBED_MODEL)

    print(" ---- Connecting to ChromaDB ----")
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f" --> {collection.count()} chunks loaded from vectorstore")

    print(f" ---- Connecting to Groq ----")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    return embedder, collection, groq_client

# Step1: Retrieve relevant chunks

def retrieve(question: str, embedder, collection, top_k: int = TOP_K) -> list[dict]:
    
    #embed the question given by clients
    question_vector = embedder.encode(question).tolist()

    #searching chroma
    results = collection.query(
        query_embeddings = [question_vector],
        n_results = top_k,
        include=["documents", "metadatas", "distances"]
    )

    # results to clean list
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return chunks

# Step2 : Build the prompt

def build_promt(question: str, chunks: list[dict]) -> str:
    """
    Here we can build the prompt that gets sends to the LLM
    that includes retrieved chunks as context with their info as well
    """
    context_blocks = []
    
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        source=(
            f"[Source is {i+1}]"
            f"File is: {meta['filename']} ||"
            f"Page is: {meta['page_num']} ||"
            f"Lines are: {meta['line_start']}-{meta['line_end']}"
        )
        context_blocks.append(f"{source}\n{chunk['text']}")
    # making complete context
    context = "\n\n----\n\n".join(context_blocks)

    # system prompt
    prompt = f"""You are an intelligent and Expert assistant for Mahesh solutions.
    Your task or job is to answer the employee questions using ONLY the company documents provided below.

    Rules: 
    1. Answer using ONLY the information in the provided context.
    2. If the answer is not in the context, Say: I couldn't find this is company policy.
    3. Always end your answer with a citations section listening which sources you have used.
    4. Be concise and good professional 
    
    CONTEXT : {context}
    
    QUESTION: {question}

    ANSWER: 
    """
    return prompt


# ---- Generation Answer ---- 

def generate_answer(prompt: str, groq_client: Groq) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role":"system",
                "content": "You are a helpful enterprise knowledge assistant. Answer questions accurately using only the provided company documents. "
            },
            {
                "role":"user",
                "content": prompt
            }

        ],
        max_tokens=MAX_TOKENS,
        temperature=0.1  # low temp = more factual, less creative
    )
    return response.choices[0].message.content

# -- STEP 4 : Formatting the final response

def format_response(answer: str, chunks: list[dict]) -> str:
    # Print the answer cleanly with retrieved source info

    output = []
    output.append(f"\n{'='*60}")
    output.append("  ANSWER")
    output.append(f"{'='*60}")
    output.append(answer)
    output.append(f"\n{'─'*60}")
    output.append("  RETRIEVED SOURCES")
    output.append(f"{'─'*60}")

    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        similarity = round(1 - chunk["distance"],3) 
        output.append(
            f"  [{i+1}] {meta['filename']} | "
            f"Page {meta['page_num']} | "
            f"Lines {meta['line_start']}–{meta['line_end']} | "
            f"Similarity: {similarity}"
        )

    output.append(f"{'='*60}\n")    
    return "\n".join(output)



# ---- Main Agent Loop ----

def run_agent():
    print(f"\n{'='*60}")
    print("  MAHESH'S SOLUTIONS KNOWLEDGE AGENT")
    print("  Ask questions about company policies and documents.")
    print("  Type 'exit','q','quit','end' to quit.")
    print(f"{'='*60}\n")

    """loading everything once i.e following tuple unpacking as
    load_components returns 3 (embedder, collection, groq_client)"""

    embedder, collection, groq_client = load_components()

    while True:
        question = input("Your Question:  ").strip()

        if not question:
            continue

        if question.lower() in ['exit','quit','q','Quit','end','End']:
            print("Good Bye dude, I was Happy clarifying you")
            break

        print(f"\n Searching Documents ....\n ")
        
        # Pipeline: retrieve -> build prompt -> generate -> format
        chunks = retrieve(question,embedder,collection)
        prompt = build_promt(question, chunks)
        answer = generate_answer(prompt, groq_client)
        output = format_response(answer, chunks)

        print(output)


if __name__ == "__main__":
    run_agent()        

    

