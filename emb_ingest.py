"""
Doc ingestion Pipeline

Reads all pdfs -> parses into chunks -> embeds each chunk and stores
everything in ChromaDB

"""

import os
import time
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from meta_parser import PDFParser

print("Script Started !!")

#------
# Configuraation
#------

DOCS_DIR = "docs"
VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "Mahesh_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 20
OVERLAP = 4           # Try changing chunksize,overlap to experiment
BATCH_SIZE = 50         # effective outputs

# Setup

def load_embedding_model() -> SentenceTransformer:
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    print(f"  -> Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model

def get_chroma_collection(reset: bool = False) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f" Existing collection deleted")
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name= COLLECTION_NAME,
        metadata={"hnsw:space":"cosine"}
    )
    return collection

# Core Ingestion

def embed_chunks(chunks, model: SentenceTransformer) -> list[list[float]]:
    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True,batch_size=32)
    return embeddings.tolist()

def insert_into_chroma(chunks, embeddings, collection: chromadb.Collection):     
    total = len(chunks)
    inserted = 0

    for i in range(0, total, BATCH_SIZE):
        batch_chunks = chunks[i : i + BATCH_SIZE]
        batch_embeddings = embeddings[i : i + BATCH_SIZE]

        collection.add(
            ids = [c.chunk_id for c in batch_chunks],
            embeddings= batch_embeddings,
            documents= [c.text for c in batch_chunks],
            metadatas= [c.to_metadata() for c in batch_chunks],

        )    
        inserted += len(batch_chunks)
        print(f"  -> Inserted {inserted}/{total} chunks")   


#------------
# Main Pipeline
#------------

def ingest(reset: bool = True):

    print("Script Started !!")
    start = time.time()

    # Step1: Checking pdfs
    pdf_files = sorted(Path(DOCS_DIR).glob("*.pdf*"))
    if not pdf_files:
        print(f'No pdfs found in {DOCS_DIR} - Add documents and try again')
        return
    
    print(f'Found {len(pdf_files)} pdfs in {DOCS_DIR}\n')
    for f in pdf_files:
        print(f' -- {f.name}')

    # Step2: Parsing all pdfs    

    parsing = PDFParser(chunk_size = CHUNK_SIZE, overlap = OVERLAP)
    all_chunks = []

    for pdf_path in pdf_files:
        chunks = parsing.parse(str(pdf_path))
        all_chunks.extend(chunks)

    print(f"\n Total chunks across all documents : {len(all_chunks)}")

    # Deduplicate by chunk_id
    seen = set()
    unique_chunks = []
    for chunk in all_chunks:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            unique_chunks.append(chunk)

    all_chunks = unique_chunks
    print(f"  After deduplication: {len(all_chunks)} unique chunks")

    # Step3: Embed

    model = load_embedding_model()
    embedding = embed_chunks(all_chunks, model)

    print(f"\n Embedded {len(embedding)} chunks successfully")

    # Step4: Store in ChromaDB

    collection = get_chroma_collection(reset=reset)
    insert_into_chroma(all_chunks, embedding, collection)

    # Done
    elapsed = round(time.time() - start, 2)
    total_docs = collection.count()
    print(f"  INGESTION COMPLETE")
    print(f"{'='*55}")
    print(f"  Documents processed : {len(pdf_files)}")
    print(f"  Total chunks stored : {total_docs}")
    print(f"  Time taken          : {elapsed}s")
    print(f"  VectorStore path    : {VECTORSTORE_DIR}/")
    print(f"\n  You can now run agent.py to start querying.")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    ingest(reset=True)