"""
Phase 5 — RAG Evaluation Harness
Session 1: Ingest Paul Graham essays into ChromaDB

Usage:
    export OPENAI_API_KEY=sk-...
    python phase5/ingest.py

Loads .txt files from phase5/data/essays/, splits into chunks,
embeds with OpenAI text-embedding-3-small, persists to ChromaDB.
To use real Paul Graham essays: drop .txt files into phase5/data/essays/
and re-run this script.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # loads .env from project root

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Config ────────────────────────────────────────────────────────────────────
ESSAYS_DIR    = Path(__file__).parent / "data" / "essays"
CHROMA_DIR    = Path(__file__).parent / "data" / "chroma_db"
COLLECTION    = "pg_essays"
CHUNK_SIZE    = 400   # ~500 chars — small enough for precise retrieval
CHUNK_OVERLAP = 50


def load_essays(essays_dir: Path) -> list[dict]:
    """Load all .txt essay files. Returns list of {source, text} dicts."""
    docs = []
    for path in sorted(essays_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if len(text) < 100:
            print(f"  skip {path.name} (stub: {len(text)} chars)")
            continue
        docs.append({"source": path.stem, "text": text})
        print(f"  loaded {path.name}: {len(text)} chars")
    return docs


def chunk_docs(docs: list[dict]) -> list[dict]:
    """Split each essay into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = []
    for doc in docs:
        parts = splitter.split_text(doc["text"])
        for i, part in enumerate(parts):
            chunks.append({
                "id":     f"{doc['source']}_chunk_{i:03d}",
                "text":   part,
                "source": doc["source"],
                "chunk":  i,
            })
    return chunks


def build_collection(chunks: list[dict]) -> chromadb.Collection:
    """Embed chunks and persist to ChromaDB."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Clean re-ingest
    try:
        client.delete_collection(COLLECTION)
        print(f"  deleted existing '{COLLECTION}' collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    ids       = [c["id"]   for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "chunk": c["chunk"]} for c in chunks]

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
        )
        print(f"  embedded batch {i // batch_size + 1}/{-(-len(chunks) // batch_size)}")

    return collection


def verify_retrieval(collection: chromadb.Collection) -> None:
    """Smoke-test: run 3 queries and print top result for each."""
    queries = [
        "How do you find a good startup idea?",
        "What is the relationship between writing and thinking?",
        "How should you approach hard work?",
    ]
    print("\n── Retrieval smoke test ──────────────────────────────────────────────")
    for q in queries:
        results = collection.query(query_texts=[q], n_results=2)
        top_doc  = results["documents"][0][0]
        top_meta = results["metadatas"][0][0]
        top_dist = results["distances"][0][0]
        print(f"\nQ: {q}")
        print(f"  source: {top_meta['source']} | chunk: {top_meta['chunk']} | distance: {top_dist:.4f}")
        print(f"  text:   {top_doc[:120]}...")


def main():
    print("── Loading essays ────────────────────────────────────────────────────")
    docs = load_essays(ESSAYS_DIR)
    if not docs:
        print("ERROR: no essays found. Add .txt files to phase5/data/essays/")
        sys.exit(1)
    print(f"  {len(docs)} essays loaded\n")

    print("── Chunking ──────────────────────────────────────────────────────────")
    chunks = chunk_docs(docs)
    print(f"  {len(chunks)} chunks created\n")

    print("── Embedding + indexing into ChromaDB ───────────────────────────────")
    collection = build_collection(chunks)
    print(f"  collection '{COLLECTION}' ready: {collection.count()} vectors\n")

    verify_retrieval(collection)
    print(f"\n✓ Session 1 complete — corpus ingested and retrieval verified.")
    print(f"  ChromaDB persisted at: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
