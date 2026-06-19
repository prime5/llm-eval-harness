"""
Phase 5 — RAG Evaluation Harness
Session 2: Retrieval + LLM answer generation pipeline

Usage (CLI):
    python3 phase5/rag_chain.py --query "How do you find a good startup idea?"
    python3 phase5/rag_chain.py --query "What is the role of curiosity in great work?" --top-k 3

Importable by eval_suite.py (Session 3):
    from phase5.rag_chain import RAGChain
    chain = RAGChain()
    result = chain.run("How do you find startup ideas?")
    # result.answer, result.chunks, result.sources
"""

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR = Path(__file__).parent / "data" / "chroma_db"
COLLECTION = "pg_essays"
TOP_K      = 3   # chunks to retrieve per query
MODEL      = "gpt-4o-mini"

SYSTEM_PROMPT = """\
You are a precise assistant. Answer the user's question using ONLY the context \
provided below. Do not use outside knowledge.

If the context does not contain enough information to answer, respond with exactly:
"INSUFFICIENT CONTEXT: [brief reason]"

Be concise. Do not repeat the question. Do not add preamble."""

CONTEXT_TEMPLATE = """\
Context (retrieved passages):
{context}

Question: {question}
Answer:"""


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    text:     str
    source:   str
    chunk_id: int
    distance: float


@dataclass
class RAGResult:
    query:   str
    answer:  str
    chunks:  list[RetrievedChunk]
    sources: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.sources = list(dict.fromkeys(c.source for c in self.chunks))

    def context_text(self) -> str:
        """The exact context string fed to the LLM — used by faithfulness eval."""
        return "\n\n---\n\n".join(
            f"[{c.source}]\n{c.text}" for c in self.chunks
        )


# ── RAG Chain ─────────────────────────────────────────────────────────────────
class RAGChain:
    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k
        self._collection = self._load_collection()
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _load_collection(self) -> chromadb.Collection:
        if not CHROMA_DIR.exists():
            raise RuntimeError(
                f"ChromaDB not found at {CHROMA_DIR}. Run `python3 phase5/ingest.py` first."
            )
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")

        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small",
        )
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return client.get_collection(name=COLLECTION, embedding_function=ef)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Retrieve top-k chunks from ChromaDB for a query."""
        results = self._collection.query(
            query_texts=[query],
            n_results=self.top_k,
        )
        chunks = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(RetrievedChunk(
                text=text,
                source=meta["source"],
                chunk_id=meta["chunk"],
                distance=dist,
            ))
        return chunks

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        """Generate a grounded answer from retrieved chunks."""
        context = "\n\n---\n\n".join(
            f"[{c.source}]\n{c.text}" for c in chunks
        )
        user_message = CONTEXT_TEMPLATE.format(
            context=context,
            question=query,
        )
        response = self._client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0,   # deterministic — eval needs reproducibility
        )
        return response.choices[0].message.content.strip()

    def run(self, query: str) -> RAGResult:
        """Full pipeline: retrieve → generate → return structured result."""
        chunks = self.retrieve(query)
        answer = self.generate(query, chunks)
        return RAGResult(query=query, answer=answer, chunks=chunks)


# ── CLI ───────────────────────────────────────────────────────────────────────
def print_result(result: RAGResult) -> None:
    print(f"\n{'─' * 60}")
    print(f"Query:   {result.query}")
    print(f"Sources: {', '.join(result.sources)}")
    print(f"\nRetrieved chunks ({len(result.chunks)}):")
    for i, c in enumerate(result.chunks):
        print(f"  [{i+1}] {c.source} chunk {c.chunk_id} (distance: {c.distance:.4f})")
        print(f"       {c.text[:100]}...")
    print(f"\nAnswer:\n{result.answer}")
    print(f"{'─' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="RAG chain — retrieve + generate")
    parser.add_argument("--query",  required=True, help="Question to answer")
    parser.add_argument("--top-k",  type=int, default=TOP_K,
                        help=f"Chunks to retrieve (default: {TOP_K})")
    args = parser.parse_args()

    chain = RAGChain(top_k=args.top_k)
    result = chain.run(args.query)
    print_result(result)


if __name__ == "__main__":
    main()
