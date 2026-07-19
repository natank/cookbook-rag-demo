"""
Minimal RAG demo for the RAG lecture.

Knowledge base: jcallaghan/The-Cookbook (cloned into ./cookbook/recipes/*.md).
One chunk per recipe file, embedded with OpenAI, stored in a local persisted
Chroma collection, retrieved by similarity, answered by Claude with citations.

Usage:
    uv run rag_demo.py --ingest          # build/refresh the vector store
    uv run rag_demo.py "your question"   # ask a question
"""

import argparse
import sys
from pathlib import Path

import chromadb
import yaml
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

RECIPES_DIR = Path(__file__).parent / "cookbook" / "recipes"
CHROMA_DIR = Path(__file__).parent / ".chroma"
COLLECTION_NAME = "cookbook_recipes"
EMBEDDING_MODEL = "text-embedding-3-small"
CLAUDE_MODEL = "claude-sonnet-4-5"
TOP_K = 3

_openai_client = None
_anthropic_client = None


def openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def anthropic_client() -> Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = Anthropic()
    return _anthropic_client


def load_recipes() -> list[dict]:
    """Read every recipe file, split YAML frontmatter from body, one chunk per file."""
    recipes = []
    for path in sorted(RECIPES_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = path.stem.replace("-", " ").title()
        if text.startswith("---"):
            _, frontmatter, body = text.split("---", 2)
            meta = yaml.safe_load(frontmatter) or {}
            title = meta.get("title", title)
        else:
            body = text
        recipes.append({"filename": path.name, "title": title, "text": body.strip()})
    return recipes


def ingest() -> int:
    recipes = load_recipes()
    print(f"Loaded {len(recipes)} recipe files from {RECIPES_DIR}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    client.delete_collection(COLLECTION_NAME) if COLLECTION_NAME in [
        c.name for c in client.list_collections()
    ] else None
    collection = client.create_collection(COLLECTION_NAME)

    texts = [r["text"] for r in recipes]
    embeddings = (
        openai_client()
        .embeddings.create(model=EMBEDDING_MODEL, input=texts)
    )

    collection.add(
        ids=[r["filename"] for r in recipes],
        embeddings=[e.embedding for e in embeddings.data],
        documents=texts,
        metadatas=[{"title": r["title"], "filename": r["filename"]} for r in recipes],
    )
    print(f"Embedded and stored {len(recipes)} chunks in {CHROMA_DIR}")
    return len(recipes)


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = (
        openai_client()
        .embeddings.create(model=EMBEDDING_MODEL, input=[question])
        .data[0]
        .embedding
    )
    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    return [
        {"title": meta["title"], "filename": meta["filename"], "text": doc}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


def answer(question: str) -> dict:
    chunks = retrieve(question)
    context = "\n\n---\n\n".join(
        f"Recipe: {c['title']} (file: {c['filename']})\n{c['text']}" for c in chunks
    )

    prompt = (
        "Answer the question using only the recipes provided below. "
        "Cite which recipe(s) you used by title.\n\n"
        f"Recipes:\n{context}\n\nQuestion: {question}"
    )

    response = anthropic_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "answer": response.content[0].text,
        "sources": [{"title": c["title"], "filename": c["filename"]} for c in chunks],
    }


def main():
    parser = argparse.ArgumentParser(description="RAG demo over The-Cookbook recipes")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--ingest", action="store_true", help="Build the vector store")
    args = parser.parse_args()

    if args.ingest:
        ingest()
        return

    if not args.question:
        parser.error("Provide a question, or run with --ingest first")

    if not CHROMA_DIR.exists():
        print("No vector store found — run with --ingest first.", file=sys.stderr)
        sys.exit(1)

    result = answer(args.question)
    print(result["answer"])
    sources = ", ".join(f"{s['title']} ({s['filename']})" for s in result["sources"])
    print("\nSources:", sources)


if __name__ == "__main__":
    main()
