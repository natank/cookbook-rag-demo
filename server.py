"""
Local API server for the RAG demo web UI.

Wraps rag_demo.py's ingest/retrieve/answer functions behind a small FastAPI app and
serves the static UI in web/.

Usage:
    uv run uvicorn server:app --reload
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import rag_demo

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="RAG Lecture Demo")


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    title: str
    filename: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class IngestResponse(BaseModel):
    status: str
    count: int


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/ingest", response_model=IngestResponse)
def ingest():
    count = rag_demo.ingest()
    return {"status": "ok", "count": count}


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not rag_demo.CHROMA_DIR.exists():
        raise HTTPException(
            status_code=503,
            detail="No vector store found — run ingest first.",
        )
    return rag_demo.answer(request.question)


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
