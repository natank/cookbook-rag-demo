# RAG Lecture Demo

A minimal, single-file Retrieval-Augmented Generation (RAG) application built as the
live demo for the "RAG Overview" lecture (see `RAG_lecture.md` in the `ai-career-path`
repo). Its purpose is to make the lecture's concepts — ingestion, embeddings, retrieval,
grounded generation with citations — concrete in a runnable app, not to be a production
system.

## Background

- **Knowledge base:** [`jcallaghan/The-Cookbook`](https://github.com/jcallaghan/The-Cookbook),
  a public GitHub repo of real recipes written in markdown. Chosen deliberately as a
  non-technical, relatable topic (recipes, not API docs) so the demo reads clearly to a
  general lecture audience. Cloned locally into `cookbook/`.
- **Scope decision:** kept intentionally small — one script, not a multi-phase project —
  to stay demoable within a lecture. This is a separate, simpler sibling to the
  `course-rag-agent` project (which covers the same concepts in depth over several
  phases: raw SDK → RAG → LangChain → LangGraph agent → eval).
- **Chunking:** one chunk per recipe file. Each file's YAML frontmatter (`title`, etc.) is
  parsed out and the title used as citation metadata; the markdown body becomes the
  embedded text.

## How it works

```
cookbook/recipes/*.md ──▶ ingest (--ingest) ──▶ OpenAI embeddings ──▶ local Chroma store (.chroma/)

question ──▶ embed question ──▶ similarity search (top-k) ──▶ Claude answers using
                                                                only retrieved recipes,
                                                                citing source file(s)
```

- **Embeddings:** OpenAI `text-embedding-3-small`.
- **Vector store:** local, persisted [Chroma](https://www.trychroma.com/) collection
  (`.chroma/` — gitignored, rebuilt via `--ingest`).
- **Generation:** Anthropic Claude (`claude-sonnet-4-5`), prompted to answer only from the
  retrieved recipe chunks and cite which recipe(s) it used.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and API keys for both Anthropic and OpenAI.

1. Copy `.env.example` to `.env` and fill in your keys (`.env` is gitignored):
   ```
   cp .env.example .env
   ```
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   OPENAI_API_KEY=sk-...
   ```
2. Install dependencies (already declared in `pyproject.toml`):
   ```
   uv sync
   ```

## Usage

Build (or rebuild) the vector store from the recipe corpus:

```
uv run rag_demo.py --ingest
```

Ask a question:

```
uv run rag_demo.py "how do I cook a duck breast?"
uv run rag_demo.py "what desserts can I make?"
```

Each answer prints the grounded response followed by a `Sources:` line listing the
recipe title(s) and filename(s) actually retrieved and used.

## Web UI (Docker/Podman)

A minimal browser UI is also available, backed by a local FastAPI server
(`server.py`) that wraps the same `ingest`/`retrieve`/`answer` functions used by the
CLI. It's containerized for local dev with `docker-compose` (works with `podman-compose`
too):

```
podman-compose up --build
```

Then open [http://localhost:8000](http://localhost:8000), click **Ingest / Rebuild
index** once (equivalent to `--ingest`), and start asking questions.

- Requires `.env` with `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`, same as the CLI (see
  Setup above — copy `.env.example` to `.env` first if you haven't already).
- `cookbook/` and `.chroma/` are bind-mounted into the container, so the ingested index
  and recipe corpus persist across container restarts/rebuilds.
- To run the server locally without a container: `uv run uvicorn server:app --reload`.

## Notes

- Re-run `--ingest` any time `cookbook/recipes/` changes — the store isn't watched or
  auto-refreshed.
- `TOP_K` (default 3) and the models used are configured as constants at the top of
  `rag_demo.py`.
