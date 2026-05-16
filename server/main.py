from pathlib import Path

from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient

from .ingest import ensure_collection, ingest_pdf
from .rag.embeddings import get_embedder
from .rag.retrieval import list_docs, search

mcp = FastMCP("studybuddy")
qdrant = QdrantClient(host="localhost", port=6333)
# Don't eager-load the embedder at import time: the MCP client expects a
# prompt initialize response over stdio, and a 5-10s block here looks like
# a dead server to it. The first search_notes call will load it lazily.


@mcp.tool()
def search_notes(query: str, top_k: int = 5, source: str | None = None) -> list[dict]:
    """Search textbook/notes for passages relevant to the query."""
    return search(qdrant, query, top_k=top_k, source_filter=source)


@mcp.tool()
def list_documents() -> list[str]:
    """List all ingested documents."""
    return list_docs(qdrant)


@mcp.tool()
def add_document(path: str) -> dict:
    """Ingest a PDF at `path` into the corpus. Returns {source, chunks}."""
    p = Path(path)
    if not p.exists() or p.suffix.lower() != ".pdf":
        return {"error": f"not a PDF or does not exist: {path}"}
    ensure_collection(qdrant)
    chunks = ingest_pdf(qdrant, p)
    return {"source": p.name, "chunks": chunks}


if __name__ == "__main__":
    mcp.run()
