from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient

from .rag.retrieval import search, list_docs
from .rag.embeddings import get_embedder

mcp = FastMCP("studybuddy")
qdrant = QdrantClient(host="localhost", port=6333)
get_embedder()  # warm up the model at startup

@mcp.tool()
def search_notes(query: str, top_k: int = 5, source: str | None = None) -> list[dict]:
    """Search textbook/notes for passages relevant to the query."""
    return search(qdrant, query, top_k=top_k, source_filter=source)

@mcp.tool()
def list_documents() -> list[str]:
    """List all ingested documents."""
    return list_docs(qdrant)

if __name__ == "__main__":
    mcp.run()