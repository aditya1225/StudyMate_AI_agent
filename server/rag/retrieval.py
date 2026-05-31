"""
Retrieval: given a user query, find the most relevant chunks in Qdrant.

The shape of this module:
    - search(): the main RAG retrieval call
    - list_docs(): metadata helper for the MCP tool
    - Both take the qdrant client and embedder as arguments rather than
      creating them globally. This makes the module testable and avoids
      hidden state.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from .embeddings import embed_query

COLLECTION_NAME = "studybuddy"


def search(
    qdrant: QdrantClient,
    query: str,
    top_k: int = 5,
    source_filter: str | None = None,
) -> list[dict]:
    """
    Semantic search over the corpus.

    Returns a list of dicts with text, source, and score, ready to be
    serialized back to the MCP client.

    Why top_k = 5 by default:
        - Enough context for most questions without overwhelming the LLM
        - "Lost in the middle" effect kicks in past ~10 chunks
        - Easy to override per-call from the MCP tool

    source_filter lets the user say "only search inside chapter_3.pdf".
    Qdrant filters happen *during* the vector search, not after — so this
    is efficient even on large collections.
    """
    query_vector = embed_query(query)

    qdrant_filter = None
    if source_filter:
        qdrant_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
        )

    # qdrant-client >=1.13 removed .search() in favor of .query_points().
    # The shape differs slightly: query_points() returns a QueryResponse
    # whose `.points` field holds the hits.
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    return [
        {
            "text": hit.payload["text"],
            "source": hit.payload["source"],
            "chunk_index": hit.payload["chunk_index"],
            "page": hit.payload.get("page"),
            "score": float(hit.score),
        }
        for hit in response.points
    ]


def list_docs(qdrant: QdrantClient) -> list[str]:
    """
    Return the list of unique source documents in the collection.

    Qdrant doesn't have a native "distinct" operation, so we scroll
    through all points and dedupe in Python. For a personal corpus
    (hundreds of docs, thousands of chunks) this is fine. At larger
    scale you'd maintain a separate "documents" collection.
    """
    sources: set[str] = set()
    next_offset = None

    while True:
        points, next_offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=next_offset,
            with_payload=["source"],   # only fetch the field we need
            with_vectors=False,         # skip the heavy data
        )
        for p in points:
            if p.payload and "source" in p.payload:
                sources.add(p.payload["source"])
        if next_offset is None:
            break

    return sorted(sources)