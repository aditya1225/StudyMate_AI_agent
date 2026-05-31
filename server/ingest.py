"""
Ingest: PDF -> text -> chunks -> embeddings -> Qdrant.

The pipeline:
    1. Read each page of the PDF with pypdf
    2. Chunk each page individually so we can attach a page number to every chunk
    3. Embed the chunks in a single batch
    4. Upsert into Qdrant with payload {text, source, chunk_index, page}

Why per-page chunking:
    The simplest way to keep an honest page number on each chunk is to never let
    a chunk straddle a page boundary. We lose a little cross-page context, but
    citations stay correct, which matters more for a study assistant.

Re-ingesting the same file is safe: we delete the existing points for that
source before upserting the new ones.
"""

import argparse
import sys
import uuid
from functools import partial
from pathlib import Path

# All [ingest] prints go to stderr so this module can be safely imported
# and called from the MCP server (whose stdout is the JSONRPC wire).
log = partial(print, file=sys.stderr)

from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .rag.chunking import Chunk, chunk_text
from .rag.embeddings import EMBEDDING_DIM, embed_texts
from .rag.retrieval import COLLECTION_NAME


def ensure_collection(qdrant: QdrantClient) -> None:
    """Create the Qdrant collection on first run."""
    if qdrant.collection_exists(COLLECTION_NAME):
        return
    log(f"[ingest] Creating collection {COLLECTION_NAME!r} (dim={EMBEDDING_DIM}, cosine)")
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


def pdf_to_pages(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number_1_indexed, page_text), ...]."""
    reader = PdfReader(str(path))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def chunks_with_pages(
    pages: list[tuple[int, str]],
    source: str,
) -> list[tuple[Chunk, int]]:
    """Chunk each page and pair the chunk with the page it came from."""
    out: list[tuple[Chunk, int]] = []
    running_idx = 0
    for page_num, page_text in pages:
        for c in chunk_text(page_text, source=source):
            # Rewrite chunk_index to be unique across the whole document
            # rather than per-page.
            out.append((Chunk(c.text, c.source, running_idx), page_num))
            running_idx += 1
    return out


def delete_existing(qdrant: QdrantClient, source: str) -> None:
    """Drop any existing points for this source so re-ingest is idempotent."""
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source))]
        ),
    )


def ingest_pdf(qdrant: QdrantClient, path: Path) -> int:
    """Ingest a single PDF. Returns the number of chunks upserted."""
    source = path.name
    log(f"[ingest] {source}")

    pages = pdf_to_pages(path)
    log(f"[ingest]   {len(pages)} pages")

    pairs = chunks_with_pages(pages, source=source)
    if not pairs:
        log(f"[ingest]   no text extracted, skipping")
        return 0

    log(f"[ingest]   {len(pairs)} chunks, embedding...")
    vectors = embed_texts([c.text for c, _ in pairs])

    delete_existing(qdrant, source)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "text": chunk.text,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "page": page,
            },
        )
        for (chunk, page), vec in zip(pairs, vectors)
    ]

    # Batch the upsert. A single 5000-point HTTP call easily blows past the
    # default httpx read timeout; 128 per batch keeps each round-trip small.
    batch_size = 128
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
        log(f"[ingest]   upserted {i + len(batch)}/{len(points)}")
    return len(points)


def ingest_paths(qdrant: QdrantClient, paths: list[Path]) -> int:
    """Ingest a list of PDF files or directories. Returns total chunk count."""
    pdfs: list[Path] = []
    for p in paths:
        if p.is_dir():
            pdfs.extend(sorted(p.glob("*.pdf")))
        elif p.suffix.lower() == ".pdf":
            pdfs.append(p)
        else:
            log(f"[ingest] skipping non-PDF: {p}")

    if not pdfs:
        log("[ingest] no PDFs found")
        return 0

    ensure_collection(qdrant)
    total = 0
    for path in pdfs:
        total += ingest_pdf(qdrant, path)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into Qdrant for RAG.")
    parser.add_argument("paths", nargs="+", help="PDF file(s) or directory of PDFs")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    args = parser.parse_args()

    qdrant = QdrantClient(host=args.host, port=args.port)
    total = ingest_paths(qdrant, [Path(p) for p in args.paths])
    log(f"[ingest] done. {total} chunks ingested.")


if __name__ == "__main__":
    main()