"""
Embeddings: convert text into fixed-size vectors so that
semantically similar texts land close together in vector space.

Why a small model (bge-small-en-v1.5):
    - 384 dimensions, ~130MB, runs comfortably on CPU
    - Consistently top-tier on MTEB for its size class
    - Max input 512 tokens — keep chunks under that

Why we lazy-load and cache:
    - The model takes ~5s to load and ~500MB of RAM
    - Loading once at import time means subsequent calls are fast
"""

from functools import lru_cache
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # must match the model; used to create the Qdrant collection


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """
    Load and cache the embedding model.

    lru_cache(maxsize=1) means this function runs exactly once per process,
    regardless of how many times it's called. The first call pays the cost;
    every later call returns the same instance instantly.
    """
    print(f"[embeddings] Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print("[embeddings] Ready.")
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts.

    Batching is much faster than embedding one at a time because the model
    can parallelize across the batch on a single forward pass.
    """
    model = get_embedder()
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=len(texts) > 50,
        convert_to_numpy=True,
        normalize_embeddings=True,  # critical — see below
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """
    Embed a single search query.

    Note on bge models: the authors recommend prefixing queries (but not
    documents) with an instruction for retrieval. This nudges the model
    to produce vectors better suited for query-document matching.
    """
    model = get_embedder()
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    vector = model.encode(
        prefixed,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vector.tolist()