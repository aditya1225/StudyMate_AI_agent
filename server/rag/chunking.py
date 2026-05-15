"""
Chunking: splits raw document text into pieces small enough to embed
but large enough to carry meaning.

Why this matters: embedding models compress whatever you give them
into a single fixed-size vector. A 10-page chunk gets averaged into
one point in vector space — useless for retrieval. A 1-sentence
chunk has too little context. Somewhere in between is the sweet spot.
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str         # filename or doc id
    chunk_index: int    # position within the document


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 600,
    overlap: int = 80,
) -> list[Chunk]:
    """
    Split text into overlapping chunks by character count.

    Why character-based instead of token-based:
        - Simple, no tokenizer dependency
        - Roughly 4 chars ≈ 1 token for English, so 600 chars ≈ 150 tokens
        - For tighter control, swap in tiktoken later

    Why overlap:
        - A sentence that gets split across two chunks would lose meaning
          in both. Overlap means the boundary sentence appears in full
          in at least one chunk.

    Why we try to split on paragraph/sentence boundaries:
        - Cutting mid-sentence produces chunks that embed poorly.
        - We prefer breaks at "\\n\\n", then "\\n", then ". ", then a hard cut.
    """
    if not text.strip():
        return []

    # Separators in order of preference — paragraph, line, sentence, word
    separators = ["\n\n", "\n", ". ", " "]

    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # If we're not at the end of the document, try to find a clean break
        if end < len(text):
            for sep in separators:
                # Look for the separator in the last 20% of the chunk
                window_start = end - int(chunk_size * 0.2)
                sep_pos = text.rfind(sep, window_start, end)
                if sep_pos != -1 and sep_pos > start:
                    end = sep_pos + len(sep)
                    break

        chunk_text_str = text[start:end].strip()
        if chunk_text_str:
            chunks.append(Chunk(text=chunk_text_str, source=source, chunk_index=idx))
            idx += 1

        # Advance start, leaving `overlap` characters behind
        # max(...) protects against infinite loop if end <= start + overlap
        start = max(end - overlap, start + 1)

    return chunks