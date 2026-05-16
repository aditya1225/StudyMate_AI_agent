"""
Eval harness for the retrieval layer.

What this measures:
    Recall@k — for each (question, expected_keywords) pair, did at least one
    expected keyword show up in the top_k retrieved chunks? This is a coarse
    proxy for "did retrieval find the right passage", but it's the cheapest
    useful signal: no LLM call, no human grading, runs in seconds.

How to use:
    1. Edit the questions list below (or pass --file path/to/questions.json)
       to reflect your actual corpus.
    2. Ingest your PDFs first.
    3. python eval.py
    4. Tweak chunk_size, top_k, the embedding model, or the query prefix.
       Re-run. Watch the score.

The point of this harness is not absolute scoring — it's the *delta* you see
when you change one thing.
"""

import argparse
import json
from pathlib import Path

from qdrant_client import QdrantClient

from server.rag.retrieval import search

# Edit these to match your corpus. 15-20 questions is enough to feel signal;
# more is better but you'll spend more time writing them than running them.
DEFAULT_QUESTIONS: list[dict] = [
    # {"question": "What is backpropagation?", "keywords": ["chain rule", "gradient"]},
    # {"question": "Define convex function", "keywords": ["second derivative", "epigraph"]},
]


def load_questions(path: Path | None) -> list[dict]:
    if path is None:
        return DEFAULT_QUESTIONS
    return json.loads(path.read_text(encoding="utf-8"))


def hits_any_keyword(text: str, keywords: list[str]) -> bool:
    haystack = text.lower()
    return any(kw.lower() in haystack for kw in keywords)


def evaluate(qdrant: QdrantClient, questions: list[dict], top_k: int) -> None:
    if not questions:
        print("No questions to evaluate. Edit DEFAULT_QUESTIONS or pass --file.")
        return

    passed = 0
    for i, q in enumerate(questions, 1):
        question = q["question"]
        keywords = q["keywords"]

        results = search(qdrant, question, top_k=top_k)
        joined = "\n".join(r["text"] for r in results)
        ok = hits_any_keyword(joined, keywords)

        passed += int(ok)
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] Q{i}: {question}")
        if not ok:
            print(f"        expected one of: {keywords}")
            for r in results[:3]:
                snippet = r["text"][:120].replace("\n", " ")
                page = r.get("page")
                page_str = f" p.{page}" if page else ""
                print(f"        - {r['source']}{page_str} (score={r['score']:.3f}): {snippet}...")

    total = len(questions)
    pct = 100 * passed / total if total else 0.0
    print(f"\nRecall@{top_k}: {passed}/{total} ({pct:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall@k eval for retrieval.")
    parser.add_argument("--file", type=Path, help="JSON file of {question, keywords} pairs")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    args = parser.parse_args()

    qdrant = QdrantClient(host=args.host, port=args.port)
    questions = load_questions(args.file)
    evaluate(qdrant, questions, top_k=args.top_k)


if __name__ == "__main__":
    main()
