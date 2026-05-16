"""
Thin wrapper around the Anthropic API for the chat client.

Why pull this out of chat.py:
    - The system prompt and model choice are the things you actually want to
      tweak; keeping them next to the loop makes both noisier.
    - Tests / eval scripts can import LLM without dragging in the MCP bridge.
"""

from anthropic import Anthropic

DEFAULT_MODEL = "claude-sonnet-4-5"

SYSTEM_PROMPT = """\
You are StudyBuddy, a study assistant grounded in the user's personal notes
and textbooks.

Rules:
- For anything the user could plausibly have notes on, call `search_notes`
  before answering. Don't rely on your own knowledge when their corpus could
  contain the answer.
- If the first search misses, try a reworded query before giving up. You can
  also call `list_documents` to see what's available.
- Cite every claim you take from the corpus inline as
  `(source.pdf, p. <page>)`. If a chunk has no page, cite just the source.
- If the search returns nothing useful, say so plainly — don't invent.
- Keep answers concise. The user is studying, not reading prose.
"""


class LLM:
    def __init__(self, model: str = DEFAULT_MODEL, system: str = SYSTEM_PROMPT):
        self._client = Anthropic()
        self.model = model
        self.system = system

    def respond(self, messages: list[dict], tools: list[dict], max_tokens: int = 2048):
        """One Anthropic call. Returns the raw response so the caller can
        inspect stop_reason and content blocks."""
        return self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system,
            tools=tools,
            messages=messages,
        )
