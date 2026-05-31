"""
LLM client — provider-agnostic via LangChain.

The chat loop (chat.py) only depends on `make_llm()` returning a
LangChain `BaseChatModel`. Swap the implementation here to change
providers without touching the agent loop.

Why LangChain at all:
    LangChain normalizes message shapes, tool calls, and streaming across
    providers. The same `create_react_agent` works against Anthropic,
    OpenAI, Google, or a local Ollama model with one import + one
    constructor change.

What's still Anthropic-specific:
    Prompt caching (`cache_control`) is an Anthropic feature. Other
    providers will ignore the cache_control hint on the system prompt.
    When you swap providers you lose that cost optimization automatically.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

DEFAULT_MODEL = "claude-sonnet-4-5"

SYSTEM_PROMPT_TEXT = """\
You are StudyBuddy, a study assistant grounded ONLY in the user's personal
notes and textbooks. You are not allowed to answer from your own prior
knowledge.

Rules:
- Always call `search_notes` first. If the first query misses, try one or
  two reworded queries. You can also call `list_documents` to see what's
  available.
- If `search_notes` errors or returns nothing useful, say so plainly and
  STOP. Do not fall back to your own knowledge, even if you know the
  answer. Report the failure and let the user decide what to do.
- Cite every claim you take from the corpus inline as
  `(source.pdf, p. <page>)`. If a chunk has no page, cite just the source.
- Keep answers concise. The user is studying, not reading prose.
"""


def make_llm() -> BaseChatModel:
    """Build the chat model. This is the only place to swap providers.

    Current: Anthropic Claude Sonnet 4.5.

    To swap:
        # OpenAI
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o", temperature=0)

        # Google Gemini
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-1.5-pro")

        # Local via Ollama (free, runs on your machine)
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.1:8b")

    Each provider reads its own API key from env:
        ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, ...
    """
    return ChatAnthropic(
        model=DEFAULT_MODEL,
        max_tokens=2048,
        timeout=60,
    )


def make_system_message() -> SystemMessage:
    """Build the system message with provider-aware caching.

    The `cache_control` block tells Anthropic to cache everything up to and
    including the system prompt (tools render first, so they're cached
    too). Other providers receive a plain text system prompt — the
    cache_control field is silently ignored on the LangChain side.
    """
    return SystemMessage(
        content=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT_TEXT,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    )
