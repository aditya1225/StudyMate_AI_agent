"""
Interactive chat loop.

Architecture:
    chat.py (this file)
        -> LangGraph ReAct agent (handles the tool-use loop)
            -> ChatAnthropic / ChatOpenAI / ... (from llm.py)
            -> MCP tools (from langchain-mcp-adapters)

The MCP server is launched as a subprocess by the adapter. The adapter
takes the same stdio command we used to drive ourselves, but exposes the
tools as native LangChain tools the agent can invoke.
"""

import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()  # ANTHROPIC_API_KEY (or whichever provider) must be set before make_llm()

from langchain_core.messages import AIMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from .llm import make_llm, make_system_message


def _print_usage(new_messages: list) -> None:
    """Sum token usage across all AI messages produced this turn."""
    inp = out = cache_r = cache_w = 0
    for m in new_messages:
        if not isinstance(m, AIMessage):
            continue
        u = getattr(m, "usage_metadata", None) or {}
        inp += u.get("input_tokens", 0)
        out += u.get("output_tokens", 0)
        details = u.get("input_token_details", {}) or {}
        cache_r += details.get("cache_read", 0)
        cache_w += details.get("cache_creation", 0)
    if inp or out:
        print(f"[tokens: in={inp} out={out} cache_read={cache_r} cache_write={cache_w}]")


async def chat():
    llm = make_llm()

    # Launch the MCP server as a subprocess and expose its tools to LangChain.
    # sys.executable ensures the subprocess uses the same venv interpreter.
    mcp_client = MultiServerMCPClient({
        "studybuddy": {
            "command": sys.executable,
            "args": ["-m", "server.main"],
            "transport": "stdio",
        }
    })
    tools = await mcp_client.get_tools()
    print(f"[loaded {len(tools)} tool(s) from MCP: {', '.join(t.name for t in tools)}]")

    agent = create_react_agent(llm, tools, prompt=make_system_message())

    messages: list = []
    while True:
        user = input("\nYou: ").strip()
        if not user:
            break
        messages.append(HumanMessage(content=user))
        prior_len = len(messages)

        try:
            result = await agent.ainvoke({"messages": messages})
        except Exception as e:
            # Don't kill the loop on transient API errors. Drop the
            # unanswered user turn so the next prompt is clean.
            print(f"\n[llm error] {e}")
            messages.pop()
            continue

        messages = result["messages"]
        new_messages = messages[prior_len:]

        # Surface what tools were called this turn — keeps the trace visible.
        for m in new_messages:
            if isinstance(m, AIMessage) and m.tool_calls:
                for tc in m.tool_calls:
                    print(f"[calling {tc['name']}({tc['args']})]")

        _print_usage(new_messages)

        # Final answer is the last AIMessage's text content.
        last = messages[-1]
        if isinstance(last, AIMessage):
            text = last.content if isinstance(last.content, str) else "".join(
                block.get("text", "")
                for block in last.content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            if text:
                print(f"\nClaude: {text}")


if __name__ == "__main__":
    asyncio.run(chat())
