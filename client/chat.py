import asyncio

from dotenv import load_dotenv

load_dotenv()  # picks up ANTHROPIC_API_KEY from .env before Anthropic() reads env

from .llm import LLM
from .mcp_client import MCPBridge


async def chat():
    llm = LLM()
    async with MCPBridge(["python", "-m", "server.main"]) as mcp:
        tools = await mcp.list_tools()
        messages: list[dict] = []

        while True:
            user = input("\nYou: ").strip()
            if not user:
                break
            messages.append({"role": "user", "content": user})

            # Inner loop: keep going until Claude stops calling tools
            while True:
                try:
                    resp = llm.respond(messages, tools)
                except Exception as e:
                    # Don't kill the MCP session over a transient API error
                    # (rate limit, no credits, network blip). Drop the
                    # unanswered user turn so the next prompt is clean.
                    print(f"\n[llm error] {e}")
                    messages.pop()
                    break
                messages.append({"role": "assistant", "content": resp.content})

                if resp.stop_reason != "tool_use":
                    for block in resp.content:
                        if block.type == "text":
                            print(f"\nClaude: {block.text}")
                    break

                tool_results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        print(f"[calling {block.name}({block.input})]")
                        output = await mcp.call_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "\n".join(output),
                        })
                messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    asyncio.run(chat())