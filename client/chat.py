import asyncio
from anthropic import Anthropic
from mcp_client import MCPBridge

anthropic = Anthropic()

async def chat():
    async with MCPBridge(["python", "-m", "server.main"]) as mcp:
        tools = await mcp.list_tools()
        messages = []

        while True:
            user = input("\nYou: ").strip()
            if not user:
                break
            messages.append({"role": "user", "content": user})

            # Inner loop: keep going until Claude stops calling tools
            while True:
                resp = anthropic.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=2048,
                    tools=tools,
                    messages=messages,
                )
                messages.append({"role": "assistant", "content": resp.content})

                if resp.stop_reason != "tool_use":
                    # Print final text answer
                    for block in resp.content:
                        if block.type == "text":
                            print(f"\nClaude: {block.text}")
                    break

                # Execute every tool call Claude requested
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