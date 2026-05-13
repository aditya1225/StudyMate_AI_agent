from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPBridge:
    def __init__(self, server_cmd: list[str]):
        self.params = StdioServerParameters(command=server_cmd[0], args=server_cmd[1:])
        self.session = None

    async def __aenter__(self):
        self._stdio_cm = stdio_client(self.params)
        read, write = await self._stdio_cm.__aenter__()
        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        await self.session.initialize()
        return self

    async def list_tools(self):
        """Returns tools in Anthropic API format."""
        result = await self.session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.inputSchema,
            }
            for t in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict):
        result = await self.session.call_tool(name, arguments)
        return [c.text for c in result.content if hasattr(c, "text")]

    async def __aexit__(self, *args):
        await self.session.__aexit__(*args)
        await self._stdio_cm.__aexit__(*args)