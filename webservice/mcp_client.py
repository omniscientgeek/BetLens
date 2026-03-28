"""
MCP Client — Async client for the betstamp-intelligence MCP server.

Provides on-demand connections to the MCP server via stdio transport,
tool discovery, and tool execution. Designed for use with the Anthropic
API tool-use loop so that the direct API provider can call MCP tools
without needing the Claude Agent SDK / Node.js wrapper.
"""

import os
import sys
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

logger = logging.getLogger(__name__)

# Resolve the MCP server script path relative to this file's location
# webservice/ -> ../mcp-server/mcp_server.py
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MCP_SERVER_SCRIPT = os.path.join(_PROJECT_ROOT, "mcp-server", "mcp_server.py")

# Maximum characters to keep from a tool result before truncating
# (prevents context window blowout from huge MCP responses)
MAX_TOOL_RESULT_CHARS = 200_000


class McpClient:
    """Async MCP client that spawns the betstamp-intelligence server on demand.

    Usage::

        client = McpClient()
        async with client.connect() as session:
            tools = await client.get_anthropic_tools(session)
            result = await client.call_tool(session, "list_events", {"filename": "data.json"})
    """

    def __init__(self, server_script: Optional[str] = None):
        self._server_script = server_script or _MCP_SERVER_SCRIPT

    @asynccontextmanager
    async def connect(self):
        """Context manager that spawns the MCP server and yields a ClientSession.

        The subprocess is automatically cleaned up when the context exits.
        """
        if not os.path.isfile(self._server_script):
            raise RuntimeError(
                f"MCP server script not found at {self._server_script}. "
                "Ensure the mcp-server/ directory exists."
            )

        # Use the same Python interpreter that's running the webservice
        python_bin = sys.executable or "python"

        server_params = StdioServerParameters(
            command=python_bin,
            args=[self._server_script],
            cwd=_PROJECT_ROOT,
        )

        logger.info("Connecting to MCP server: %s", self._server_script)

        try:
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    logger.info("MCP session initialized")
                    yield session
        except BaseExceptionGroup as eg:
            # anyio TaskGroup wraps subprocess/connection errors in an
            # ExceptionGroup — unwrap to surface the real cause.
            real = eg.exceptions[0] if len(eg.exceptions) == 1 else eg
            logger.error("MCP connection failed (unwrapped): %s: %s", type(real).__name__, real)
            raise RuntimeError(f"MCP connection failed: {type(real).__name__}: {real}") from real

    async def get_tools(self, session: ClientSession) -> list:
        """Fetch tool definitions from the MCP server.

        Returns a list of MCP Tool objects.
        """
        result = await session.list_tools()
        tools = result.tools if hasattr(result, "tools") else result
        logger.info("MCP server exposes %d tools", len(tools))
        return tools

    @staticmethod
    def tools_to_anthropic_format(mcp_tools: list) -> list[dict]:
        """Convert MCP tool definitions to Anthropic API ``tools`` parameter format.

        MCP and Anthropic both use JSON Schema for input definitions, so the
        conversion is mostly structural renaming.
        """
        anthropic_tools = []
        for tool in mcp_tools:
            # MCP Tool has: name, description, inputSchema (dict)
            input_schema = getattr(tool, "inputSchema", None) or {}
            # Ensure the schema has a top-level type
            if "type" not in input_schema:
                input_schema["type"] = "object"
            anthropic_tools.append({
                "name": tool.name,
                "description": getattr(tool, "description", "") or "",
                "input_schema": input_schema,
            })
        return anthropic_tools

    async def call_tool(
        self,
        session: ClientSession,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Execute an MCP tool and return the text result.

        Returns the concatenated text content from the tool response.
        If the tool returns an error, the error message is returned as text
        (so the AI can handle it gracefully rather than crashing the loop).
        """
        logger.info("MCP tool call: %s(%s)", tool_name, json.dumps(arguments)[:200])

        try:
            result = await session.call_tool(tool_name, arguments)
        except Exception as exc:
            error_msg = f"MCP tool error ({tool_name}): {exc}"
            logger.warning(error_msg)
            return json.dumps({"error": error_msg})

        # Extract text from content blocks
        parts = []
        for block in (result.content or []):
            if hasattr(block, "text"):
                parts.append(block.text)
            elif hasattr(block, "data"):
                # Binary/blob content — skip or summarize
                parts.append(f"[binary data: {len(block.data)} bytes]")

        text = "\n".join(parts)

        # Check for tool-level error flag
        if getattr(result, "isError", False):
            logger.warning("MCP tool %s returned error: %s", tool_name, text[:200])

        # Truncate very large results to avoid context window blowout
        if len(text) > MAX_TOOL_RESULT_CHARS:
            logger.warning(
                "MCP tool %s result truncated: %d -> %d chars",
                tool_name, len(text), MAX_TOOL_RESULT_CHARS,
            )
            text = text[:MAX_TOOL_RESULT_CHARS] + "\n\n[... result truncated due to size ...]"

        logger.info("MCP tool %s returned %d chars", tool_name, len(text))
        return text


# Module-level singleton for convenience
mcp_client = McpClient()
