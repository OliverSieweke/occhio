"""Singleton FastMCP instance shared across the occhio MCP server.

Kept in its own module to avoid circular imports — this file must not
import anything from the rest of the occhio package.
"""

from fastmcp import FastMCP

mcp: FastMCP = FastMCP("occhio_mcp")
