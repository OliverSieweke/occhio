"""Occhio MCP server — exposes research tools to agents."""

import importlib
import pkgutil

from occhio.mcp.instance import mcp
from occhio.mcp import servers


def _register_tools() -> None:
    """Auto-discover and register tools from all sub-servers."""
    for _, name, _ in pkgutil.iter_modules(servers.__path__):
        importlib.import_module(f"occhio.mcp.servers.{name}")


def main() -> None:
    """CLI entrypoint for occhio-mcp."""
    _register_tools()
    mcp.run()  # Default: stdio transport


if __name__ == "__main__":
    main()
