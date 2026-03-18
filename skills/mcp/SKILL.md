---
name: mcp
description: MCP server development and tool creation for occhio. Use when adding, testing, or extending MCP tools and server modules.
---

# MCP Server Development

This skill covers creating and extending the occhio MCP (Model Context Protocol) server, which exposes research tools to AI agents.

## Overview

The MCP server lives in `src/occhio/mcp/` and is structured for easy extension:

```
src/occhio/mcp/
├── __init__.py
├── server.py              # FastMCP server + CLI entrypoint
├── graphql_client.py      # Shared async helpers (e.g., GraphQL)
└── servers/
    └── alignmentforum.py  # Tool definitions for a specific API
```

## Quick Commands

```bash
# Install MCP dependencies
uv sync --extra mcp

# Run the server (stdio mode)
uv run --extra mcp occhio-mcp

# Test with MCP Inspector
uv run --extra mcp fastmcp dev inspector src/occhio/mcp/server.py

# Run unit tests
.venv/bin/pytest tests/test_mcp/ -v
```

## Adding a New Tool Module

### 1. Create the Tool File

Create a new file in `src/occhio/mcp/servers/`, e.g., `arxiv.py`:

```python
"""ArXiv tools for MCP server."""

import json
from occhio.mcp.server import mcp

# Read-only tool annotations
READ_ONLY_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,  # External API
}

@mcp.tool(name="arxiv_search", annotations=READ_ONLY_ANNOTATIONS)
async def arxiv_search(query: str, limit: int = 10) -> str:
    """Search arXiv papers.

    Args:
        query: Search query string.
        limit: Maximum results (1-50).

    Returns:
        JSON list of paper summaries.
    """
    # Implementation...
    return json.dumps(results, indent=2)
```

### 2. Register the Module

Add the import to `server.py`:

```python
def _register_tools() -> None:
    from occhio.mcp.servers import alignmentforum  # noqa: F401
    from occhio.mcp.servers import arxiv  # noqa: F401 — add new modules here
```

### 3. Add Tests

Create `tests/test_mcp/test_arxiv.py`:

```python
import json
from unittest.mock import AsyncMock, patch
import pytest

from occhio.mcp.servers.arxiv import arxiv_search

class TestArxivSearch:
    @pytest.mark.asyncio
    async def test_returns_papers(self):
        with patch("occhio.mcp.servers.arxiv.some_client", new_callable=AsyncMock) as mock:
            mock.return_value = [{"title": "Test Paper"}]
            result = await arxiv_search(query="test")

        parsed = json.loads(result)
        assert len(parsed) >= 1
```

**Important**: Test directory must NOT be named `mcp/` as it shadows the `mcp` package. Use `test_mcp/` or similar.

## MCP Configuration

### Canonical Config Location

The MCP server config lives at `.ai/mcp/mcp.json`. It uses absolute paths to ensure it works regardless of where the IDE launches the server from:

```json
{
  "mcpServers": {
    "occhio": {
      "command": "uv",
      "args": [
        "run",
        "--extra", "mcp",
        "--directory", "/Users/os/dev/professional/lasr/occhio",
        "python", "-m", "occhio.mcp.server"
      ],
      "type": "stdio"
    }
  }
}
```

**Important**: Update the `--directory` path to match your project location.

**Alternative (CLI script)**: If you're running from the project root in a terminal:
```json
{
  "mcpServers": {
    "occhio": {
      "command": "uv",
      "args": ["run", "--extra", "mcp", "occhio-mcp"],
      "type": "stdio"
    }
  }
}
```
This simpler version only works when `cwd` is the project root.

### Agent Symlinks

Symlinks point various agent config locations to the canonical file:

- `.claude/mcp.json` → `../.ai/mcp/mcp.json`
- `.cursor/mcp.json` → `../.ai/mcp/mcp.json`
- `.windsurf/mcp.json` → `../.ai/mcp/mcp.json`
- `.codex/mcp.json` → `../.ai/mcp/mcp.json`

To add a new agent, create its symlink:

```bash
mkdir -p .newagent
ln -sf ../.ai/mcp/mcp.json .newagent/mcp.json
```

## Tool Guidelines

### Tool Annotations

All read-only tools should include:

```python
annotations={
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,  # If calling external APIs
}
```

### Return Format

- Always return JSON strings (tools must return strings)
- Use `json.dumps(data, indent=2)` for readable output
- Include error information in the JSON on failures:
  ```python
  return json.dumps({"error": "Description of what went wrong"})
  ```

### Parameter Validation

- Clamp numeric parameters to sensible ranges:
  ```python
  limit = max(1, min(50, limit))  # Between 1 and 50
  ```
- Validate required parameters and return errors early

### Async Best Practices

- Use `httpx.AsyncClient` for HTTP requests
- Create clients per-request unless performance requires connection pooling
- Set reasonable timeouts (default: 30 seconds)

## Dependencies

MCP dependencies are optional and specified in `pyproject.toml`:

```toml
[project.optional-dependencies]
mcp = [
    "fastmcp>=2.0",
    "httpx>=0.27",
]

[project.scripts]
occhio-mcp = "occhio.mcp.server:main"
```

Add new dependencies here as needed (e.g., `arxiv` for arXiv API access).

## Testing External APIs

### Verify Query Shape First

Before implementing, test API queries directly:

```bash
curl -s -X POST https://api.example.com/graphql \
  -H "Content-Type: application/json" \
  -H "User-Agent: occhio-mcp/1.0" \
  -d '{"query": "{ ... }"}' | python -m json.tool
```

### Mock in Unit Tests

Always mock external calls in tests:

```python
with patch("occhio.mcp.servers.mymodule.graphql_query", new_callable=AsyncMock) as mock:
    mock.return_value = {"data": {...}}
    result = await my_tool(...)
```

## Existing Tools

### Alignment Forum (`af_*`)

- `af_search_posts` - List/search posts (top, new, old, by tag)
- `af_get_post` - Get full post content by ID or slug
- `af_get_comments` - Get comments on a post
- `af_get_user` - Get user profile by slug or ID
- `af_get_tag` - Get tag info and top posts with that tag

All tools use the LessWrong GraphQL API (`https://www.lesswrong.com/graphql`).
