"""Async GraphQL client for the Alignment Forum / LessWrong API."""

import json

import httpx

# Use lesswrong.com as it's more reliable and includes AF posts
AF_GRAPHQL_URL = "https://www.lesswrong.com/graphql"
DEFAULT_TIMEOUT = 30.0
DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "occhio-mcp/1.0",
}


async def graphql_query(query: str, variables: dict | None = None) -> dict:
    """Execute a GraphQL query against the Alignment Forum/LessWrong API.

    Args:
        query: GraphQL query string.
        variables: Optional variables dict.

    Returns:
        The `data` field from the GraphQL response.

    Raises:
        httpx.HTTPStatusError: On non-2xx responses.
        ValueError: If the response contains GraphQL errors.
    """
    payload = {"query": query, "variables": variables or {}}
    async with httpx.AsyncClient(
        timeout=DEFAULT_TIMEOUT, headers=DEFAULT_HEADERS
    ) as client:
        response = await client.post(AF_GRAPHQL_URL, json=payload)
        response.raise_for_status()
        result = response.json()

    if "errors" in result:
        raise ValueError(f"GraphQL errors: {json.dumps(result['errors'], indent=2)}")

    return result.get("data", {})
