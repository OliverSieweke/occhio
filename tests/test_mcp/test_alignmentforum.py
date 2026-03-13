"""Tests for Alignment Forum tools."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from occhio.mcp.servers.alignmentforum import (
    af_get_comments,
    af_get_post,
    af_get_tag,
    af_get_user,
    af_search_posts,
)


class TestAfSearchPosts:
    """Tests for af_search_posts tool."""

    @pytest.mark.asyncio
    async def test_returns_posts(self):
        mock_data = {
            "posts": {
                "results": [
                    {
                        "_id": "test123",
                        "title": "Test Post",
                        "slug": "test-post",
                        "pageUrl": "https://example.com/posts/test123",
                        "postedAt": "2024-01-01T00:00:00Z",
                        "baseScore": 100,
                        "voteCount": 50,
                        "commentCount": 10,
                        "user": {"username": "testuser", "displayName": "Test User"},
                    }
                ]
            }
        }

        with patch(
            "occhio.mcp.servers.alignmentforum.graphql_query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = mock_data
            result = await af_search_posts(limit=1)

        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["_id"] == "test123"
        assert parsed[0]["title"] == "Test Post"

    @pytest.mark.asyncio
    async def test_limits_clamped(self):
        with patch(
            "occhio.mcp.servers.alignmentforum.graphql_query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = {"posts": {"results": []}}

            # Test limit too high
            await af_search_posts(limit=100)
            query = mock_query.call_args[0][0]
            assert "limit: 50" in query

            # Test limit too low
            await af_search_posts(limit=0)
            query = mock_query.call_args[0][0]
            assert "limit: 1" in query


class TestAfGetPost:
    """Tests for af_get_post tool."""

    @pytest.mark.asyncio
    async def test_requires_id_or_slug(self):
        result = await af_get_post()
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_returns_post_with_stripped_body(self):
        mock_data = {
            "post": {
                "result": {
                    "_id": "test123",
                    "title": "Test Post",
                    "slug": "test-post",
                    "pageUrl": "https://example.com",
                    "postedAt": "2024-01-01T00:00:00Z",
                    "modifiedAt": "2024-01-02T00:00:00Z",
                    "baseScore": 100,
                    "voteCount": 50,
                    "commentCount": 10,
                    "htmlBody": "<p>Hello <strong>world</strong></p>",
                    "user": {"username": "testuser", "displayName": "Test User"},
                    "tags": [{"name": "AI", "slug": "ai"}],
                }
            }
        }

        with patch(
            "occhio.mcp.servers.alignmentforum.graphql_query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = mock_data
            result = await af_get_post(id="test123")

        parsed = json.loads(result)
        assert parsed["_id"] == "test123"
        assert "bodyText" in parsed
        assert "htmlBody" not in parsed
        assert parsed["bodyText"] == "Hello world"

    @pytest.mark.asyncio
    async def test_truncates_long_body(self):
        long_text = "x" * 20000
        mock_data = {
            "post": {
                "result": {
                    "_id": "test123",
                    "title": "Test",
                    "slug": "test",
                    "pageUrl": "https://example.com",
                    "postedAt": "2024-01-01T00:00:00Z",
                    "modifiedAt": None,
                    "baseScore": 0,
                    "voteCount": 0,
                    "commentCount": 0,
                    "htmlBody": f"<p>{long_text}</p>",
                    "user": None,
                    "tags": [],
                }
            }
        }

        with patch(
            "occhio.mcp.servers.alignmentforum.graphql_query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = mock_data
            result = await af_get_post(id="test123", max_length=100)

        parsed = json.loads(result)
        assert len(parsed["bodyText"]) <= 120  # 100 + "... [truncated]"
        assert "[truncated]" in parsed["bodyText"]


class TestAfGetComments:
    """Tests for af_get_comments tool."""

    @pytest.mark.asyncio
    async def test_strips_html_from_comments(self):
        mock_data = {
            "comments": {
                "results": [
                    {
                        "_id": "comment1",
                        "postedAt": "2024-01-01T00:00:00Z",
                        "baseScore": 10,
                        "voteCount": 5,
                        "htmlBody": "<p>Comment <em>text</em></p>",
                        "user": {"username": "user1", "displayName": "User One"},
                        "parentCommentId": None,
                    }
                ]
            }
        }

        with patch(
            "occhio.mcp.servers.alignmentforum.graphql_query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = mock_data
            result = await af_get_comments(post_id="test123")

        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["bodyText"] == "Comment text"
        assert "htmlBody" not in parsed[0]


class TestAfGetUser:
    """Tests for af_get_user tool."""

    @pytest.mark.asyncio
    async def test_requires_slug_or_id(self):
        result = await af_get_user()
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_lowercases_slug(self):
        with patch(
            "occhio.mcp.servers.alignmentforum.graphql_query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = {"user": {"result": None}}
            await af_get_user(slug="TestUser")
            query = mock_query.call_args[0][0]
            assert 'slug: "testuser"' in query


class TestAfGetTag:
    """Tests for af_get_tag tool."""

    @pytest.mark.asyncio
    async def test_requires_id_or_name(self):
        result = await af_get_tag()
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_returns_tag_and_posts(self):
        tag_data = {
            "tag": {
                "result": {
                    "_id": "tag123",
                    "name": "AI Safety",
                    "slug": "ai-safety",
                    "postCount": 100,
                }
            }
        }
        posts_data = {
            "posts": {
                "results": [
                    {
                        "_id": "post1",
                        "title": "Post 1",
                        "slug": "post-1",
                        "pageUrl": "",
                        "baseScore": 50,
                        "postedAt": "",
                        "user": None,
                    }
                ]
            }
        }

        call_count = [0]

        async def mock_query(query):
            call_count[0] += 1
            if "tag(" in query:
                return tag_data
            return posts_data

        with patch(
            "occhio.mcp.servers.alignmentforum.graphql_query", side_effect=mock_query
        ):
            result = await af_get_tag(id="tag123")

        parsed = json.loads(result)
        assert "tag" in parsed
        assert "posts" in parsed
        assert parsed["tag"]["name"] == "AI Safety"
        assert len(parsed["posts"]) == 1
