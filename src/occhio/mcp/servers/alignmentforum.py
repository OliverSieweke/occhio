"""Alignment Forum / LessWrong tools for MCP server."""

import json

from occhio.mcp.graphql_client import graphql_query
from occhio.mcp.instance import mcp

# Tool annotations for all read-only tools
READ_ONLY_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,  # Interacts with external API
}


@mcp.tool(name="af_search_posts", annotations=READ_ONLY_ANNOTATIONS)
async def af_search_posts(
    view: str = "top",
    limit: int = 10,
    tag_id: str | None = None,
) -> str:
    """Search/list posts from the Alignment Forum / LessWrong.

    Args:
        view: Post ordering view - one of "top", "new", "old". Defaults to "top".
        limit: Maximum number of posts to return (1-50). Defaults to 10.
        tag_id: Optional tag ID to filter posts. Use af_get_tag to find tag IDs.

    Returns:
        JSON list of post summaries with id, title, slug, url, date, score, and author.
    """
    limit = max(1, min(50, limit))

    terms = f'view: "{view}", limit: {limit}'
    if tag_id:
        terms += f', tagId: "{tag_id}"'

    query = f"""{{
        posts(input: {{ terms: {{ {terms} }} }}) {{
            results {{
                _id
                title
                slug
                pageUrl
                postedAt
                baseScore
                voteCount
                commentCount
                user {{ username displayName }}
            }}
        }}
    }}"""

    data = await graphql_query(query)
    results = data.get("posts", {}).get("results", [])

    return json.dumps(results, indent=2)


@mcp.tool(name="af_get_post", annotations=READ_ONLY_ANNOTATIONS)
async def af_get_post(
    id: str | None = None,
    slug: str | None = None,
    max_length: int = 10000,
) -> str:
    """Get full content of a single post by ID or slug.

    Args:
        id: Post ID (e.g., "uMQ3cqWDPHhjtiesc"). Either id or slug must be provided.
        slug: Post slug (e.g., "agi-ruin-a-list-of-lethalities"). Either id or slug must be provided.
        max_length: Maximum length of body text to return. Defaults to 10000.

    Returns:
        JSON object with full post details including body text.
    """
    if not id and not slug:
        return json.dumps({"error": "Either 'id' or 'slug' must be provided"})

    selector = f'_id: "{id}"' if id else f'slug: "{slug}"'

    query = f"""{{
        post(input: {{ selector: {{ {selector} }} }}) {{
            result {{
                _id
                title
                slug
                pageUrl
                postedAt
                modifiedAt
                baseScore
                voteCount
                commentCount
                contents {{ markdown }}
                user {{ username displayName }}
                tags {{ name slug }}
            }}
        }}
    }}"""

    data = await graphql_query(query)
    post = data.get("post", {}).get("result")

    if not post:
        return json.dumps(
            {"error": f"Post not found with {'id' if id else 'slug'}: {id or slug}"}
        )

    # Extract markdown and truncate
    if post.get("contents"):
        markdown = post["contents"].get("markdown", "")
        if len(markdown) > max_length:
            markdown = markdown[:max_length] + "\n\n... [truncated]"
        post["body"] = markdown
        del post["contents"]

    return json.dumps(post, indent=2)


@mcp.tool(name="af_get_comments", annotations=READ_ONLY_ANNOTATIONS)
async def af_get_comments(
    post_id: str,
    limit: int = 20,
    view: str = "postCommentsTop",
) -> str:
    """Get comments on a post.

    Args:
        post_id: The ID of the post to get comments for.
        limit: Maximum number of comments to return (1-100). Defaults to 20.
        view: Comment ordering - "postCommentsTop" (by score) or "postCommentsNew". Defaults to "postCommentsTop".

    Returns:
        JSON list of comments with id, date, score, author, body, and parent comment id.
    """
    limit = max(1, min(100, limit))

    query = f"""{{
        comments(input: {{ terms: {{ postId: "{post_id}", view: "{view}", limit: {limit} }} }}) {{
            results {{
                _id
                postedAt
                baseScore
                voteCount
                contents {{ markdown }}
                user {{ username displayName }}
                parentCommentId
            }}
        }}
    }}"""

    data = await graphql_query(query)
    comments = data.get("comments", {}).get("results", [])

    # Extract markdown from comment bodies
    for comment in comments:
        if comment.get("contents"):
            comment["body"] = comment["contents"].get("markdown", "")
            del comment["contents"]

    return json.dumps(comments, indent=2)


@mcp.tool(name="af_get_user", annotations=READ_ONLY_ANNOTATIONS)
async def af_get_user(
    slug: str | None = None,
    id: str | None = None,
) -> str:
    """Get user profile information.

    Args:
        slug: User slug (lowercase username, e.g., "eliezer_yudkowsky"). Either slug or id must be provided.
        id: User ID. Either slug or id must be provided.

    Returns:
        JSON object with user profile including karma, post count, and comment count.
    """
    if not slug and not id:
        return json.dumps({"error": "Either 'slug' or 'id' must be provided"})

    # Slugs must be lowercase
    if slug:
        slug = slug.lower()
        selector = f'slug: "{slug}"'
    else:
        selector = f'_id: "{id}"'

    query = f"""{{
        user(input: {{ selector: {{ {selector} }} }}) {{
            result {{
                _id
                username
                displayName
                slug
                karma
                postCount
                commentCount
                createdAt
            }}
        }}
    }}"""

    data = await graphql_query(query)
    user = data.get("user", {}).get("result")

    if not user:
        return json.dumps(
            {"error": f"User not found with {'slug' if slug else 'id'}: {slug or id}"}
        )

    return json.dumps(user, indent=2)


@mcp.tool(name="af_get_tag", annotations=READ_ONLY_ANNOTATIONS)
async def af_get_tag(
    id: str | None = None,
    name: str | None = None,
    limit: int = 10,
) -> str:
    """Get tag/concept information and top posts with that tag.

    Note: Tag lookup by name uses a search which may return multiple results.
    For precise lookup, use the tag ID.

    Args:
        id: Tag ID for exact lookup. Either id or name must be provided.
        name: Tag name to search for. Either id or name must be provided.
        limit: Maximum number of posts to return with the tag (1-50). Defaults to 10.

    Returns:
        JSON object with tag info and list of top posts with that tag.
    """
    if not id and not name:
        return json.dumps({"error": "Either 'id' or 'name' must be provided"})

    limit = max(1, min(50, limit))

    # If searching by name, first find matching tags
    if name and not id:
        search_query = """{
            tags(input: { terms: { limit: 10 } }) {
                results {
                    _id
                    name
                    slug
                    postCount
                }
            }
        }"""
        search_data = await graphql_query(search_query)
        tags = search_data.get("tags", {}).get("results", [])

        # Find matching tag (case-insensitive)
        name_lower = name.lower()
        matching_tag = None
        for tag in tags:
            if (
                tag.get("name", "").lower() == name_lower
                or tag.get("slug", "").lower() == name_lower
            ):
                matching_tag = tag
                break

        if not matching_tag:
            # Return search results as suggestions
            return json.dumps(
                {
                    "error": f"Tag '{name}' not found in top tags. Try using an exact tag ID.",
                    "suggestions": [
                        {"name": t["name"], "slug": t["slug"], "id": t["_id"]}
                        for t in tags[:5]
                    ],
                }
            )

        id = matching_tag["_id"]

    # Get tag details
    tag_query = f"""{{
        tag(input: {{ selector: {{ _id: "{id}" }} }}) {{
            result {{
                _id
                name
                slug
                postCount
            }}
        }}
    }}"""

    tag_data = await graphql_query(tag_query)
    tag = tag_data.get("tag", {}).get("result")

    if not tag:
        return json.dumps({"error": f"Tag not found with id: {id}"})

    # Get posts with this tag
    posts_query = f"""{{
        posts(input: {{ terms: {{ tagId: "{id}", view: "top", limit: {limit} }} }}) {{
            results {{
                _id
                title
                slug
                pageUrl
                baseScore
                postedAt
                user {{ username displayName }}
            }}
        }}
    }}"""

    posts_data = await graphql_query(posts_query)
    posts = posts_data.get("posts", {}).get("results", [])

    return json.dumps(
        {
            "tag": tag,
            "posts": posts,
        },
        indent=2,
    )
