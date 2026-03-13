"""Tests for the GraphQL client."""

from occhio.mcp.graphql_client import strip_html


class TestStripHtml:
    """Tests for strip_html function."""

    def test_strips_basic_tags(self):
        html = "<p>Hello <strong>world</strong></p>"
        assert strip_html(html) == "Hello world"

    def test_handles_empty_string(self):
        assert strip_html("") == ""

    def test_handles_none(self):
        assert strip_html(None) == ""

    def test_normalizes_whitespace(self):
        html = "<p>Hello</p>   <p>World</p>"
        assert strip_html(html) == "Hello World"

    def test_handles_nested_tags(self):
        html = "<div><p><span>Nested</span> content</p></div>"
        assert strip_html(html) == "Nested content"

    def test_handles_attributes(self):
        html = '<a href="https://example.com" class="link">Link text</a>'
        assert strip_html(html) == "Link text"


# Integration tests for graphql_query would require mocking httpx
# or actually hitting the API, which is not suitable for unit tests.
# Consider using pytest-httpx for mocked tests in a separate file.
