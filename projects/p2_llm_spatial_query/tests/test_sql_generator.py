"""Tests for the LLM SQL generator module.

Covers SQL extraction from LLM responses (markdown fences, bare text,
semicolons) and the generate_sql() function with mocked HTTP requests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from projects.p2_llm_spatial_query.src.sql_generator import generate_sql, parse_sql_from_response

# -----------------------------------------------------------------------
# parse_sql_from_response
# -----------------------------------------------------------------------

class TestParseSqlFromResponse:
    """Tests for parse_sql_from_response()."""

    def test_from_markdown_fence(self) -> None:
        """SQL wrapped in ```sql ... ``` fences should be extracted."""
        text = "Here is the query:\n```sql\nSELECT * FROM harvest_units\n```"
        result = parse_sql_from_response(text)
        assert result == "SELECT * FROM harvest_units"

    def test_from_bare_text(self) -> None:
        """Bare SQL without any fencing should be returned as-is (stripped)."""
        text = "  SELECT unit_name FROM streams WHERE stream_class = 'I'  "
        result = parse_sql_from_response(text)
        assert result == "SELECT unit_name FROM streams WHERE stream_class = 'I'"

    def test_strips_trailing_semicolon(self) -> None:
        """Trailing semicolons should be removed from extracted SQL."""
        text = "```sql\nSELECT * FROM harvest_units;\n```"
        result = parse_sql_from_response(text)
        assert not result.endswith(";")
        assert result == "SELECT * FROM harvest_units"

    def test_no_language_tag(self) -> None:
        """Code fences without a 'sql' language tag should still be extracted."""
        text = "```\nSELECT COUNT(*) FROM streams\n```"
        result = parse_sql_from_response(text)
        assert result == "SELECT COUNT(*) FROM streams"


# -----------------------------------------------------------------------
# generate_sql (mocked HTTP)
# -----------------------------------------------------------------------

class TestGenerateSql:
    """Tests for generate_sql() with mocked requests."""

    CONFIG = {
        "llm": {
            "endpoint": "http://localhost:8080/v1",
            "model": "test",
            "max_tokens": 100,
            "temperature": 0.0,
        },
    }

    @patch("projects.p2_llm_spatial_query.src.sql_generator.requests.post")
    def test_success(self, mock_post: MagicMock) -> None:
        """A 200 response with valid JSON should return parsed SQL."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "```sql\nSELECT * FROM harvest_units\n```"}}
            ]
        }
        mock_post.return_value = mock_response

        result = generate_sql("system prompt", "user prompt", self.CONFIG)
        assert result == "SELECT * FROM harvest_units"
        mock_post.assert_called_once()

    @patch("projects.p2_llm_spatial_query.src.sql_generator.requests.post")
    def test_connection_error(self, mock_post: MagicMock) -> None:
        """A ConnectionError from requests should raise ConnectionError."""
        mock_post.side_effect = requests.exceptions.ConnectionError("refused")

        with pytest.raises(ConnectionError, match="unreachable"):
            generate_sql("system", "user", self.CONFIG)

    @patch("projects.p2_llm_spatial_query.src.sql_generator.requests.post")
    def test_http_error(self, mock_post: MagicMock) -> None:
        """A 500 HTTP error should raise RuntimeError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error"
        )
        mock_post.return_value = mock_response

        with pytest.raises(RuntimeError, match="LLM endpoint error"):
            generate_sql("system", "user", self.CONFIG)

    @patch("projects.p2_llm_spatial_query.src.sql_generator.requests.post")
    def test_timeout(self, mock_post: MagicMock) -> None:
        """A Timeout exception should raise ConnectionError."""
        mock_post.side_effect = requests.exceptions.Timeout("timed out")

        with pytest.raises(ConnectionError, match="timed out"):
            generate_sql("system", "user", self.CONFIG)

    @patch("projects.p2_llm_spatial_query.src.sql_generator.requests.post")
    def test_bad_response_format(self, mock_post: MagicMock) -> None:
        """A response missing 'choices' should raise RuntimeError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"error": "unexpected format"}
        mock_post.return_value = mock_response

        with pytest.raises(RuntimeError, match="Unexpected LLM response"):
            generate_sql("system", "user", self.CONFIG)
