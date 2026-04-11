"""Smoke tests for P2 LLM Spatial Query.

Quick sanity checks that core modules import, key functions are callable,
and basic operations produce expected types. Designed to run in < 1 second.
"""

from __future__ import annotations

from pathlib import Path


class TestP2Smoke:
    """Fast smoke tests for P2 spatial query pipeline."""

    def test_imports(self):
        """All core P2 modules should import without error."""
        from projects.p2_llm_spatial_query.src import (
            executor,  # noqa: F401
            formatter,  # noqa: F401
            prompt_builder,  # noqa: F401
            schema_extractor,  # noqa: F401
            sql_generator,  # noqa: F401
            sql_validator,  # noqa: F401
        )

    def test_sanitize_sql_runs(self):
        """sanitize_sql should strip comments and normalize whitespace."""
        from projects.p2_llm_spatial_query.src.sql_validator import sanitize_sql

        result = sanitize_sql("SELECT  *  FROM  t  -- comment")
        assert result == "SELECT * FROM t"

    def test_validate_sql_accepts_select(self):
        """A simple SELECT should pass validation."""
        from projects.p2_llm_spatial_query.src.sql_validator import validate_sql

        config = {
            "safety": {
                "blocked_operations": ["DELETE", "DROP"],
                "allowed_operations": ["SELECT"],
            }
        }
        is_valid, reason = validate_sql("SELECT * FROM t", config)
        assert is_valid, reason

    def test_validate_sql_blocks_delete(self):
        """DELETE should be blocked."""
        from projects.p2_llm_spatial_query.src.sql_validator import validate_sql

        config = {
            "safety": {
                "blocked_operations": ["DELETE"],
                "allowed_operations": ["SELECT"],
            }
        }
        is_valid, _ = validate_sql("DELETE FROM t", config)
        assert not is_valid

    def test_parse_sql_from_response_runs(self):
        """parse_sql_from_response should extract SQL from markdown."""
        from projects.p2_llm_spatial_query.src.sql_generator import (
            parse_sql_from_response,
        )

        result = parse_sql_from_response("```sql\nSELECT 1\n```")
        assert result == "SELECT 1"

    def test_format_error_runs(self):
        """format_error should produce a string with suggestions."""
        from projects.p2_llm_spatial_query.src.formatter import format_error

        result = format_error("test error", "test query")
        assert "Error" in result
        assert "Suggestions" in result

    def test_format_results_empty(self):
        """format_results should handle an empty DataFrame."""
        import pandas as pd

        from projects.p2_llm_spatial_query.src.formatter import format_results

        result = format_results(pd.DataFrame(), "test query")
        assert "No results found" in result

    def test_schema_extraction_runs(self, gpkg_path: Path, default_config: dict):
        """extract_schema should return a dict with layers."""
        from projects.p2_llm_spatial_query.src.schema_extractor import extract_schema

        schema = extract_schema(gpkg_path)
        assert "layers" in schema
        assert len(schema["layers"]) > 0
