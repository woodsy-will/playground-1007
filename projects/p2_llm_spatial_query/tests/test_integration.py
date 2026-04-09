"""Integration tests chaining P2 LLM spatial query modules.

Tests the end-to-end flow: schema extraction -> prompt building ->
SQL validation -> query execution -> result formatting.
"""

from __future__ import annotations

from pathlib import Path

from projects.p2_llm_spatial_query.src.executor import execute_query
from projects.p2_llm_spatial_query.src.formatter import format_results
from projects.p2_llm_spatial_query.src.prompt_builder import (
    build_system_prompt,
    build_user_prompt,
)
from projects.p2_llm_spatial_query.src.schema_extractor import extract_schema
from projects.p2_llm_spatial_query.src.sql_validator import validate_sql

# -----------------------------------------------------------------------
# Schema -> Prompt pipeline
# -----------------------------------------------------------------------

class TestSchemaToPrompt:
    """Verify that schema extraction feeds correctly into prompt building."""

    def test_schema_extract_to_prompt_build(
        self,
        gpkg_path: Path,
        default_config: dict,
    ) -> None:
        """extract_schema -> build_system_prompt -> build_user_prompt produces
        a complete prompt containing schema information, safety constraints,
        and the user query."""
        # Step 1: Extract schema from the synthetic GeoPackage
        schema = extract_schema(gpkg_path)
        assert "layers" in schema
        assert len(schema["layers"]) > 0

        # Step 2: Build the system prompt using extracted schema
        system_prompt = build_system_prompt(schema, default_config)

        # System prompt should contain schema layer names
        for layer_name in schema["layers"]:
            assert layer_name in system_prompt

        # System prompt should contain safety constraints
        assert "SELECT" in system_prompt
        assert "NEVER" in system_prompt or "never" in system_prompt.lower()

        # System prompt should mention allowed spatial functions
        for op in default_config["safety"]["allowed_operations"]:
            if op.startswith("ST_"):
                assert op in system_prompt

        # System prompt should mention blocked operations
        for op in default_config["safety"]["blocked_operations"]:
            assert op in system_prompt

        # Step 3: Build the user prompt
        few_shots = [
            {
                "question": "How many harvest units are there?",
                "sql": "SELECT COUNT(*) FROM harvest_units",
            },
        ]
        user_query = "Show all streams near harvest units"
        user_prompt = build_user_prompt(user_query, few_shots, default_config)

        # User prompt should contain the user query
        assert user_query in user_prompt

        # User prompt should contain the few-shot example
        assert "How many harvest units" in user_prompt
        assert "SELECT COUNT(*)" in user_prompt


# -----------------------------------------------------------------------
# Validate -> Execute -> Format pipeline
# -----------------------------------------------------------------------

class TestValidateAndExecute:
    """Verify the validate -> execute -> format chain with real data."""

    def test_validate_execute_format_chain(
        self,
        gpkg_path: Path,
        default_config: dict,
    ) -> None:
        """A valid SELECT query should pass validation, execute against the
        synthetic GeoPackage, and produce a formatted summary string."""
        sql = "SELECT * FROM harvest_units"

        # Step 1: Validate the SQL
        is_valid, reason = validate_sql(sql, default_config)
        assert is_valid, f"Validation failed: {reason}"

        # Step 2: Execute the query
        results = execute_query(sql, gpkg_path, default_config)
        assert len(results) > 0, "Expected rows from synthetic harvest_units"

        # Step 3: Format the results
        user_query = "Show me all harvest units"
        output = format_results(results, user_query)

        # Output should include the query and row count
        assert "Query: Show me all harvest units" in output
        assert f"Rows returned: {len(results)}" in output

        # Output should include column names
        assert "Columns:" in output
