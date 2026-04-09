"""Integration tests chaining P2 LLM spatial query modules.

Tests the end-to-end flow: schema extraction -> prompt building ->
SQL validation -> query execution -> result formatting.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

from projects.p2_llm_spatial_query.src.executor import execute_query
from projects.p2_llm_spatial_query.src.formatter import format_results
from projects.p2_llm_spatial_query.src.pipeline import run_query
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


# -----------------------------------------------------------------------
# Full pipeline with mocked LLM
# -----------------------------------------------------------------------

class TestRunQueryMocked:
    """Verify run_query end-to-end with the LLM layer mocked out."""

    def test_run_query_returns_valid_result(
        self,
        gpkg_path: Path,
        default_config: dict,
        tmp_path: Path,
    ) -> None:
        """run_query should return a dict with expected keys when the LLM
        generate_sql call is mocked to return a valid SELECT statement."""
        # Build a temporary YAML config file with all required keys
        config = {
            "safety": default_config["safety"],
            "llm": default_config["llm"],
            "geopackage": {"path": str(gpkg_path)},
            "rag": {
                "few_shot_examples": "nonexistent.yaml",
                "schema_metadata": "nonexistent.yaml",
                "top_k": 5,
            },
            "schema": default_config.get("schema", {}),
        }
        config_path = tmp_path / "test_pipeline_config.yaml"
        config_path.write_text(yaml.dump(config))

        # Minimal schema metadata to satisfy prompt building
        mock_schema = {
            "layers": {
                "harvest_units": {
                    "description": "Proposed timber harvest units",
                    "columns": {
                        "unit_id": {"type": "integer", "description": "ID"},
                        "geometry": {"type": "polygon", "srid": 3310},
                    },
                },
            },
        }

        with (
            patch(
                "projects.p2_llm_spatial_query.src.sql_generator.generate_sql",
                return_value="SELECT * FROM harvest_units",
            ),
            patch(
                "projects.p2_llm_spatial_query.src.schema_extractor.load_schema_metadata",
                return_value=mock_schema,
            ),
            patch(
                "projects.p2_llm_spatial_query.src.prompt_builder.load_few_shot_examples",
                return_value=[],
            ),
        ):
            result = run_query("Show me all harvest units", str(config_path))

        # Verify result structure
        assert isinstance(result, dict)
        for key in ("sql", "is_valid", "results_summary", "raw_results"):
            assert key in result, f"Missing key: {key}"

        # The mocked SQL is a valid SELECT, so validation should pass
        assert result["is_valid"] is True
        assert result["raw_results"] is not None
        assert len(result["raw_results"]) > 0
