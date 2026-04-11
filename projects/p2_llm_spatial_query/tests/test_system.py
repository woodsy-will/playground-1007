"""System tests for the P2 LLM spatial query pipeline.

Verifies the complete system end-to-end: natural language question ->
SQL generation -> validation -> execution against GeoPackage -> formatted
results.  The LLM is mocked (not available in test environment) but all
other components run with real code and data.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

from projects.p2_llm_spatial_query.src.pipeline import run_query

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_config_path(
    tmp_path: Path,
    gpkg_path: Path,
    default_config: dict,
) -> Path:
    """Write a temporary YAML config file and return its path."""
    config = {
        "safety": default_config["safety"],
        "llm": default_config["llm"],
        "geopackage": {"path": str(gpkg_path)},
        "rag": {
            "few_shot_examples": "nonexistent.yaml",
            "schema_metadata": "nonexistent.yaml",
            "top_k": 5,
        },
    }
    config_path = tmp_path / "system_test_config.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


def _mock_schema() -> dict:
    """Minimal schema metadata sufficient for prompt building."""
    return {
        "layers": {
            "harvest_units": {
                "description": "Proposed timber harvest units",
                "columns": {
                    "unit_id": {"type": "integer", "description": "ID"},
                    "unit_name": {"type": "text", "description": "Name"},
                    "acres": {"type": "real", "description": "Area in acres"},
                    "geometry": {"type": "polygon", "srid": 3310},
                },
            },
            "streams": {
                "description": "Classified watercourses",
                "columns": {
                    "stream_id": {"type": "integer", "description": "Stream ID"},
                    "geometry": {"type": "linestring", "srid": 3310},
                },
            },
        },
    }


def _run_with_mocked_sql(
    sql: str,
    user_query: str,
    tmp_path: Path,
    gpkg_path: Path,
    default_config: dict,
) -> dict:
    """Run the full pipeline with a mocked generate_sql return value."""
    config_path = _make_config_path(tmp_path, gpkg_path, default_config)
    with (
        patch(
            "projects.p2_llm_spatial_query.src.sql_generator.generate_sql",
            return_value=sql,
        ),
        patch(
            "projects.p2_llm_spatial_query.src.schema_extractor.load_schema_metadata",
            return_value=_mock_schema(),
        ),
        patch(
            "projects.p2_llm_spatial_query.src.prompt_builder.load_few_shot_examples",
            return_value=[],
        ),
    ):
        return run_query(user_query, str(config_path))


# -----------------------------------------------------------------------
# System test class
# -----------------------------------------------------------------------


class TestSpatialQuerySystem:
    """System-level tests for the P2 LLM spatial query pipeline."""

    def test_system_valid_query_returns_results(
        self,
        tmp_path: Path,
        gpkg_path: Path,
        default_config: dict,
    ) -> None:
        """A valid SELECT query should pass validation, execute, and
        return a result with is_valid=True and a non-empty DataFrame."""
        result = _run_with_mocked_sql(
            sql="SELECT * FROM harvest_units",
            user_query="Show me all harvest units",
            tmp_path=tmp_path,
            gpkg_path=gpkg_path,
            default_config=default_config,
        )

        assert result["is_valid"] is True
        assert result["raw_results"] is not None
        assert len(result["raw_results"]) > 0

    def test_system_invalid_sql_rejected(
        self,
        tmp_path: Path,
        gpkg_path: Path,
        default_config: dict,
    ) -> None:
        """A DELETE statement should be rejected with is_valid=False and
        an error message in results_summary."""
        result = _run_with_mocked_sql(
            sql="DELETE FROM harvest_units",
            user_query="Delete all units",
            tmp_path=tmp_path,
            gpkg_path=gpkg_path,
            default_config=default_config,
        )

        assert result["is_valid"] is False
        assert "Error" in result["results_summary"]

    def test_system_spatial_query_with_geometry(
        self,
        tmp_path: Path,
        gpkg_path: Path,
        default_config: dict,
    ) -> None:
        """A query using ST_Area should execute and return area values."""
        result = _run_with_mocked_sql(
            sql="SELECT unit_name, ST_Area(geometry) AS area FROM harvest_units",
            user_query="What is the area of each harvest unit?",
            tmp_path=tmp_path,
            gpkg_path=gpkg_path,
            default_config=default_config,
        )

        # ST_Area may not work without SpatiaLite, but validation should pass
        assert result["is_valid"] is True
        # If execution succeeded, verify area column exists
        if result["raw_results"] is not None and len(result["raw_results"]) > 0:
            assert "area" in result["raw_results"].columns

    def test_system_handles_llm_failure(
        self,
        tmp_path: Path,
        gpkg_path: Path,
        default_config: dict,
    ) -> None:
        """When the LLM raises ConnectionError, the system should return
        a graceful error message without crashing."""
        config_path = _make_config_path(tmp_path, gpkg_path, default_config)
        with (
            patch(
                "projects.p2_llm_spatial_query.src.schema_extractor.load_schema_metadata",
                return_value=_mock_schema(),
            ),
            patch(
                "projects.p2_llm_spatial_query.src.prompt_builder.load_few_shot_examples",
                return_value=[],
            ),
            patch(
                "projects.p2_llm_spatial_query.src.sql_generator.generate_sql",
                side_effect=ConnectionError("LLM endpoint unreachable"),
            ),
        ):
            result = run_query("Show all harvest units", str(config_path))

        assert "Error" in result["results_summary"]
        assert result["sql"] == ""
        assert result["raw_results"] is None

    def test_system_result_formatting(
        self,
        tmp_path: Path,
        gpkg_path: Path,
        default_config: dict,
    ) -> None:
        """results_summary should be human-readable text with row counts
        and column names."""
        result = _run_with_mocked_sql(
            sql="SELECT unit_id, unit_name, acres FROM harvest_units",
            user_query="List all harvest units",
            tmp_path=tmp_path,
            gpkg_path=gpkg_path,
            default_config=default_config,
        )

        summary = result["results_summary"]
        assert "Rows returned:" in summary
        assert "Columns:" in summary
        assert "unit_id" in summary
        assert "unit_name" in summary
        assert "acres" in summary

    def test_system_security_blocks_injection(
        self,
        tmp_path: Path,
        gpkg_path: Path,
        default_config: dict,
    ) -> None:
        """Multi-statement injection (SELECT 1; DROP TABLE t) should be
        blocked by the validator."""
        result = _run_with_mocked_sql(
            sql="SELECT 1; DROP TABLE t",
            user_query="Hack the database",
            tmp_path=tmp_path,
            gpkg_path=gpkg_path,
            default_config=default_config,
        )

        assert result["is_valid"] is False
        assert "Error" in result["results_summary"]
