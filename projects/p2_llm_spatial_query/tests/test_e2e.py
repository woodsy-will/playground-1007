"""End-to-end tests for the LLM spatial query pipeline.

User story: A forest manager asks a natural language question about their
GeoPackage data, gets SQL generated, validated, executed, and results
formatted.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from projects.p2_llm_spatial_query.src.pipeline import run_query


def _write_e2e_config(tmp_path: Path, gpkg_path: Path) -> Path:
    """Write a YAML config for the query pipeline."""
    # Create a minimal few-shot file
    few_shot_path = tmp_path / "few_shot_queries.yaml"
    few_shot_path.write_text(
        "examples:\n"
        "  - question: 'Show all harvest units'\n"
        "    sql: 'SELECT * FROM harvest_units'\n"
    )

    # Create a minimal schema metadata file
    schema_path = tmp_path / "schema_metadata.yaml"
    schema_content = {
        "layers": {
            "harvest_units": {
                "description": "Proposed timber harvest units",
                "columns": {
                    "unit_id": {"type": "integer", "description": "Unique unit ID"},
                    "unit_name": {"type": "text", "description": "Unit name"},
                    "acres": {"type": "real", "description": "Area in acres"},
                    "prescription": {"type": "text", "description": "Prescription"},
                    "geometry": {"type": "polygon", "srid": 3310},
                },
            },
            "streams": {
                "description": "Classified watercourses",
                "columns": {
                    "stream_id": {"type": "integer", "description": "Stream ID"},
                    "stream_class": {"type": "text", "description": "Class"},
                    "geometry": {"type": "linestring", "srid": 3310},
                },
            },
        },
    }
    with open(schema_path, "w") as f:
        yaml.dump(schema_content, f)

    config = {
        "llm": {
            "model": "test-model",
            "endpoint": "http://localhost:8080/v1",
            "max_tokens": 512,
            "temperature": 0.1,
        },
        "geopackage": {
            "path": str(gpkg_path),
            "layers": ["harvest_units", "streams", "sensitive_habitats"],
        },
        "safety": {
            "allowed_operations": [
                "SELECT",
                "ST_Buffer",
                "ST_Intersects",
                "ST_Within",
                "ST_Contains",
                "ST_Area",
                "ST_Length",
            ],
            "blocked_operations": [
                "DELETE",
                "DROP",
                "UPDATE",
                "INSERT",
                "ALTER",
                "TRUNCATE",
            ],
        },
        "rag": {
            "few_shot_examples": str(few_shot_path),
            "schema_metadata": str(schema_path),
            "chunk_size": 512,
            "top_k": 5,
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


class TestSpatialQueryE2E:
    """E2E tests simulating a forest manager querying spatial data."""

    # ------------------------------------------------------------------ #
    # 1. Simple question -- SELECT * FROM harvest_units
    # ------------------------------------------------------------------ #
    def test_user_asks_simple_question(
        self, tmp_path: Path, gpkg_path: Path
    ) -> None:
        config_path = _write_e2e_config(tmp_path, gpkg_path)

        with patch(
            "projects.p2_llm_spatial_query.src.sql_generator.generate_sql",
            return_value="SELECT * FROM harvest_units",
        ):
            result = run_query("Show me all harvest units", str(config_path))

        assert result["is_valid"] is True
        assert isinstance(result["raw_results"], pd.DataFrame)
        assert len(result["raw_results"]) > 0
        assert isinstance(result["results_summary"], str)
        assert len(result["results_summary"]) > 0

    # ------------------------------------------------------------------ #
    # 2. Spatial question -- ST_Area
    # ------------------------------------------------------------------ #
    def test_user_asks_spatial_question(
        self, tmp_path: Path, gpkg_path: Path
    ) -> None:
        config_path = _write_e2e_config(tmp_path, gpkg_path)
        sql = "SELECT unit_name, acres FROM harvest_units WHERE acres > 20"

        with patch(
            "projects.p2_llm_spatial_query.src.sql_generator.generate_sql",
            return_value=sql,
        ):
            result = run_query(
                "Which harvest units are larger than 20 acres?",
                str(config_path),
            )

        assert result["is_valid"] is True
        df = result["raw_results"]
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "acres" in df.columns

    # ------------------------------------------------------------------ #
    # 3. Aggregate question -- COUNT(*)
    # ------------------------------------------------------------------ #
    def test_user_asks_aggregate_question(
        self, tmp_path: Path, gpkg_path: Path
    ) -> None:
        config_path = _write_e2e_config(tmp_path, gpkg_path)
        sql = "SELECT COUNT(*) AS cnt FROM harvest_units"

        with patch(
            "projects.p2_llm_spatial_query.src.sql_generator.generate_sql",
            return_value=sql,
        ):
            result = run_query(
                "How many harvest units are there?",
                str(config_path),
            )

        assert result["is_valid"] is True
        df = result["raw_results"]
        assert len(df) == 1
        assert df["cnt"].iloc[0] > 0

    # ------------------------------------------------------------------ #
    # 4. Bad SQL -- DROP TABLE is blocked
    # ------------------------------------------------------------------ #
    def test_user_gets_helpful_error_on_bad_sql(
        self, tmp_path: Path, gpkg_path: Path
    ) -> None:
        config_path = _write_e2e_config(tmp_path, gpkg_path)

        with patch(
            "projects.p2_llm_spatial_query.src.sql_generator.generate_sql",
            return_value="DROP TABLE harvest_units",
        ):
            result = run_query(
                "Delete all harvest units",
                str(config_path),
            )

        assert result["is_valid"] is False
        assert "Error" in result["results_summary"]
        assert "Suggestions" in result["results_summary"]

    # ------------------------------------------------------------------ #
    # 5. LLM down -- ConnectionError produces graceful message
    # ------------------------------------------------------------------ #
    def test_user_gets_error_on_llm_down(
        self, tmp_path: Path, gpkg_path: Path
    ) -> None:
        config_path = _write_e2e_config(tmp_path, gpkg_path)

        with patch(
            "projects.p2_llm_spatial_query.src.sql_generator.generate_sql",
            side_effect=ConnectionError("LLM endpoint unreachable"),
        ):
            result = run_query(
                "Show me all harvest units",
                str(config_path),
            )

        assert result["is_valid"] is False
        assert "LLM" in result["results_summary"]

    # ------------------------------------------------------------------ #
    # 6. Result metadata -- all expected keys present
    # ------------------------------------------------------------------ #
    def test_user_result_includes_metadata(
        self, tmp_path: Path, gpkg_path: Path
    ) -> None:
        config_path = _write_e2e_config(tmp_path, gpkg_path)

        with patch(
            "projects.p2_llm_spatial_query.src.sql_generator.generate_sql",
            return_value="SELECT * FROM harvest_units",
        ):
            result = run_query("Show me all harvest units", str(config_path))

        expected_keys = {"sql", "is_valid", "validation_reason", "results_summary", "raw_results"}
        assert expected_keys.issubset(set(result.keys()))
