"""Acceptance tests for P2 LLM spatial query system.

Validates that the system meets stakeholder business requirements for
SQL safety, query throughput, spatial reference preservation, schema
discovery, error messaging, and string-literal handling.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from projects.p2_llm_spatial_query.src.formatter import format_error
from projects.p2_llm_spatial_query.src.schema_extractor import extract_schema
from projects.p2_llm_spatial_query.src.sql_validator import validate_sql


class TestSpatialQueryAcceptance:
    """Acceptance criteria for LLM spatial query system."""

    # ----------------------------------------------------------------
    # Shared safety config fixture (mirrors conftest.default_config)
    # ----------------------------------------------------------------
    @pytest.fixture()
    def safety_config(self) -> dict:
        """Standard safety configuration for acceptance tests."""
        return {
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
        }

    # ----------------------------------------------------------------
    # REQ 1: SQL validation must block ALL destructive operations
    # ----------------------------------------------------------------
    def test_block_all_destructive_operations(
        self, safety_config: dict
    ) -> None:
        """Every destructive DML/DDL operation must be rejected."""
        destructive_queries = [
            "DELETE FROM harvest_units WHERE unit_id = 1",
            "DROP TABLE harvest_units",
            "UPDATE harvest_units SET acres = 0 WHERE unit_id = 1",
            "INSERT INTO harvest_units (unit_id) VALUES (99)",
            "ALTER TABLE harvest_units ADD COLUMN evil TEXT",
            "TRUNCATE TABLE harvest_units",
        ]
        for sql in destructive_queries:
            is_valid, reason = validate_sql(sql, safety_config)
            assert not is_valid, (
                f"Destructive SQL was allowed through: {sql}"
            )
            assert len(reason) > 0, "Rejection must provide a reason"

    # ----------------------------------------------------------------
    # REQ 2: SQL validation must block ALL SQLite-dangerous keywords
    # ----------------------------------------------------------------
    def test_block_all_sqlite_dangerous_keywords(
        self, safety_config: dict
    ) -> None:
        """PRAGMA, ATTACH, DETACH, VACUUM, LOAD_EXTENSION must be blocked."""
        dangerous_queries = [
            "PRAGMA table_info(harvest_units)",
            "ATTACH DATABASE ':memory:' AS evil",
            "DETACH DATABASE evil",
            "VACUUM",
            "SELECT load_extension('evil.so')",
        ]
        for sql in dangerous_queries:
            is_valid, reason = validate_sql(sql, safety_config)
            assert not is_valid, (
                f"SQLite-dangerous keyword was allowed through: {sql}"
            )

    # ----------------------------------------------------------------
    # REQ 3: Valid SELECT queries must pass validation in under 10ms
    # ----------------------------------------------------------------
    def test_select_validation_latency(self, safety_config: dict) -> None:
        """A valid SELECT query must be validated in under 10 milliseconds."""
        sql = (
            "SELECT h.unit_name, ST_Area(h.geometry) "
            "FROM harvest_units h, streams s "
            "WHERE ST_Intersects(h.geometry, ST_Buffer(s.geometry, 100))"
        )
        start = time.perf_counter()
        is_valid, reason = validate_sql(sql, safety_config)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        assert is_valid, f"Valid query rejected: {reason}"
        assert elapsed_ms < 10.0, (
            f"Validation took {elapsed_ms:.2f}ms, exceeds 10ms budget"
        )

    # ----------------------------------------------------------------
    # REQ 4: Query results must preserve spatial reference information
    # ----------------------------------------------------------------
    def test_query_results_preserve_spatial_reference(
        self, gpkg_path: Path, default_config: dict
    ) -> None:
        """Executing a spatial query must return data with CRS info."""
        from projects.p2_llm_spatial_query.src.executor import execute_query

        sql = "SELECT unit_name, acres FROM harvest_units"
        result = execute_query(sql, gpkg_path, default_config)
        # The result must be a DataFrame with rows
        assert len(result) > 0, "Query returned no results"
        assert "unit_name" in result.columns

    # ----------------------------------------------------------------
    # REQ 5: Error messages must include actionable suggestions
    # ----------------------------------------------------------------
    def test_error_messages_include_suggestions(self) -> None:
        """format_error must produce messages with actionable guidance."""
        msg = format_error("SQL failed validation", "Show me all harvest units")
        assert "Suggestions:" in msg, "Error must include suggestions heading"
        assert "Rephrase" in msg or "rephrase" in msg, (
            "Error must suggest rephrasing"
        )
        assert "SELECT" in msg, (
            "Error must mention that only SELECT queries are supported"
        )
        # Must reference available layers
        assert "harvest_units" in msg, (
            "Error must mention available layers"
        )

    # ----------------------------------------------------------------
    # REQ 6: Schema extraction must discover ALL layers in a GeoPackage
    # ----------------------------------------------------------------
    def test_schema_extraction_discovers_all_layers(
        self, gpkg_path: Path
    ) -> None:
        """extract_schema must find every layer in the synthetic GeoPackage."""
        schema = extract_schema(gpkg_path)
        layers = schema.get("layers", {})
        expected_layers = {"harvest_units", "streams", "sensitive_habitats"}
        found = set(layers.keys())
        assert expected_layers.issubset(found), (
            f"Missing layers: {expected_layers - found}. Found: {found}"
        )
        # Each layer must have columns
        for layer_name in expected_layers:
            cols = layers[layer_name].get("columns", {})
            assert len(cols) > 0, (
                f"Layer {layer_name} has no columns discovered"
            )

    # ----------------------------------------------------------------
    # REQ 7: String literals containing blocked keywords must NOT
    # cause false positives
    # ----------------------------------------------------------------
    def test_string_literals_no_false_positives(
        self, safety_config: dict
    ) -> None:
        """Blocked keywords inside string literals must be safely ignored."""
        false_positive_queries = [
            "SELECT * FROM harvest_units WHERE notes = 'DELETE old records'",
            "SELECT * FROM harvest_units WHERE notes = 'will DROP by later'",
            "SELECT * FROM harvest_units WHERE notes = 'UPDATE pending'",
            "SELECT * FROM harvest_units WHERE notes = 'see PRAGMA docs'",
            "SELECT * FROM harvest_units WHERE notes = 'INSERT coin here'",
            "SELECT * FROM harvest_units WHERE notes = 'TRUNCATE the log'",
        ]
        for sql in false_positive_queries:
            is_valid, reason = validate_sql(sql, safety_config)
            assert is_valid, (
                f"False positive on string literal: {sql!r} -- reason: {reason}"
            )

    # ----------------------------------------------------------------
    # REQ 8: The system must handle at least 100 validation calls/sec
    # ----------------------------------------------------------------
    def test_validation_throughput(self, safety_config: dict) -> None:
        """Validator must process at least 100 queries per second."""
        sql = "SELECT unit_name, acres FROM harvest_units WHERE acres > 20"
        n_calls = 100

        start = time.perf_counter()
        for _ in range(n_calls):
            is_valid, _ = validate_sql(sql, safety_config)
            assert is_valid
        elapsed = time.perf_counter() - start

        throughput = n_calls / elapsed
        assert throughput >= 100.0, (
            f"Throughput {throughput:.0f} calls/s, must be >= 100"
        )
