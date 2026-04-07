"""Tests for schema extraction from GeoPackage files."""

from __future__ import annotations

from pathlib import Path

import pytest

from projects.p2_llm_spatial_query.src.schema_extractor import (
    extract_schema,
    load_schema_metadata,
)


class TestExtractSchema:
    """Introspect a synthetic GeoPackage and verify structure."""

    def test_returns_layers_key(self, gpkg_path: Path) -> None:
        schema = extract_schema(gpkg_path)
        assert "layers" in schema

    def test_discovers_harvest_units(self, gpkg_path: Path) -> None:
        schema = extract_schema(gpkg_path)
        assert "harvest_units" in schema["layers"]

    def test_discovers_streams(self, gpkg_path: Path) -> None:
        schema = extract_schema(gpkg_path)
        assert "streams" in schema["layers"]

    def test_discovers_sensitive_habitats(self, gpkg_path: Path) -> None:
        schema = extract_schema(gpkg_path)
        assert "sensitive_habitats" in schema["layers"]

    def test_harvest_units_has_columns(self, gpkg_path: Path) -> None:
        schema = extract_schema(gpkg_path)
        cols = schema["layers"]["harvest_units"]["columns"]
        assert "unit_id" in cols
        assert "unit_name" in cols
        assert "acres" in cols
        assert "prescription" in cols

    def test_streams_has_columns(self, gpkg_path: Path) -> None:
        schema = extract_schema(gpkg_path)
        cols = schema["layers"]["streams"]["columns"]
        assert "stream_id" in cols
        assert "stream_class" in cols

    def test_geometry_column_detected(self, gpkg_path: Path) -> None:
        schema = extract_schema(gpkg_path)
        hu_cols = schema["layers"]["harvest_units"]["columns"]
        # The geometry column should exist with a type and srid
        geom_cols = [
            c for c, info in hu_cols.items()
            if info.get("srid") is not None
        ]
        assert len(geom_cols) > 0

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_schema(tmp_path / "nonexistent.gpkg")


class TestLoadSchemaMetadata:
    """Load schema metadata from YAML config."""

    def test_loads_from_existing_file(self, tmp_path: Path) -> None:
        import yaml

        meta = {
            "layers": {
                "test_layer": {
                    "description": "A test layer",
                    "columns": {"id": {"type": "integer"}},
                }
            }
        }
        meta_path = tmp_path / "schema_metadata.yaml"
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

        config = {"rag": {"schema_metadata": str(meta_path)}}
        result = load_schema_metadata(config)
        assert "layers" in result
        assert "test_layer" in result["layers"]

    def test_file_not_found_raises(self) -> None:
        config = {"rag": {"schema_metadata": "/nonexistent/path.yaml"}}
        with pytest.raises(FileNotFoundError):
            load_schema_metadata(config)
