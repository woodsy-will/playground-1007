"""Shared pytest fixtures for P2 LLM spatial query tests.

Generates a small test GeoPackage using the shared synthetic data
generator so that every test module starts from a known, reproducible
state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.data.generate_synthetic import generate_synthetic_geopackage


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a session-scoped temporary directory with synthetic data."""
    d = tmp_path_factory.mktemp("p2_synthetic")
    generate_synthetic_geopackage(d)
    return d


@pytest.fixture(scope="session")
def gpkg_path(synthetic_dir: Path) -> Path:
    """Path to synthetic forestry GeoPackage."""
    return synthetic_dir / "forest_management.gpkg"


@pytest.fixture()
def default_config(gpkg_path: Path) -> dict:
    """Minimal config dict for testing — mirrors default.yaml structure."""
    return {
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
            "few_shot_examples": "configs/few_shot_queries.yaml",
            "schema_metadata": "configs/schema_metadata.yaml",
            "chunk_size": 512,
            "top_k": 5,
        },
    }


@pytest.fixture()
def sample_schema_meta() -> dict:
    """Minimal schema metadata for prompt builder tests."""
    return {
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
