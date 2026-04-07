"""Shared pytest fixtures for P4 habitat suitability tests.

Generates synthetic occurrence points and environmental predictor rasters
using the shared data generator so every test starts from a known,
reproducible state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.data.generate_synthetic import (
    generate_synthetic_occurrences,
    generate_synthetic_predictors,
)


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped temporary directory with all P4 synthetic data."""
    d = tmp_path_factory.mktemp("p4_synthetic")
    generate_synthetic_occurrences(d)
    generate_synthetic_predictors(d)
    return d


@pytest.fixture(scope="session")
def occurrences_path(synthetic_dir: Path) -> Path:
    """Path to synthetic occurrence GeoPackage."""
    return synthetic_dir / "occurrences.gpkg"


@pytest.fixture(scope="session")
def predictor_dir(synthetic_dir: Path) -> Path:
    """Directory containing synthetic predictor GeoTIFFs."""
    return synthetic_dir


@pytest.fixture()
def default_config(synthetic_dir: Path) -> dict:
    """Minimal config dict mirroring default.yaml for testing."""
    return {
        "species": {
            "name": "Pekania pennanti",
            "common_name": "Pacific fisher",
            "occurrence_sources": ["GBIF", "CNDDB"],
            "thinning_distance_km": 1.0,
        },
        "data": {
            "occurrences_path": str(synthetic_dir / "occurrences.gpkg"),
            "predictor_dir": str(synthetic_dir),
            "output_dir": str(synthetic_dir / "output"),
        },
        "predictors": {
            "topographic": ["elevation", "slope", "tpi", "twi"],
            "climatic_source": "worldclim_v2",
            "canopy_source": "nlcd",
        },
        "modeling": {
            "crs": "EPSG:3310",
            "algorithms": ["maxent", "random_forest"],
            "cv_method": "spatial_block",
            "cv_folds": 5,
            "background_sampling": "target_group",
        },
        "projection": {
            "scenarios": ["ssp245", "ssp585"],
            "time_steps": [2050, 2090],
            "climate_source": "basin_characterization_model",
        },
    }
