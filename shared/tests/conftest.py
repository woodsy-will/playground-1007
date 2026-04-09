"""Shared pytest fixtures for shared utility tests."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from shared.utils.io import make_profile, write_raster


@pytest.fixture()
def tmp_raster(tmp_path: Path) -> Path:
    """Write a small synthetic raster and return its path."""
    profile = make_profile(
        bounds=(0.0, 0.0, 100.0, 100.0),
        resolution=10.0,
        crs="EPSG:3310",
    )
    data = np.arange(100, dtype=np.float32).reshape(1, 10, 10)
    return write_raster(tmp_path / "test.tif", data, profile)


@pytest.fixture()
def tmp_yaml(tmp_path: Path) -> Path:
    """Write a small YAML config and return its path."""
    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text("data:\n  output_dir: /tmp/out\nprocessing:\n  crs: EPSG:3310\n")
    return cfg_path


@pytest.fixture()
def sample_gdf() -> gpd.GeoDataFrame:
    """A tiny GeoDataFrame for vector I/O tests."""
    return gpd.GeoDataFrame(
        {"name": ["a", "b"], "value": [1, 2]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:3310",
    )
