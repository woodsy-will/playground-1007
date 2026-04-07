"""Shared pytest fixtures for P3 ITC delineation tests.

Generates synthetic CHM, DTM, cruise-plot, and treetop data using the
shared synthetic data generator so that every test module starts from a
known, reproducible state.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from shared.data.generate_synthetic import (
    AOI_BOUNDS,
    generate_synthetic_chm,
    generate_synthetic_cruise_plots,
    generate_synthetic_dtm,
    generate_synthetic_lidar,
)


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a session-scoped temporary directory with all synthetic data."""
    d = tmp_path_factory.mktemp("p3_synthetic")
    generate_synthetic_lidar(d)
    generate_synthetic_chm(d)
    generate_synthetic_dtm(d)
    generate_synthetic_cruise_plots(d)
    return d


@pytest.fixture(scope="session")
def chm_path(synthetic_dir: Path) -> Path:
    """Path to synthetic CHM GeoTIFF."""
    return synthetic_dir / "synthetic_chm.tif"


@pytest.fixture(scope="session")
def dtm_path(synthetic_dir: Path) -> Path:
    """Path to synthetic DTM GeoTIFF."""
    return synthetic_dir / "synthetic_dtm.tif"


@pytest.fixture(scope="session")
def cruise_csv(synthetic_dir: Path) -> Path:
    """Path to synthetic cruise plot CSV."""
    return synthetic_dir / "cruise_plots.csv"


@pytest.fixture()
def default_config(synthetic_dir: Path) -> dict:
    """Minimal config dict for testing — mirrors default.yaml structure."""
    return {
        "data": {
            "lidar_dir": str(synthetic_dir),
            "cruise_plots": str(synthetic_dir / "cruise_plots.csv"),
            "output_dir": str(synthetic_dir / "output"),
        },
        "processing": {
            "crs": "EPSG:3310",
            "dtm_resolution": 1.0,
            "chm_smoothing_sigma": 0.67,
            "min_tree_height": 5.0,
            "min_crown_area": 4.0,
            "local_max_method": "variable_window",
            "segmentation_method": "watershed",
        },
        "allometry": {
            "source": "pillsbury_kirkley_1984",
            "ba_constant": 0.005454,
        },
        "validation": {
            "match_distance": 3.0,
            "stratify_by": ["species", "diameter_class"],
        },
    }


@pytest.fixture()
def sample_treetops(chm_path: Path) -> gpd.GeoDataFrame:
    """A small GeoDataFrame of treetop points placed at synthetic CHM peaks."""
    from shared.utils.io import read_raster

    data, profile = read_raster(chm_path)
    chm = data[0]
    transform = profile["transform"]

    # Find the top-5 brightest pixels as proxy treetops
    flat = chm.ravel()
    top_idx = np.argsort(flat)[-5:]
    rows, cols = np.unravel_index(top_idx, chm.shape)
    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e
    heights = chm[rows, cols]

    return gpd.GeoDataFrame(
        {"tree_id": np.arange(1, 6), "height": heights},
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs="EPSG:3310",
    )
