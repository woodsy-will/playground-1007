"""Smoke tests for P3 ITC Delineation.

Quick sanity checks that core modules import, key functions are callable,
and basic operations produce expected types. Designed to run in < 1 second.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest


class TestP3Smoke:
    """Fast smoke tests for P3 tree crown delineation."""

    def test_imports(self):
        """All core P3 modules should import without error."""
        from projects.p3_itc_delineation.src import (
            chm,  # noqa: F401
            metrics,  # noqa: F401
            segmentation,  # noqa: F401
            treetops,  # noqa: F401
            validation,  # noqa: F401
        )

    def test_detect_treetops_runs(self, chm_path: Path, default_config: dict):
        """detect_treetops should return a GeoDataFrame with tree_id."""
        from projects.p3_itc_delineation.src.treetops import detect_treetops

        result = detect_treetops(chm_path, default_config)
        assert isinstance(result, gpd.GeoDataFrame)
        assert "tree_id" in result.columns
        assert len(result) > 0

    def test_segment_crowns_runs(
        self, chm_path: Path, sample_treetops: gpd.GeoDataFrame, default_config: dict
    ):
        """segment_crowns should return polygons."""
        pytest.importorskip("skimage")
        from projects.p3_itc_delineation.src.segmentation import segment_crowns

        result = segment_crowns(chm_path, sample_treetops, default_config)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0
        assert "crown_area_m2" in result.columns

    def test_extract_metrics_runs(
        self, chm_path: Path, sample_treetops: gpd.GeoDataFrame, default_config: dict
    ):
        """extract_tree_metrics should add height and allometric columns."""
        pytest.importorskip("skimage")
        from projects.p3_itc_delineation.src.metrics import extract_tree_metrics
        from projects.p3_itc_delineation.src.segmentation import segment_crowns

        crowns = segment_crowns(chm_path, sample_treetops, default_config)
        result = extract_tree_metrics(crowns, chm_path, default_config)
        assert "max_height_m" in result.columns
        assert "dbh_inches" in result.columns
        assert "quality_flag" in result.columns

    def test_allometry_functions_run(self):
        """Allometric equations should accept arrays and return arrays."""
        from shared.utils.allometry import (
            basal_area_sqft,
            dbh_from_crown_diameter,
            stem_volume_cuft,
        )

        cd = np.array([5.0, 10.0])
        dbh = dbh_from_crown_diameter(cd)
        assert len(dbh) == 2
        assert np.all(dbh > 0)

        ba = basal_area_sqft(dbh)
        assert np.all(ba > 0)

        vol = stem_volume_cuft(dbh, np.array([50.0, 80.0]))
        assert np.all(vol > 0)
