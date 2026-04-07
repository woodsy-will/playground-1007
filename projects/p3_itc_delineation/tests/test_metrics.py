"""Tests for tree metrics extraction module."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest


class TestExtractTreeMetrics:
    """Verify zonal statistics and allometric calculations."""

    @pytest.fixture()
    def crowns_gdf(
        self, chm_path: Path, sample_treetops: gpd.GeoDataFrame, default_config: dict
    ) -> gpd.GeoDataFrame:
        """Produce crown polygons from the synthetic data for metric tests."""
        skimage = pytest.importorskip("skimage")  # noqa: F841
        from projects.p3_itc_delineation.src.segmentation import segment_crowns

        return segment_crowns(chm_path, sample_treetops, default_config)

    def test_returns_geodataframe(
        self, crowns_gdf: gpd.GeoDataFrame, chm_path: Path, default_config: dict
    ):
        """extract_tree_metrics must return a GeoDataFrame."""
        from projects.p3_itc_delineation.src.metrics import extract_tree_metrics

        result = extract_tree_metrics(crowns_gdf, chm_path, default_config)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_has_allometric_columns(
        self, crowns_gdf: gpd.GeoDataFrame, chm_path: Path, default_config: dict
    ):
        """Output must contain height, DBH, and volume columns."""
        from projects.p3_itc_delineation.src.metrics import extract_tree_metrics

        result = extract_tree_metrics(crowns_gdf, chm_path, default_config)
        for col in ("max_height_m", "mean_height_m", "dbh_inches", "stem_volume_cuft"):
            assert col in result.columns, f"Missing column: {col}"

    def test_heights_non_negative(
        self, crowns_gdf: gpd.GeoDataFrame, chm_path: Path, default_config: dict
    ):
        """Max and mean heights should be non-negative."""
        from projects.p3_itc_delineation.src.metrics import extract_tree_metrics

        result = extract_tree_metrics(crowns_gdf, chm_path, default_config)
        if len(result) > 0:
            assert np.all(result["max_height_m"].values >= 0)
            assert np.all(result["mean_height_m"].values >= 0)

    def test_dbh_non_negative(
        self, crowns_gdf: gpd.GeoDataFrame, chm_path: Path, default_config: dict
    ):
        """Estimated DBH values must be non-negative."""
        from projects.p3_itc_delineation.src.metrics import extract_tree_metrics

        result = extract_tree_metrics(crowns_gdf, chm_path, default_config)
        if len(result) > 0:
            assert np.all(result["dbh_inches"].values >= 0)

    def test_volume_non_negative(
        self, crowns_gdf: gpd.GeoDataFrame, chm_path: Path, default_config: dict
    ):
        """Estimated stem volumes must be non-negative."""
        from projects.p3_itc_delineation.src.metrics import extract_tree_metrics

        result = extract_tree_metrics(crowns_gdf, chm_path, default_config)
        if len(result) > 0:
            assert np.all(result["stem_volume_cuft"].values >= 0)


class TestAllometryValues:
    """Verify allometric equations produce mathematically correct values."""

    def test_dbh_from_known_crown_diameter(self):
        """Hand-computed: crown_diameter 6m \u2192 6*3.28084=19.685ft
        DBH = (19.685 - 3.0) / 0.25 = 66.74 inches."""
        from shared.utils.allometry import dbh_from_crown_diameter

        dbh = dbh_from_crown_diameter(np.array([6.0]))
        expected = (6.0 * 3.28084 - 3.0) / 0.25
        np.testing.assert_allclose(dbh[0], expected, atol=0.01)

    def test_volume_from_known_dbh_height(self):
        """Hand-computed: DBH=20in, height=80ft
        volume = 0.002 * 20^2 * 80 = 64.0 cuft."""
        from shared.utils.allometry import stem_volume_cuft

        vol = stem_volume_cuft(np.array([20.0]), np.array([80.0]))
        np.testing.assert_allclose(vol[0], 64.0, atol=0.01)

    def test_basal_area_from_known_dbh(self):
        """BA = 0.005454 * DBH^2. DBH=10in \u2192 BA = 0.5454 sqft."""
        from shared.utils.allometry import basal_area_sqft

        ba = basal_area_sqft(np.array([10.0]))
        np.testing.assert_allclose(ba[0], 0.5454, atol=0.001)
