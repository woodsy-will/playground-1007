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
