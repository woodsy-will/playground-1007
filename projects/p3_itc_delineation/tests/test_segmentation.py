"""Tests for crown segmentation module."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest


class TestSegmentCrowns:
    """Verify marker-controlled watershed crown segmentation."""

    def test_returns_geodataframe(
        self, chm_path: Path, sample_treetops: gpd.GeoDataFrame, default_config: dict
    ):
        """segment_crowns must return a GeoDataFrame."""
        skimage = pytest.importorskip("skimage")  # noqa: F841
        from projects.p3_itc_delineation.src.segmentation import segment_crowns

        result = segment_crowns(chm_path, sample_treetops, default_config)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_has_required_columns(
        self, chm_path: Path, sample_treetops: gpd.GeoDataFrame, default_config: dict
    ):
        """Output must have tree_id, crown_area_m2, crown_diameter_m, geometry."""
        skimage = pytest.importorskip("skimage")  # noqa: F841
        from projects.p3_itc_delineation.src.segmentation import segment_crowns

        result = segment_crowns(chm_path, sample_treetops, default_config)
        for col in ("tree_id", "crown_area_m2", "crown_diameter_m", "geometry"):
            assert col in result.columns, f"Missing column: {col}"

    def test_crown_areas_above_minimum(
        self, chm_path: Path, sample_treetops: gpd.GeoDataFrame, default_config: dict
    ):
        """All returned crowns must meet the min_crown_area threshold."""
        skimage = pytest.importorskip("skimage")  # noqa: F841
        from projects.p3_itc_delineation.src.segmentation import segment_crowns

        min_area = default_config["processing"]["min_crown_area"]
        result = segment_crowns(chm_path, sample_treetops, default_config)
        if len(result) > 0:
            assert np.all(result["crown_area_m2"].values >= min_area)

    def test_produces_polygons(
        self, chm_path: Path, sample_treetops: gpd.GeoDataFrame, default_config: dict
    ):
        """All geometries should be Polygon or MultiPolygon."""
        skimage = pytest.importorskip("skimage")  # noqa: F841
        from projects.p3_itc_delineation.src.segmentation import segment_crowns

        result = segment_crowns(chm_path, sample_treetops, default_config)
        if len(result) > 0:
            geom_types = result.geometry.geom_type.unique()
            assert all(
                g in ("Polygon", "MultiPolygon") for g in geom_types
            ), f"Unexpected geometry types: {geom_types}"
