"""Tests for treetop detection module."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np


class TestDetectTreetops:
    """Verify variable-window local maxima treetop detection."""

    def test_returns_geodataframe(self, chm_path: Path, default_config: dict):
        """detect_treetops must return a GeoDataFrame."""
        from projects.p3_itc_delineation.src.treetops import detect_treetops

        result = detect_treetops(chm_path, default_config)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_has_required_columns(self, chm_path: Path, default_config: dict):
        """Output must contain tree_id, height, and geometry columns."""
        from projects.p3_itc_delineation.src.treetops import detect_treetops

        result = detect_treetops(chm_path, default_config)
        for col in ("tree_id", "height", "geometry"):
            assert col in result.columns, f"Missing column: {col}"

    def test_detects_trees(self, chm_path: Path, default_config: dict):
        """Synthetic CHM with 5 trees should yield at least 3 detections."""
        from projects.p3_itc_delineation.src.treetops import detect_treetops

        result = detect_treetops(chm_path, default_config)
        assert len(result) >= 3, f"Expected >= 3 treetops, got {len(result)}"

    def test_heights_above_minimum(self, chm_path: Path, default_config: dict):
        """All detected heights must exceed the configured min_tree_height."""
        from projects.p3_itc_delineation.src.treetops import detect_treetops

        min_h = default_config["processing"]["min_tree_height"]
        result = detect_treetops(chm_path, default_config)
        assert np.all(result["height"].values >= min_h)

    def test_correct_crs(self, chm_path: Path, default_config: dict):
        """Output CRS must match the configured CRS."""
        from projects.p3_itc_delineation.src.treetops import detect_treetops

        result = detect_treetops(chm_path, default_config)
        assert result.crs is not None
        assert result.crs.to_epsg() == 3310
