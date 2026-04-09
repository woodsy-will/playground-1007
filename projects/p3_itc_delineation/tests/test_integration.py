"""Integration tests for the P3 ITC delineation pipeline.

Tests the end-to-end flow from CHM through treetop detection, crown
segmentation, and metric extraction using synthetic fixtures.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest


class TestChmToTreetops:
    """Test treetop detection from a CHM raster."""

    def test_chm_to_treetops(self, chm_path: Path, default_config: dict) -> None:
        """detect_treetops should return a GeoDataFrame with tree_id and height."""
        from projects.p3_itc_delineation.src.treetops import detect_treetops

        result = detect_treetops(chm_path, default_config)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0
        assert "tree_id" in result.columns
        assert "height" in result.columns
        assert result.crs is not None


class TestTreetopsToCrowns:
    """Test crown segmentation from detected treetops."""

    def test_treetops_to_crowns(self, chm_path: Path, default_config: dict) -> None:
        """segment_crowns should produce crown polygons from treetop points."""
        pytest.importorskip("skimage")

        from projects.p3_itc_delineation.src.segmentation import segment_crowns
        from projects.p3_itc_delineation.src.treetops import detect_treetops

        treetops = detect_treetops(chm_path, default_config)
        crowns = segment_crowns(chm_path, treetops, default_config)

        assert isinstance(crowns, gpd.GeoDataFrame)
        assert len(crowns) > 0
        assert "tree_id" in crowns.columns
        assert "crown_area_m2" in crowns.columns
        assert "crown_diameter_m" in crowns.columns
        # All crown areas should be positive
        assert (crowns["crown_area_m2"] > 0).all()


class TestCrownsToMetrics:
    """Test metric extraction from crown polygons."""

    def test_crowns_to_metrics(self, chm_path: Path, default_config: dict) -> None:
        """extract_tree_metrics should add height, DBH, and volume columns."""
        pytest.importorskip("skimage")

        from projects.p3_itc_delineation.src.metrics import extract_tree_metrics
        from projects.p3_itc_delineation.src.segmentation import segment_crowns
        from projects.p3_itc_delineation.src.treetops import detect_treetops

        treetops = detect_treetops(chm_path, default_config)
        crowns = segment_crowns(chm_path, treetops, default_config)
        metrics = extract_tree_metrics(crowns, chm_path, default_config)

        assert isinstance(metrics, gpd.GeoDataFrame)
        assert len(metrics) > 0

        expected_columns = [
            "tree_id",
            "crown_area_m2",
            "crown_diameter_m",
            "max_height_m",
            "mean_height_m",
            "quality_flag",
            "dbh_inches",
            "stem_volume_cuft",
        ]
        for col in expected_columns:
            assert col in metrics.columns, f"Missing column: {col}"


class TestFullDelineationChain:
    """Test the full delineation pipeline from CHM to final metrics."""

    def test_full_chain(self, chm_path: Path, default_config: dict) -> None:
        """Full pipeline: CHM -> treetops -> crowns -> metrics with all outputs valid."""
        pytest.importorskip("skimage")

        from projects.p3_itc_delineation.src.metrics import extract_tree_metrics
        from projects.p3_itc_delineation.src.segmentation import segment_crowns
        from projects.p3_itc_delineation.src.treetops import detect_treetops

        # Step 1: Detect treetops
        treetops = detect_treetops(chm_path, default_config)
        assert len(treetops) > 0

        # Step 2: Segment crowns
        crowns = segment_crowns(chm_path, treetops, default_config)
        assert len(crowns) > 0

        # Step 3: Extract metrics
        metrics = extract_tree_metrics(crowns, chm_path, default_config)
        assert len(metrics) > 0

        # Verify final output has critical columns
        assert "quality_flag" in metrics.columns
        assert "dbh_inches" in metrics.columns
        assert "stem_volume_cuft" in metrics.columns
        assert "max_height_m" in metrics.columns

        # Quality flags should be 0 (OK) for valid synthetic data
        assert (metrics["quality_flag"] == 0).all()

        # DBH and volume should be positive for valid trees
        assert (metrics["dbh_inches"] > 0).all()
        assert (metrics["stem_volume_cuft"] > 0).all()

        # Heights should be within reasonable range for synthetic data
        assert (metrics["max_height_m"] >= 0).all()
        assert (metrics["mean_height_m"] >= 0).all()
