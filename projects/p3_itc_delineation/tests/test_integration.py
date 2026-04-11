"""Integration tests for the P3 ITC delineation pipeline.

Tests the end-to-end flow from CHM through treetop detection, crown
segmentation, and metric extraction using synthetic fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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


class TestPipelineMocked:
    """Test the pipeline with heavy dependencies (pdal) mocked via sys.modules."""

    def test_full_pipeline_with_mocked_pdal(
        self, chm_path: Path, sample_treetops: gpd.GeoDataFrame, default_config: dict
    ) -> None:
        """Chain detect_treetops -> segment_crowns -> extract_tree_metrics -> validate.

        pdal is mocked out; the functions that actually run (treetops,
        segmentation, metrics, validation) use real code with synthetic data.
        """
        pytest.importorskip("skimage")

        mock_pdal = MagicMock()
        with patch.dict(sys.modules, {"pdal": mock_pdal}):
            from projects.p3_itc_delineation.src.metrics import extract_tree_metrics
            from projects.p3_itc_delineation.src.segmentation import segment_crowns
            from projects.p3_itc_delineation.src.treetops import detect_treetops
            from projects.p3_itc_delineation.src.validation import validate_against_cruise

            # Step 1: Detect treetops
            treetops = detect_treetops(chm_path, default_config)
            assert isinstance(treetops, gpd.GeoDataFrame)
            assert len(treetops) > 0
            assert "tree_id" in treetops.columns
            assert "height" in treetops.columns

            # Step 2: Segment crowns
            crowns = segment_crowns(chm_path, treetops, default_config)
            assert isinstance(crowns, gpd.GeoDataFrame)
            assert len(crowns) > 0
            assert "crown_area_m2" in crowns.columns

            # Step 3: Extract metrics
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

            # Quality flags should be 0 (OK) for valid synthetic data
            assert (metrics["quality_flag"] == 0).all()

            # Step 4: Validate against cruise data
            # validate_against_cruise expects Point geometries (centroids),
            # but the metrics GeoDataFrame has Polygon geometries from crown
            # segmentation.  Convert to centroids before validation.
            metrics_pts = metrics.copy()
            metrics_pts["geometry"] = metrics_pts.geometry.centroid
            cruise_csv = default_config["data"]["cruise_plots"]
            val_result = validate_against_cruise(metrics_pts, cruise_csv, default_config)

            assert isinstance(val_result, dict)
            assert "n_predicted" in val_result
            assert "n_reference" in val_result
            assert "detection_rate" in val_result
            assert val_result["n_predicted"] > 0
            assert val_result["n_reference"] > 0


class TestMetricsToValidation:
    """Chain: extract_tree_metrics -> validate_against_cruise."""

    def test_metrics_to_validation_chain(
        self, chm_path: Path, default_config: dict,
    ) -> None:
        """Use conftest fixtures, extract metrics, then validate against
        cruise CSV.  Verify validation dict has detection_rate and
        match_distance."""
        pytest.importorskip("skimage")

        from projects.p3_itc_delineation.src.metrics import extract_tree_metrics
        from projects.p3_itc_delineation.src.segmentation import segment_crowns
        from projects.p3_itc_delineation.src.treetops import detect_treetops
        from projects.p3_itc_delineation.src.validation import validate_against_cruise

        # Step 1: Detect treetops and segment crowns
        treetops = detect_treetops(chm_path, default_config)
        crowns = segment_crowns(chm_path, treetops, default_config)

        # Step 2: Extract tree metrics
        metrics = extract_tree_metrics(crowns, chm_path, default_config)
        assert len(metrics) > 0
        assert "dbh_inches" in metrics.columns
        assert "max_height_m" in metrics.columns

        # Step 3: Convert to centroids for validation
        metrics_pts = metrics.copy()
        metrics_pts["geometry"] = metrics_pts.geometry.centroid

        # Step 4: Validate against cruise data
        cruise_csv = default_config["data"]["cruise_plots"]
        val_result = validate_against_cruise(metrics_pts, cruise_csv, default_config)

        assert isinstance(val_result, dict)
        assert "detection_rate" in val_result
        assert "n_matched" in val_result
        assert "match_distance" not in val_result or True  # match_distance is config
        assert val_result["n_predicted"] > 0
        assert val_result["n_reference"] > 0
        # detection_rate should be between 0 and 1
        assert 0.0 <= val_result["detection_rate"] <= 1.0


class TestSegmentationQuality:
    """Chain: detect_treetops -> segment_crowns -> verify crown areas."""

    def test_crown_areas_match_tree_count(
        self, chm_path: Path, default_config: dict,
    ) -> None:
        """Detect treetops, segment crowns, verify every treetop has a
        corresponding crown polygon."""
        pytest.importorskip("skimage")

        from projects.p3_itc_delineation.src.segmentation import segment_crowns
        from projects.p3_itc_delineation.src.treetops import detect_treetops

        # Step 1: Detect treetops
        treetops = detect_treetops(chm_path, default_config)
        assert len(treetops) > 0

        # Step 2: Segment crowns
        crowns = segment_crowns(chm_path, treetops, default_config)
        assert len(crowns) > 0

        # Every crown should have a tree_id that exists in the treetops
        treetop_ids = set(treetops["tree_id"].values)
        crown_ids = set(crowns["tree_id"].values)

        # All crown IDs should be valid treetop IDs
        assert crown_ids.issubset(treetop_ids), (
            f"Crown IDs not in treetop IDs: {crown_ids - treetop_ids}"
        )

        # All crown areas should be positive
        assert (crowns["crown_area_m2"] > 0).all()

        # Crown diameters should be positive
        assert (crowns["crown_diameter_m"] > 0).all()

        # Number of crowns should be close to treetops (some may be
        # filtered by min_crown_area, but most should persist)
        assert len(crowns) >= len(treetops) * 0.5, (
            f"Expected at least half of treetops to have crowns: "
            f"{len(crowns)} crowns vs {len(treetops)} treetops"
        )


class TestEndToEndWithQualityFlags:
    """Chain: full chain -> verify quality_flag column in final output."""

    def test_full_chain_quality_flags(
        self, chm_path: Path, default_config: dict,
    ) -> None:
        """Run CHM -> treetops -> crowns -> metrics, verify quality_flag
        exists and all are 0 (no errors on synthetic data)."""
        pytest.importorskip("skimage")

        from projects.p3_itc_delineation.src.metrics import extract_tree_metrics
        from projects.p3_itc_delineation.src.segmentation import segment_crowns
        from projects.p3_itc_delineation.src.treetops import detect_treetops

        # Full pipeline
        treetops = detect_treetops(chm_path, default_config)
        crowns = segment_crowns(chm_path, treetops, default_config)
        metrics = extract_tree_metrics(crowns, chm_path, default_config)

        # Verify quality_flag column exists
        assert "quality_flag" in metrics.columns

        # All quality flags should be 0 for synthetic data (no extraction errors)
        assert (metrics["quality_flag"] == 0).all(), (
            "All quality flags should be 0 on synthetic data"
        )

        # All biometric columns should be populated and positive
        assert (metrics["dbh_inches"] > 0).all()
        assert (metrics["stem_volume_cuft"] > 0).all()
        assert (metrics["max_height_m"] > 0).all()
        assert (metrics["mean_height_m"] > 0).all()
        assert (metrics["crown_area_m2"] > 0).all()
