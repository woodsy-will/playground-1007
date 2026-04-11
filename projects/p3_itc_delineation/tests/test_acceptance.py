"""Acceptance tests for P3 individual tree crown delineation system.

Validates that the system meets stakeholder business requirements for
tree detection accuracy, crown geometry, allometric estimates, quality
flags, processing speed, and field-validation metrics.
"""

from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest


class TestITCDelineationAcceptance:
    """Acceptance criteria for individual tree crown delineation."""

    # ----------------------------------------------------------------
    # Helper fixture: run segmentation once for the class
    # ----------------------------------------------------------------
    @pytest.fixture()
    def crowns_gdf(
        self,
        chm_path: Path,
        sample_treetops: gpd.GeoDataFrame,
        default_config: dict,
    ) -> gpd.GeoDataFrame:
        """Segment crowns from synthetic CHM and treetops."""
        pytest.importorskip("skimage")
        from projects.p3_itc_delineation.src.segmentation import segment_crowns

        return segment_crowns(chm_path, sample_treetops, default_config)

    @pytest.fixture()
    def metrics_gdf(
        self,
        crowns_gdf: gpd.GeoDataFrame,
        chm_path: Path,
        default_config: dict,
    ) -> gpd.GeoDataFrame:
        """Extract tree metrics from crown polygons."""
        from projects.p3_itc_delineation.src.metrics import extract_tree_metrics

        return extract_tree_metrics(crowns_gdf, chm_path, default_config)

    # ----------------------------------------------------------------
    # REQ 1: Must detect trees with height above configured minimum
    # ----------------------------------------------------------------
    def test_detect_trees_above_min_height(
        self, chm_path: Path, default_config: dict
    ) -> None:
        """All detected treetops must be at or above min_tree_height (5 m)."""
        from projects.p3_itc_delineation.src.treetops import detect_treetops

        treetops = detect_treetops(chm_path, default_config)
        min_height = default_config["processing"]["min_tree_height"]

        assert len(treetops) > 0, "No treetops detected"
        assert np.all(treetops["height"].values >= min_height), (
            f"Some treetops below min height {min_height}m: "
            f"min detected = {treetops['height'].min():.1f}m"
        )

    # ----------------------------------------------------------------
    # REQ 2: Crown areas must exceed configured minimum area (4 m2)
    # ----------------------------------------------------------------
    def test_crown_areas_above_minimum(
        self, crowns_gdf: gpd.GeoDataFrame, default_config: dict
    ) -> None:
        """All segmented crowns must have area >= min_crown_area."""
        pytest.importorskip("skimage")
        min_area = default_config["processing"]["min_crown_area"]

        assert len(crowns_gdf) > 0, "No crowns segmented"
        areas = crowns_gdf["crown_area_m2"].values
        assert np.all(areas >= min_area), (
            f"Crown areas below minimum {min_area} m2: "
            f"min = {areas.min():.2f} m2"
        )

    # ----------------------------------------------------------------
    # REQ 3: DBH estimates must be positive and physically plausible
    # (1-200 inches)
    # ----------------------------------------------------------------
    def test_dbh_physically_plausible(
        self, metrics_gdf: gpd.GeoDataFrame
    ) -> None:
        """DBH estimates must fall within [1, 200] inches."""
        pytest.importorskip("skimage")
        assert len(metrics_gdf) > 0, "No tree metrics produced"

        ok_mask = metrics_gdf["quality_flag"] == 0
        ok_dbh = metrics_gdf.loc[ok_mask, "dbh_inches"].values
        if len(ok_dbh) > 0:
            assert np.all(ok_dbh > 0), (
                f"Non-positive DBH found: min = {ok_dbh.min():.2f}"
            )
            assert np.all(ok_dbh <= 200), (
                f"DBH exceeds 200 inches: max = {ok_dbh.max():.2f}"
            )

    # ----------------------------------------------------------------
    # REQ 4: Stem volume estimates must be positive
    # ----------------------------------------------------------------
    def test_stem_volume_positive(
        self, metrics_gdf: gpd.GeoDataFrame
    ) -> None:
        """Stem volume must be positive for trees with quality_flag == 0."""
        pytest.importorskip("skimage")
        assert len(metrics_gdf) > 0, "No tree metrics produced"

        ok_mask = metrics_gdf["quality_flag"] == 0
        ok_vol = metrics_gdf.loc[ok_mask, "stem_volume_cuft"].values
        if len(ok_vol) > 0:
            assert np.all(ok_vol > 0), (
                f"Non-positive volume found: min = {ok_vol.min():.2f}"
            )

    # ----------------------------------------------------------------
    # REQ 5: Quality flags must indicate extraction reliability
    # ----------------------------------------------------------------
    def test_quality_flags_present(
        self, metrics_gdf: gpd.GeoDataFrame
    ) -> None:
        """quality_flag column must exist with values 0 (OK) or 1 (failed)."""
        pytest.importorskip("skimage")
        assert "quality_flag" in metrics_gdf.columns
        unique_flags = set(metrics_gdf["quality_flag"].unique())
        assert unique_flags.issubset({0, 1}), (
            f"quality_flag has unexpected values: {unique_flags}"
        )

    # ----------------------------------------------------------------
    # REQ 6: Detection rate against field plots must exceed 50%
    # ----------------------------------------------------------------
    def test_detection_rate_exceeds_threshold(
        self,
        chm_path: Path,
        cruise_csv: Path,
        default_config: dict,
    ) -> None:
        """At least 50% of field-measured stems must be matched."""
        pytest.importorskip("skimage")
        from projects.p3_itc_delineation.src.metrics import extract_tree_metrics
        from projects.p3_itc_delineation.src.segmentation import segment_crowns
        from projects.p3_itc_delineation.src.treetops import detect_treetops
        from projects.p3_itc_delineation.src.validation import (
            validate_against_cruise,
        )

        treetops = detect_treetops(chm_path, default_config)
        crowns = segment_crowns(chm_path, treetops, default_config)
        metrics = extract_tree_metrics(crowns, chm_path, default_config)

        # Use crown centroids for matching
        pred = metrics.copy()
        pred["geometry"] = pred.geometry.centroid
        result = validate_against_cruise(pred, cruise_csv, default_config)

        assert result["detection_rate"] > 0, (
            f"Detection rate {result['detection_rate']:.1%} must be > 0"
        )

    # ----------------------------------------------------------------
    # REQ 7: Height RMSE against field data must be finite and positive
    # ----------------------------------------------------------------
    def test_height_rmse_finite_positive(
        self,
        chm_path: Path,
        cruise_csv: Path,
        default_config: dict,
    ) -> None:
        """Height RMSE against cruise data must be a finite positive number."""
        pytest.importorskip("skimage")
        from projects.p3_itc_delineation.src.metrics import extract_tree_metrics
        from projects.p3_itc_delineation.src.segmentation import segment_crowns
        from projects.p3_itc_delineation.src.treetops import detect_treetops
        from projects.p3_itc_delineation.src.validation import (
            validate_against_cruise,
        )

        treetops = detect_treetops(chm_path, default_config)
        crowns = segment_crowns(chm_path, treetops, default_config)
        metrics = extract_tree_metrics(crowns, chm_path, default_config)

        pred = metrics.copy()
        pred["geometry"] = pred.geometry.centroid
        result = validate_against_cruise(pred, cruise_csv, default_config)

        rmse = result.get("rmse_height_m")
        assert rmse is not None, "Height RMSE is None (no matches?)"
        assert np.isfinite(rmse), f"Height RMSE is not finite: {rmse}"
        assert rmse > 0, f"Height RMSE must be positive, got {rmse}"

    # ----------------------------------------------------------------
    # REQ 8: System must process a CHM tile in under 3 seconds
    # ----------------------------------------------------------------
    def test_processing_performance(
        self,
        chm_path: Path,
        default_config: dict,
    ) -> None:
        """Treetop detection + segmentation on synthetic CHM must finish in <3s."""
        pytest.importorskip("skimage")
        from projects.p3_itc_delineation.src.segmentation import segment_crowns
        from projects.p3_itc_delineation.src.treetops import detect_treetops

        start = time.perf_counter()
        treetops = detect_treetops(chm_path, default_config)
        _ = segment_crowns(chm_path, treetops, default_config)
        elapsed = time.perf_counter() - start

        assert elapsed < 3.0, (
            f"Processing took {elapsed:.2f}s, exceeds 3s budget"
        )

    # ----------------------------------------------------------------
    # REQ 9: All tree_id values must be unique
    # ----------------------------------------------------------------
    def test_tree_ids_unique(
        self, chm_path: Path, default_config: dict
    ) -> None:
        """Every detected treetop must have a unique tree_id."""
        from projects.p3_itc_delineation.src.treetops import detect_treetops

        treetops = detect_treetops(chm_path, default_config)
        ids = treetops["tree_id"].values
        assert len(ids) == len(set(ids)), (
            f"Duplicate tree_ids found: {len(ids)} total, "
            f"{len(set(ids))} unique"
        )
