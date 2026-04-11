"""System tests for the P3 ITC delineation pipeline.

Verifies the complete system end-to-end: a CHM raster is ingested,
treetops are detected, crowns are segmented, per-tree biometrics are
extracted, and the inventory is validated against field cruise data.
All tests run against synthetic data using the shared fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("skimage")

import geopandas as gpd  # noqa: E402

from projects.p3_itc_delineation.src.metrics import extract_tree_metrics  # noqa: E402
from projects.p3_itc_delineation.src.segmentation import segment_crowns  # noqa: E402
from projects.p3_itc_delineation.src.treetops import detect_treetops  # noqa: E402
from projects.p3_itc_delineation.src.validation import validate_against_cruise  # noqa: E402
from shared.utils.io import read_raster  # noqa: E402

# -----------------------------------------------------------------------
# Shared fixture: run the full CHM->treetops->crowns->metrics chain once
# -----------------------------------------------------------------------


@pytest.fixture()
def system_inventory(
    chm_path: Path,
    default_config: dict,
) -> gpd.GeoDataFrame:
    """Run the full delineation chain and return the enriched inventory."""
    treetops = detect_treetops(chm_path, default_config)
    crowns = segment_crowns(chm_path, treetops, default_config)
    metrics = extract_tree_metrics(crowns, chm_path, default_config)
    return metrics


@pytest.fixture()
def chm_max(chm_path: Path) -> float:
    """Return the maximum value in the CHM raster."""
    data, _ = read_raster(chm_path)
    return float(np.nanmax(data[0]))


# -----------------------------------------------------------------------
# System test class
# -----------------------------------------------------------------------


class TestITCDelineationSystem:
    """System-level tests for the P3 ITC delineation pipeline."""

    def test_system_produces_tree_inventory(
        self, system_inventory: gpd.GeoDataFrame,
    ) -> None:
        """The full chain must produce a GeoDataFrame with tree_id,
        crown_area_m2, dbh_inches, stem_volume_cuft, and quality_flag."""
        inv = system_inventory
        assert isinstance(inv, gpd.GeoDataFrame)
        assert len(inv) > 0

        expected_columns = [
            "tree_id",
            "crown_area_m2",
            "dbh_inches",
            "stem_volume_cuft",
            "quality_flag",
        ]
        for col in expected_columns:
            assert col in inv.columns, f"Missing column: {col}"

    def test_system_tree_count_reasonable(
        self,
        system_inventory: gpd.GeoDataFrame,
        chm_path: Path,
    ) -> None:
        """Number of detected trees should be > 0 and < total pixels."""
        data, _ = read_raster(chm_path)
        total_pixels = data[0].size

        n_trees = len(system_inventory)
        assert n_trees > 0, "No trees detected"
        assert n_trees < total_pixels, (
            f"Tree count ({n_trees}) >= total pixels ({total_pixels})"
        )

    def test_system_heights_match_chm(
        self,
        system_inventory: gpd.GeoDataFrame,
        chm_max: float,
    ) -> None:
        """Max height per tree should not exceed max CHM value."""
        inv = system_inventory
        assert "max_height_m" in inv.columns
        max_tree_height = inv["max_height_m"].max()
        assert max_tree_height <= chm_max + 1e-6, (
            f"Max tree height ({max_tree_height:.2f}) exceeds "
            f"CHM max ({chm_max:.2f})"
        )

    def test_system_crown_areas_positive(
        self, system_inventory: gpd.GeoDataFrame,
    ) -> None:
        """All crown areas should be > 0."""
        assert (system_inventory["crown_area_m2"] > 0).all(), (
            "Found crown areas <= 0"
        )

    def test_system_allometry_consistent(
        self, system_inventory: gpd.GeoDataFrame,
    ) -> None:
        """Trees with larger crowns should generally have larger DBH
        (positive correlation)."""
        inv = system_inventory
        if len(inv) < 3:
            pytest.skip("Need at least 3 trees for correlation check")

        crown_areas = inv["crown_area_m2"].values
        dbh_values = inv["dbh_inches"].values

        # Compute Pearson correlation
        corr = np.corrcoef(crown_areas, dbh_values)[0, 1]
        assert corr > 0, (
            f"Expected positive correlation between crown area and DBH, "
            f"got r={corr:.3f}"
        )

    def test_system_validation_produces_metrics(
        self,
        chm_path: Path,
        default_config: dict,
    ) -> None:
        """Full chain including validation against cruise data should
        produce detection_rate, rmse_height, etc."""
        treetops = detect_treetops(chm_path, default_config)
        crowns = segment_crowns(chm_path, treetops, default_config)
        metrics = extract_tree_metrics(crowns, chm_path, default_config)

        # Convert to centroids for validation matching
        metrics_pts = metrics.copy()
        metrics_pts["geometry"] = metrics_pts.geometry.centroid

        cruise_csv = default_config["data"]["cruise_plots"]
        val_result = validate_against_cruise(metrics_pts, cruise_csv, default_config)

        assert isinstance(val_result, dict)
        assert "detection_rate" in val_result
        assert "n_predicted" in val_result
        assert "n_reference" in val_result
        assert "n_matched" in val_result
        assert "rmse_height_m" in val_result
        assert val_result["n_predicted"] > 0
        assert val_result["n_reference"] > 0
        assert 0.0 <= val_result["detection_rate"] <= 1.0
