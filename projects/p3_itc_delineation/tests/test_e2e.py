"""End-to-end tests for the ITC delineation pipeline.

User story: A forester processes a LiDAR-derived CHM to get individual
tree measurements (height, DBH, volume) and validates against field plots.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

skimage = pytest.importorskip("skimage")

import geopandas as gpd  # noqa: E402

from projects.p3_itc_delineation.src.metrics import extract_tree_metrics  # noqa: E402
from projects.p3_itc_delineation.src.segmentation import segment_crowns  # noqa: E402
from projects.p3_itc_delineation.src.treetops import detect_treetops  # noqa: E402
from projects.p3_itc_delineation.src.validation import validate_against_cruise  # noqa: E402
from shared.utils.io import read_vector, write_vector  # noqa: E402


class TestITCDelineationE2E:
    """E2E tests simulating a forester's tree inventory workflow."""

    @pytest.fixture()
    def full_chain(
        self, chm_path: Path, cruise_csv: Path, default_config: dict
    ) -> dict:
        """Run the full CHM-to-metrics chain and return all artefacts."""
        treetops = detect_treetops(chm_path, default_config)
        crowns = segment_crowns(chm_path, treetops, default_config)
        metrics = extract_tree_metrics(crowns, chm_path, default_config)

        pred_centroids = metrics.copy()
        pred_centroids["geometry"] = pred_centroids.geometry.centroid
        validation = validate_against_cruise(
            pred_centroids, cruise_csv, default_config
        )

        return {
            "treetops": treetops,
            "crowns": crowns,
            "metrics": metrics,
            "validation": validation,
            "config": default_config,
        }

    # ------------------------------------------------------------------ #
    # 1. Full flow: detect -> segment -> metrics -> verify schema
    # ------------------------------------------------------------------ #
    def test_user_processes_chm_to_tree_inventory(
        self, full_chain: dict
    ) -> None:
        metrics = full_chain["metrics"]

        assert isinstance(metrics, gpd.GeoDataFrame)
        expected_cols = [
            "tree_id",
            "crown_area_m2",
            "crown_diameter_m",
            "max_height_m",
            "mean_height_m",
            "dbh_inches",
            "stem_volume_cuft",
            "quality_flag",
        ]
        for col in expected_cols:
            assert col in metrics.columns, f"Missing column: {col}"

        assert len(metrics) > 0, "No trees detected"

    # ------------------------------------------------------------------ #
    # 2. Export to GeoPackage and round-trip
    # ------------------------------------------------------------------ #
    def test_user_exports_tree_data(
        self, tmp_path: Path, full_chain: dict
    ) -> None:
        metrics = full_chain["metrics"]
        out_path = tmp_path / "tree_inventory.gpkg"

        write_vector(metrics, out_path)
        assert out_path.exists()

        roundtrip = read_vector(out_path)

        # All columns preserved
        for col in metrics.columns:
            assert col in roundtrip.columns, f"Column lost in round-trip: {col}"

        # Same row count
        assert len(roundtrip) == len(metrics)

        # Geometry preserved
        assert roundtrip.geometry is not None
        assert all(roundtrip.geometry.is_valid)

    # ------------------------------------------------------------------ #
    # 3. Validate against field cruise data
    # ------------------------------------------------------------------ #
    def test_user_validates_against_field_data(
        self, full_chain: dict
    ) -> None:
        validation = full_chain["validation"]

        assert 0.0 <= validation["detection_rate"] <= 1.0
        rmse = validation["rmse_height_m"]
        assert rmse is None or (np.isfinite(rmse) and rmse > 0)

    # ------------------------------------------------------------------ #
    # 4. Data quality -- all flags should be 0 on synthetic data
    # ------------------------------------------------------------------ #
    def test_user_checks_data_quality(self, full_chain: dict) -> None:
        metrics = full_chain["metrics"]
        assert (metrics["quality_flag"] == 0).all(), (
            "Expected all quality_flag=0 on synthetic data, "
            f"got flags: {metrics['quality_flag'].unique()}"
        )

    # ------------------------------------------------------------------ #
    # 5. No crown below configured minimum area
    # ------------------------------------------------------------------ #
    def test_user_filters_small_crowns(self, full_chain: dict) -> None:
        crowns = full_chain["crowns"]
        min_area = full_chain["config"]["processing"]["min_crown_area"]

        assert (crowns["crown_area_m2"] >= min_area).all(), (
            f"Found crowns with area < {min_area} m^2"
        )

    # ------------------------------------------------------------------ #
    # 6. Consistency -- two runs produce same results
    # ------------------------------------------------------------------ #
    def test_user_gets_consistent_results(
        self, chm_path: Path, default_config: dict
    ) -> None:
        def _run() -> gpd.GeoDataFrame:
            treetops = detect_treetops(chm_path, default_config)
            crowns = segment_crowns(chm_path, treetops, default_config)
            return extract_tree_metrics(crowns, chm_path, default_config)

        run1 = _run()
        run2 = _run()

        assert len(run1) == len(run2), "Tree counts differ between runs"
        np.testing.assert_array_almost_equal(
            run1["max_height_m"].values,
            run2["max_height_m"].values,
            decimal=4,
        )
