"""Tests for validation module."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point


class TestValidateAgainstCruise:
    """Verify nearest-neighbour matching and metric computation."""

    @pytest.fixture()
    def predicted_gdf(self) -> gpd.GeoDataFrame:
        """Simple predicted tree inventory for testing."""
        return gpd.GeoDataFrame(
            {
                "tree_id": [1, 2, 3],
                "max_height_m": [20.0, 25.0, 18.0],
                "dbh_inches": [15.0, 20.0, 12.0],
            },
            geometry=[
                Point(-199_966.0, -49_900.0),
                Point(-199_933.0, -49_900.0),
                Point(-199_900.0, -49_900.0),
            ],
            crs="EPSG:3310",
        )

    @pytest.fixture()
    def cruise_csv_fixture(self, tmp_path: Path) -> Path:
        """Write a small cruise CSV that partially overlaps predictions."""
        df = pd.DataFrame(
            {
                "stem_x": [-199_966.5, -199_933.5, -199_870.0],
                "stem_y": [-49_900.5, -49_900.5, -49_900.0],
                "dbh_inches": [14.0, 19.0, 22.0],
                "height_ft": [65.6, 82.0, 59.0],
                "species": ["ABCO", "PIPO", "CADE"],
                "diameter_class": ["medium", "medium", "large"],
            }
        )
        path = tmp_path / "test_cruise.csv"
        df.to_csv(path, index=False)
        return path

    def test_returns_dict(
        self,
        predicted_gdf: gpd.GeoDataFrame,
        cruise_csv_fixture: Path,
        default_config: dict,
    ):
        """validate_against_cruise must return a dict."""
        from projects.p3_itc_delineation.src.validation import validate_against_cruise

        result = validate_against_cruise(predicted_gdf, cruise_csv_fixture, default_config)
        assert isinstance(result, dict)

    def test_has_required_keys(
        self,
        predicted_gdf: gpd.GeoDataFrame,
        cruise_csv_fixture: Path,
        default_config: dict,
    ):
        """Result dict must contain standard metric keys."""
        from projects.p3_itc_delineation.src.validation import validate_against_cruise

        result = validate_against_cruise(predicted_gdf, cruise_csv_fixture, default_config)
        for key in (
            "n_predicted",
            "n_reference",
            "n_matched",
            "detection_rate",
            "omission_rate",
            "commission_rate",
        ):
            assert key in result, f"Missing key: {key}"

    def test_detection_rate_range(
        self,
        predicted_gdf: gpd.GeoDataFrame,
        cruise_csv_fixture: Path,
        default_config: dict,
    ):
        """Detection rate must be between 0 and 1."""
        from projects.p3_itc_delineation.src.validation import validate_against_cruise

        result = validate_against_cruise(predicted_gdf, cruise_csv_fixture, default_config)
        assert 0.0 <= result["detection_rate"] <= 1.0

    def test_matching_within_distance(
        self,
        predicted_gdf: gpd.GeoDataFrame,
        cruise_csv_fixture: Path,
        default_config: dict,
    ):
        """Two of three predictions are within 3 m of a cruise stem."""
        from projects.p3_itc_delineation.src.validation import validate_against_cruise

        result = validate_against_cruise(predicted_gdf, cruise_csv_fixture, default_config)
        assert result["n_matched"] == 2

    def test_empty_predictions(
        self,
        cruise_csv_fixture: Path,
        default_config: dict,
    ):
        """Validation with zero predictions should return detection_rate=0."""
        from projects.p3_itc_delineation.src.validation import validate_against_cruise

        empty = gpd.GeoDataFrame(
            {"tree_id": [], "max_height_m": [], "dbh_inches": []},
            geometry=[],
            crs="EPSG:3310",
        )
        result = validate_against_cruise(empty, cruise_csv_fixture, default_config)
        assert result["detection_rate"] == 0.0
        assert result["n_matched"] == 0

    def test_stratify_by_missing_column(
        self,
        predicted_gdf: gpd.GeoDataFrame,
        cruise_csv_fixture: Path,
    ):
        """Stratifying by a column that doesn't exist in cruise data
        should skip without error (covers line 121)."""
        from projects.p3_itc_delineation.src.validation import validate_against_cruise

        config = {
            "validation": {
                "match_distance": 3.0,
                "stratify_by": ["nonexistent_column"],
            },
        }
        result = validate_against_cruise(predicted_gdf, cruise_csv_fixture, config)
        assert isinstance(result, dict)
        # The missing column should not appear in by_stratum or have empty metrics
        assert result["by_stratum"].get("nonexistent_column") is None

    def test_stratify_empty_stratum(
        self,
        predicted_gdf: gpd.GeoDataFrame,
        tmp_path: Path,
    ):
        """A stratum value with no cruise data after filtering should be
        skipped (covers line 127)."""
        import numpy as np

        from projects.p3_itc_delineation.src.validation import validate_against_cruise

        # Create cruise CSV with a stratum column that has a value with
        # stems but also NaN values (dropna removes them, so the loop skips)
        df = pd.DataFrame(
            {
                "stem_x": [-199_966.5, -199_933.5],
                "stem_y": [-49_900.5, -49_900.5],
                "dbh_inches": [14.0, 19.0],
                "height_ft": [65.6, 82.0],
                "species": ["ABCO", "PIPO"],
                "size_class": ["small", np.nan],
            }
        )
        csv_path = tmp_path / "cruise_stratum.csv"
        df.to_csv(csv_path, index=False)

        config = {
            "validation": {
                "match_distance": 3.0,
                "stratify_by": ["size_class"],
            },
        }
        result = validate_against_cruise(predicted_gdf, csv_path, config)
        assert isinstance(result, dict)
        # Only "small" should appear; NaN is dropped by dropna()
        stratum_metrics = result["by_stratum"].get("size_class", {})
        assert "small" in stratum_metrics
