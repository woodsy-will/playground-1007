"""Tests for background point generation and presence/absence matrix construction."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_bounds
from shapely.geometry import Point

from projects.p4_habitat_suitability.src.background import (
    generate_background_points,
)


@pytest.fixture()
def small_stack() -> np.ndarray:
    """A 3-band 20x20 predictor stack with finite values."""
    rng = np.random.default_rng(42)
    return rng.random((3, 20, 20)).astype(np.float32)


@pytest.fixture()
def small_profile() -> dict:
    """Rasterio profile matching the small_stack."""
    return {
        "transform": from_bounds(0, 0, 200, 200, 20, 20),  # 10m cells
        "width": 20,
        "height": 20,
        "crs": "EPSG:3310",
    }


@pytest.fixture()
def sample_occurrences(small_profile: dict) -> gpd.GeoDataFrame:
    """A small set of occurrence points within the raster extent."""
    transform = small_profile["transform"]
    points = [
        Point(
            transform.c + 5.5 * transform.a,
            transform.f + 5.5 * transform.e,
        ),
        Point(
            transform.c + 10.5 * transform.a,
            transform.f + 10.5 * transform.e,
        ),
        Point(
            transform.c + 15.5 * transform.a,
            transform.f + 15.5 * transform.e,
        ),
    ]
    return gpd.GeoDataFrame(
        {"species": ["test"] * 3},
        geometry=points,
        crs="EPSG:3310",
    )


class TestGenerateBackgroundPoints:
    """Tests for generate_background_points()."""

    def test_correct_count(
        self,
        sample_occurrences: gpd.GeoDataFrame,
        small_stack: np.ndarray,
        small_profile: dict,
        default_config: dict,
    ) -> None:
        """Should generate the requested number of background points."""
        n = 50
        result = generate_background_points(
            sample_occurrences, small_stack, small_profile, default_config, n_points=n
        )
        assert len(result) == n

    def test_has_presence_column(
        self,
        sample_occurrences: gpd.GeoDataFrame,
        small_stack: np.ndarray,
        small_profile: dict,
        default_config: dict,
    ) -> None:
        """Output should contain a 'presence' column."""
        result = generate_background_points(
            sample_occurrences, small_stack, small_profile, default_config, n_points=20
        )
        assert "presence" in result.columns

    def test_presence_all_zero(
        self,
        sample_occurrences: gpd.GeoDataFrame,
        small_stack: np.ndarray,
        small_profile: dict,
        default_config: dict,
    ) -> None:
        """All background points should have presence=0."""
        result = generate_background_points(
            sample_occurrences, small_stack, small_profile, default_config, n_points=30
        )
        assert (result["presence"] == 0).all()

    def test_no_valid_pixels_raises(
        self,
        sample_occurrences: gpd.GeoDataFrame,
        small_profile: dict,
        default_config: dict,
    ) -> None:
        """Should raise ValueError when the stack has no valid (finite) pixels."""
        nan_stack = np.full((3, 20, 20), np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="No valid pixels"):
            generate_background_points(
                sample_occurrences, nan_stack, small_profile, default_config, n_points=10
            )


class TestCreatePAMatrix:
    """Tests for create_pa_matrix()."""

    def test_output_shapes(
        self,
        sample_occurrences: gpd.GeoDataFrame,
        small_stack: np.ndarray,
        small_profile: dict,
        default_config: dict,
    ) -> None:
        """X should have shape (n_samples, n_bands) and y should be 1-D."""
        from projects.p4_habitat_suitability.src.background import create_pa_matrix

        bg = generate_background_points(
            sample_occurrences, small_stack, small_profile, default_config, n_points=20
        )
        band_names = ["band_0", "band_1", "band_2"]

        # Add presence column to occurrences for create_pa_matrix
        presence_gdf = sample_occurrences.copy()
        presence_gdf["presence"] = 1

        x_arr, y = create_pa_matrix(presence_gdf, bg, small_stack, small_profile, band_names)

        assert x_arr.ndim == 2
        assert x_arr.shape[1] == len(band_names)
        assert y.ndim == 1
        assert len(y) == x_arr.shape[0]

    def test_labels_binary(
        self,
        sample_occurrences: gpd.GeoDataFrame,
        small_stack: np.ndarray,
        small_profile: dict,
        default_config: dict,
    ) -> None:
        """Response vector y should contain only 0s and 1s."""
        from projects.p4_habitat_suitability.src.background import create_pa_matrix

        bg = generate_background_points(
            sample_occurrences, small_stack, small_profile, default_config, n_points=20
        )
        band_names = ["band_0", "band_1", "band_2"]

        presence_gdf = sample_occurrences.copy()
        presence_gdf["presence"] = 1

        _, y = create_pa_matrix(presence_gdf, bg, small_stack, small_profile, band_names)

        assert set(np.unique(y)).issubset({0, 1})
