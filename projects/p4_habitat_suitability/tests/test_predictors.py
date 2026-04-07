"""Tests for predictor stack construction and value extraction."""

from __future__ import annotations

import numpy as np
from pyproj import CRS


class TestBuildPredictorStack:
    """Verify predictor stack alignment, shape, and CRS."""

    def test_stack_shape(self, default_config: dict):
        from projects.p4_habitat_suitability.src.predictors import build_predictor_stack

        stack, profile, band_names = build_predictor_stack(default_config)
        n_bands, h, w = stack.shape
        assert n_bands == len(band_names)
        assert h == profile["height"]
        assert w == profile["width"]
        assert n_bands > 0

    def test_stack_crs(self, default_config: dict):
        from projects.p4_habitat_suitability.src.predictors import build_predictor_stack

        _, profile, _ = build_predictor_stack(default_config)
        assert CRS(profile["crs"]) == CRS.from_epsg(3310)

    def test_stack_finite_values(self, default_config: dict):
        from projects.p4_habitat_suitability.src.predictors import build_predictor_stack

        stack, _, _ = build_predictor_stack(default_config)
        assert np.all(np.isfinite(stack)), "Stack contains non-finite values"

    def test_band_names_match_files(self, default_config: dict):
        from projects.p4_habitat_suitability.src.predictors import build_predictor_stack

        _, _, band_names = build_predictor_stack(default_config)
        # Synthetic data generates these predictors
        expected = {"bio1_mean_temp", "bio12_annual_precip", "canopy_cover",
                    "elevation", "slope", "tpi"}
        assert set(band_names) == expected


class TestExtractValuesAtPoints:
    """Verify raster value extraction at point locations."""

    def test_extraction_returns_all_bands(self, default_config: dict):
        from projects.p4_habitat_suitability.src.occurrences import load_occurrences
        from projects.p4_habitat_suitability.src.predictors import (
            build_predictor_stack,
            extract_values_at_points,
        )

        gdf = load_occurrences(default_config)
        stack, profile, band_names = build_predictor_stack(default_config)
        df = extract_values_at_points(stack, profile, gdf, band_names)

        assert len(df) > 0
        for name in band_names:
            assert name in df.columns

    def test_extraction_no_nans(self, default_config: dict):
        from projects.p4_habitat_suitability.src.occurrences import load_occurrences
        from projects.p4_habitat_suitability.src.predictors import (
            build_predictor_stack,
            extract_values_at_points,
        )

        gdf = load_occurrences(default_config)
        stack, profile, band_names = build_predictor_stack(default_config)
        df = extract_values_at_points(stack, profile, gdf, band_names)

        assert not df.isna().any().any(), "Extracted values contain NaN"
