"""Tests for suitability projection and thresholding."""

from __future__ import annotations

import numpy as np


class TestProjectSuitability:
    """Verify suitability projection produces valid raster output."""

    def test_projection_shape_matches_stack(self, default_config: dict):
        from projects.p4_habitat_suitability.src.background import (
            create_pa_matrix,
            generate_background_points,
        )
        from projects.p4_habitat_suitability.src.modeling import train_random_forest
        from projects.p4_habitat_suitability.src.occurrences import load_occurrences
        from projects.p4_habitat_suitability.src.predictors import build_predictor_stack
        from projects.p4_habitat_suitability.src.projection import project_suitability

        gdf = load_occurrences(default_config)
        stack, profile, band_names = build_predictor_stack(default_config)
        bg = generate_background_points(gdf, stack, profile, default_config, n_points=100)
        X, y = create_pa_matrix(gdf, bg, stack, profile, band_names)  # noqa: N806
        model = train_random_forest(X, y, default_config)

        suitability, out_profile = project_suitability(model, stack, profile)

        assert suitability.shape == (profile["height"], profile["width"])
        assert out_profile["count"] == 1

    def test_projection_values_in_0_1(self, default_config: dict):
        from projects.p4_habitat_suitability.src.background import (
            create_pa_matrix,
            generate_background_points,
        )
        from projects.p4_habitat_suitability.src.modeling import train_random_forest
        from projects.p4_habitat_suitability.src.occurrences import load_occurrences
        from projects.p4_habitat_suitability.src.predictors import build_predictor_stack
        from projects.p4_habitat_suitability.src.projection import project_suitability

        gdf = load_occurrences(default_config)
        stack, profile, band_names = build_predictor_stack(default_config)
        bg = generate_background_points(gdf, stack, profile, default_config, n_points=100)
        X, y = create_pa_matrix(gdf, bg, stack, profile, band_names)  # noqa: N806
        model = train_random_forest(X, y, default_config)

        suitability, _ = project_suitability(model, stack, profile)
        valid = suitability[np.isfinite(suitability)]

        assert valid.min() >= 0.0, "Suitability contains values < 0"
        assert valid.max() <= 1.0, "Suitability contains values > 1"


class TestThresholdSuitability:
    """Verify binary thresholding produces correct output."""

    def test_threshold_produces_binary(self):
        from projects.p4_habitat_suitability.src.projection import threshold_suitability

        suit = np.array([[0.1, 0.6], [0.9, 0.3]], dtype=np.float32)
        binary = threshold_suitability(suit, threshold=0.5)

        assert binary.dtype == np.uint8
        assert set(np.unique(binary)).issubset({0, 1})
        expected = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(binary, expected)

    def test_threshold_handles_nan(self):
        from projects.p4_habitat_suitability.src.projection import threshold_suitability

        suit = np.array([[np.nan, 0.8], [0.2, np.nan]], dtype=np.float32)
        binary = threshold_suitability(suit, threshold=0.5)

        assert binary[0, 0] == 0  # NaN -> 0
        assert binary[0, 1] == 1
        assert binary[1, 0] == 0
        assert binary[1, 1] == 0  # NaN -> 0
