"""Smoke tests for P4 Habitat Suitability.

Quick sanity checks that core modules import, key functions are callable,
and basic operations produce expected types. Designed to run in < 1 second.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.transform import from_bounds


class TestP4Smoke:
    """Fast smoke tests for P4 habitat suitability modeling."""

    def test_imports(self):
        """All core P4 modules should import without error."""
        from projects.p4_habitat_suitability.src import (
            background,  # noqa: F401
            change_analysis,  # noqa: F401
            occurrences,  # noqa: F401
            predictors,  # noqa: F401
            projection,  # noqa: F401
        )

    def test_load_occurrences_runs(self, default_config: dict):
        """load_occurrences should return a GeoDataFrame."""
        from projects.p4_habitat_suitability.src.occurrences import load_occurrences

        result = load_occurrences(default_config)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0

    def test_thin_occurrences_runs(self, default_config: dict):
        """thin_occurrences should return fewer or equal points."""
        from projects.p4_habitat_suitability.src.occurrences import (
            load_occurrences,
            thin_occurrences,
        )

        occ = load_occurrences(default_config)
        thinned = thin_occurrences(occ, distance_km=0.5, config=default_config)
        assert isinstance(thinned, gpd.GeoDataFrame)
        assert len(thinned) <= len(occ)

    def test_build_predictor_stack_runs(self, default_config: dict):
        """build_predictor_stack should return a 3D array + profile + names."""
        from projects.p4_habitat_suitability.src.predictors import (
            build_predictor_stack,
        )

        stack, profile, band_names = build_predictor_stack(default_config)
        assert stack.ndim == 3
        assert len(band_names) == stack.shape[0]
        assert "crs" in profile

    def test_compute_change_runs(self):
        """compute_change should classify pixels into 4 classes."""
        from projects.p4_habitat_suitability.src.change_analysis import compute_change

        current = np.array([[0.8, 0.2], [0.6, 0.1]])
        future = np.array([[0.7, 0.6], [0.3, 0.1]])
        result = compute_change(current, future, threshold=0.5)
        assert result.dtype == np.uint8
        assert result.shape == (2, 2)

    def test_summarize_change_runs(self):
        """summarize_change should return a DataFrame with area stats."""
        from projects.p4_habitat_suitability.src.change_analysis import (
            compute_change,
            summarize_change,
        )

        change = compute_change(
            np.random.default_rng(1).random((10, 10)),
            np.random.default_rng(2).random((10, 10)),
        )
        transform = from_bounds(0, 0, 300, 300, 10, 10)
        profile = {"transform": transform}
        result = summarize_change(change, profile)
        assert isinstance(result, pd.DataFrame)
        assert "area_ha" in result.columns

    def test_generate_background_points_runs(self, default_config: dict):
        """generate_background_points should return points with presence=0."""
        from projects.p4_habitat_suitability.src.background import (
            generate_background_points,
        )
        from projects.p4_habitat_suitability.src.occurrences import (
            load_occurrences,
            thin_occurrences,
        )
        from projects.p4_habitat_suitability.src.predictors import (
            build_predictor_stack,
        )

        occ = load_occurrences(default_config)
        thinned = thin_occurrences(occ, 0.5, default_config)
        stack, profile, _ = build_predictor_stack(default_config)
        bg = generate_background_points(thinned, stack, profile, default_config, n_points=50)
        assert isinstance(bg, gpd.GeoDataFrame)
        assert "presence" in bg.columns
        assert (bg["presence"] == 0).all()

    def test_threshold_suitability_runs(self):
        """threshold_suitability should produce a binary map."""
        from projects.p4_habitat_suitability.src.projection import (
            threshold_suitability,
        )

        suit = np.array([[0.3, 0.7], [0.5, 0.9]])
        binary = threshold_suitability(suit, threshold=0.5)
        assert set(np.unique(binary[~np.isnan(binary)])).issubset({0, 1})
