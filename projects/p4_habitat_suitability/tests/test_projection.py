"""Tests for suitability projection and thresholding."""

from __future__ import annotations

import numpy as np
import pytest


class TestProjectSuitability:
    """Verify suitability projection produces valid raster output."""

    def test_projection_shape_matches_stack(self, default_config: dict):
        pytest.importorskip("sklearn")
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
        pytest.importorskip("sklearn")
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


class TestEnsembleProject:
    """Verify AUC-weighted ensemble projection."""

    def _make_model_and_metrics(self, default_config):
        """Helper: train two models and get CV metrics."""
        pytest.importorskip("sklearn")
        from projects.p4_habitat_suitability.src.background import (
            create_pa_matrix,
            generate_background_points,
        )
        from projects.p4_habitat_suitability.src.modeling import (
            train_maxent,
            train_random_forest,
        )
        from projects.p4_habitat_suitability.src.occurrences import load_occurrences
        from projects.p4_habitat_suitability.src.predictors import build_predictor_stack

        gdf = load_occurrences(default_config)
        stack, profile, band_names = build_predictor_stack(default_config)
        bg = generate_background_points(
            gdf, stack, profile, default_config, n_points=100,
        )
        X, y = create_pa_matrix(gdf, bg, stack, profile, band_names)  # noqa: N806

        models = {
            "maxent": train_maxent(X, y, default_config),
            "random_forest": train_random_forest(X, y, default_config),
        }
        cv_metrics = {
            "maxent": {"auc_mean": 0.75},
            "random_forest": {"auc_mean": 0.85},
        }
        return models, cv_metrics, stack, profile

    def test_ensemble_shape_matches_stack(self, default_config: dict):
        pytest.importorskip("sklearn")
        from projects.p4_habitat_suitability.src.projection import ensemble_project

        models, cv_metrics, stack, profile = self._make_model_and_metrics(
            default_config,
        )
        ensemble, uncertainty, out_profile, weights = ensemble_project(
            models, cv_metrics, stack, profile,
        )

        assert ensemble.shape == (profile["height"], profile["width"])
        assert uncertainty.shape == ensemble.shape
        assert out_profile["count"] == 1

    def test_ensemble_values_in_0_1(self, default_config: dict):
        pytest.importorskip("sklearn")
        from projects.p4_habitat_suitability.src.projection import ensemble_project

        models, cv_metrics, stack, profile = self._make_model_and_metrics(
            default_config,
        )
        ensemble, _, _, _ = ensemble_project(
            models, cv_metrics, stack, profile,
        )
        valid = ensemble[np.isfinite(ensemble)]
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0

    def test_weights_sum_to_one(self, default_config: dict):
        pytest.importorskip("sklearn")
        from projects.p4_habitat_suitability.src.projection import ensemble_project

        models, cv_metrics, stack, profile = self._make_model_and_metrics(
            default_config,
        )
        _, _, _, weights = ensemble_project(
            models, cv_metrics, stack, profile,
        )
        assert len(weights) == 2
        np.testing.assert_allclose(sum(weights.values()), 1.0, atol=1e-10)

    def test_weights_reflect_auc(self, default_config: dict):
        """Model with higher AUC should get higher weight."""
        pytest.importorskip("sklearn")
        from projects.p4_habitat_suitability.src.projection import ensemble_project

        models, cv_metrics, stack, profile = self._make_model_and_metrics(
            default_config,
        )
        _, _, _, weights = ensemble_project(
            models, cv_metrics, stack, profile,
        )
        assert weights["random_forest"] > weights["maxent"]

    def test_single_model_degenerates(self, default_config: dict):
        """Ensemble with one model equals that model's projection."""
        pytest.importorskip("sklearn")
        from projects.p4_habitat_suitability.src.projection import (
            ensemble_project,
            project_suitability,
        )

        models, cv_metrics, stack, profile = self._make_model_and_metrics(
            default_config,
        )
        single = {"random_forest": models["random_forest"]}
        single_cv = {"random_forest": cv_metrics["random_forest"]}

        ensemble, _, _, weights = ensemble_project(
            single, single_cv, stack, profile,
        )
        direct, _ = project_suitability(
            models["random_forest"], stack, profile,
        )

        # Single model gets weight 1.0
        assert weights["random_forest"] == 1.0
        # Results should be identical
        valid = np.isfinite(ensemble) & np.isfinite(direct)
        np.testing.assert_allclose(ensemble[valid], direct[valid], atol=1e-6)

    def test_uncertainty_non_negative(self, default_config: dict):
        pytest.importorskip("sklearn")
        from projects.p4_habitat_suitability.src.projection import ensemble_project

        models, cv_metrics, stack, profile = self._make_model_and_metrics(
            default_config,
        )
        _, uncertainty, _, _ = ensemble_project(
            models, cv_metrics, stack, profile,
        )
        valid = uncertainty[np.isfinite(uncertainty)]
        assert np.all(valid >= 0.0)

    def test_empty_models_raises(self):
        import pytest

        from projects.p4_habitat_suitability.src.projection import ensemble_project

        with pytest.raises(ValueError, match="At least one model"):
            ensemble_project({}, {}, np.zeros((1, 2, 2)), {})

    def test_nan_auc_gets_default_weight(self, default_config: dict):
        """Model with NaN AUC should receive weight 0.5."""
        pytest.importorskip("sklearn")
        from projects.p4_habitat_suitability.src.projection import ensemble_project

        models, _, stack, profile = self._make_model_and_metrics(
            default_config,
        )
        cv_metrics = {
            "maxent": {"auc_mean": np.nan},
            "random_forest": {"auc_mean": 0.5},
        }
        _, _, _, weights = ensemble_project(
            models, cv_metrics, stack, profile,
        )
        # Both should be 0.5 raw \u2192 0.5 each normalised
        np.testing.assert_allclose(weights["maxent"], 0.5, atol=1e-10)
        np.testing.assert_allclose(weights["random_forest"], 0.5, atol=1e-10)


# -----------------------------------------------------------------------
# Mock-based tests that do NOT require sklearn
# -----------------------------------------------------------------------


class TestProjectSuitabilityMock:
    """Cover project_suitability using mock models (no sklearn needed)."""

    def _make_mock_model(self, n_classes: int = 2):
        from unittest.mock import MagicMock

        model = MagicMock()
        model.predict_proba = MagicMock(
            side_effect=lambda X: np.column_stack(  # noqa: N803
                [np.full(len(X), 0.3), np.full(len(X), 0.7)]
            )
        )
        del model.scaler_
        return model

    def test_project_suitability_with_mock_model(self):
        from projects.p4_habitat_suitability.src.projection import project_suitability

        model = self._make_mock_model()
        stack = np.random.default_rng(0).random((3, 4, 5)).astype(np.float32)
        profile = {"height": 4, "width": 5, "count": 3, "dtype": "float32"}

        suitability, out_profile = project_suitability(model, stack, profile)

        assert suitability.shape == (4, 5)
        assert out_profile["count"] == 1
        valid = suitability[np.isfinite(suitability)]
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0
        np.testing.assert_allclose(valid, 0.7, atol=1e-6)

    def test_project_suitability_with_scaler(self):
        from unittest.mock import MagicMock

        from projects.p4_habitat_suitability.src.projection import project_suitability

        model = self._make_mock_model()
        scaler = MagicMock()
        scaler.transform = MagicMock(side_effect=lambda x: x)
        model.scaler_ = scaler

        stack = np.random.default_rng(1).random((2, 3, 3)).astype(np.float32)
        profile = {"height": 3, "width": 3, "count": 2, "dtype": "float32"}

        suitability, _ = project_suitability(model, stack, profile)

        assert suitability.shape == (3, 3)
        scaler.transform.assert_called_once()

    def test_project_suitability_handles_nan_pixels(self):
        from projects.p4_habitat_suitability.src.projection import project_suitability

        model = self._make_mock_model()
        stack = np.ones((2, 2, 2), dtype=np.float32)
        stack[0, 0, 0] = np.nan

        profile = {"height": 2, "width": 2, "count": 2, "dtype": "float32"}
        suitability, _ = project_suitability(model, stack, profile)

        assert np.isnan(suitability[0, 0])
        assert np.isfinite(suitability[0, 1])


class TestEnsembleProjectMock:
    """Cover ensemble_project using mock models (no sklearn needed)."""

    def _make_mock_model(self, prob: float = 0.7):
        from unittest.mock import MagicMock

        model = MagicMock()
        model.predict_proba = MagicMock(
            side_effect=lambda X: np.column_stack(  # noqa: N803
                [np.full(len(X), 1.0 - prob), np.full(len(X), prob)]
            )
        )
        del model.scaler_
        return model

    def test_ensemble_project_with_mock_models(self):
        from projects.p4_habitat_suitability.src.projection import ensemble_project

        models = {
            "model_a": self._make_mock_model(prob=0.6),
            "model_b": self._make_mock_model(prob=0.8),
        }
        cv_metrics = {
            "model_a": {"auc_mean": 0.7},
            "model_b": {"auc_mean": 0.9},
        }
        stack = np.random.default_rng(0).random((2, 3, 3)).astype(np.float32)
        profile = {"height": 3, "width": 3, "count": 2, "dtype": "float32"}

        ensemble, uncertainty, out_profile, weights = ensemble_project(
            models, cv_metrics, stack, profile,
        )

        assert ensemble.shape == (3, 3)
        assert uncertainty.shape == (3, 3)
        assert out_profile["count"] == 1
        np.testing.assert_allclose(sum(weights.values()), 1.0, atol=1e-10)
        assert weights["model_b"] > weights["model_a"]

    def test_ensemble_project_single_model(self):
        from unittest.mock import MagicMock

        from projects.p4_habitat_suitability.src.projection import (
            ensemble_project,
            project_suitability,
        )

        model = MagicMock()
        model.predict_proba = MagicMock(
            side_effect=lambda X: np.column_stack(  # noqa: N803
                [np.full(len(X), 0.35), np.full(len(X), 0.65)]
            )
        )
        del model.scaler_

        models = {"only": model}
        cv_metrics = {"only": {"auc_mean": 0.8}}
        stack = np.random.default_rng(2).random((2, 2, 2)).astype(np.float32)
        profile = {"height": 2, "width": 2, "count": 2, "dtype": "float32"}

        ensemble, _, _, weights = ensemble_project(
            models, cv_metrics, stack, profile,
        )

        assert weights["only"] == 1.0
        direct, _ = project_suitability(model, stack, profile)
        valid = np.isfinite(ensemble) & np.isfinite(direct)
        np.testing.assert_allclose(ensemble[valid], direct[valid], atol=1e-6)

    def test_ensemble_project_nan_auc(self):
        from projects.p4_habitat_suitability.src.projection import ensemble_project

        models = {
            "a": self._make_mock_model(prob=0.5),
            "b": self._make_mock_model(prob=0.5),
        }
        cv_metrics = {
            "a": {"auc_mean": np.nan},
            "b": {"auc_mean": 0.5},
        }
        stack = np.random.default_rng(3).random((2, 2, 2)).astype(np.float32)
        profile = {"height": 2, "width": 2, "count": 2, "dtype": "float32"}

        _, _, _, weights = ensemble_project(
            models, cv_metrics, stack, profile,
        )

        np.testing.assert_allclose(weights["a"], 0.5, atol=1e-10)
        np.testing.assert_allclose(weights["b"], 0.5, atol=1e-10)
