"""Tests for model training and spatial block cross-validation."""

from __future__ import annotations

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")


class TestTrainModels:
    """Verify that models train successfully on synthetic data."""

    def test_maxent_trains(self, default_config: dict):
        from projects.p4_habitat_suitability.src.background import (
            create_pa_matrix,
            generate_background_points,
        )
        from projects.p4_habitat_suitability.src.modeling import train_maxent
        from projects.p4_habitat_suitability.src.occurrences import load_occurrences
        from projects.p4_habitat_suitability.src.predictors import build_predictor_stack

        gdf = load_occurrences(default_config)
        stack, profile, band_names = build_predictor_stack(default_config)
        bg = generate_background_points(gdf, stack, profile, default_config, n_points=100)
        X, y = create_pa_matrix(gdf, bg, stack, profile, band_names)  # noqa: N806

        model = train_maxent(X, y, default_config)
        assert hasattr(model, "predict_proba")
        probs = model.predict_proba(model.scaler_.transform(X))
        assert probs.shape == (len(X), 2)

    def test_random_forest_trains(self, default_config: dict):
        from projects.p4_habitat_suitability.src.background import (
            create_pa_matrix,
            generate_background_points,
        )
        from projects.p4_habitat_suitability.src.modeling import train_random_forest
        from projects.p4_habitat_suitability.src.occurrences import load_occurrences
        from projects.p4_habitat_suitability.src.predictors import build_predictor_stack

        gdf = load_occurrences(default_config)
        stack, profile, band_names = build_predictor_stack(default_config)
        bg = generate_background_points(gdf, stack, profile, default_config, n_points=100)
        X, y = create_pa_matrix(gdf, bg, stack, profile, band_names)  # noqa: N806

        model = train_random_forest(X, y, default_config)
        assert hasattr(model, "predict_proba")
        probs = model.predict_proba(X)
        assert probs.shape == (len(X), 2)


class TestSpatialBlockCV:
    """Verify cross-validation returns AUC and TSS metrics."""

    def test_cv_returns_auc_and_tss(self, default_config: dict):
        from projects.p4_habitat_suitability.src.background import (
            create_pa_matrix,
            generate_background_points,
        )
        from projects.p4_habitat_suitability.src.modeling import (
            spatial_block_cv,
            train_random_forest,
        )
        from projects.p4_habitat_suitability.src.occurrences import load_occurrences
        from projects.p4_habitat_suitability.src.predictors import (
            build_predictor_stack,
            extract_values_at_points,
        )

        gdf = load_occurrences(default_config)
        stack, profile, band_names = build_predictor_stack(default_config)
        bg = generate_background_points(gdf, stack, profile, default_config, n_points=200)
        X, y = create_pa_matrix(gdf, bg, stack, profile, band_names)  # noqa: N806

        # Build coords array
        pres_df = extract_values_at_points(stack, profile, gdf, band_names)
        bg_df = extract_values_at_points(stack, profile, bg, band_names)
        coords = np.vstack([
            pres_df[["x", "y"]].values,
            bg_df[["x", "y"]].values,
        ])[:len(X)]

        result = spatial_block_cv(
            X, y, coords,
            model_fn=lambda Xt, yt: train_random_forest(Xt, yt),  # noqa: N803
            config=default_config,
        )

        assert "auc_mean" in result
        assert "tss_mean" in result
        # AUC should be between 0 and 1 (or NaN if insufficient data)
        if not np.isnan(result["auc_mean"]):
            assert 0.0 <= result["auc_mean"] <= 1.0


class TestPipelineAlgorithmDispatch:
    """Verify the pipeline rejects unknown algorithm names."""

    def test_unknown_algorithm_raises(self, default_config: dict, tmp_path):

        import yaml

        from projects.p4_habitat_suitability.src.pipeline import run_pipeline

        # Write the test config to a temp YAML file with an invalid algo
        cfg = dict(default_config)
        cfg["modeling"] = dict(cfg["modeling"])
        cfg["modeling"]["algorithms"] = ["bogus_model"]

        tmp_cfg = tmp_path / "bad_algo.yaml"
        tmp_cfg.write_text(yaml.dump(cfg))

        with pytest.raises(ValueError, match="Unknown algorithm"):
            run_pipeline(str(tmp_cfg))
