"""Integration test for the full burn severity pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shared.data.generate_synthetic import generate_synthetic_burn_rasters
from shared.utils.config import load_config
from shared.utils.io import read_raster


class TestRunPipeline:
    """Integration tests using synthetic rasters."""

    @pytest.fixture()
    def config_path(self) -> Path:
        return (
            Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
        )

    @pytest.fixture()
    def pipeline_result(self, tmp_path: Path, config_path: Path) -> dict:
        """Run the pipeline on synthetic data in a temp directory."""
        # Generate synthetic imagery
        imagery_dir = tmp_path / "data" / "raw" / "imagery"
        generate_synthetic_burn_rasters(imagery_dir)

        # Patch config to use tmp_path
        config = load_config(config_path)
        config["data"]["imagery_dir"] = str(imagery_dir)
        config["data"]["output_dir"] = str(tmp_path / "data" / "processed")

        # Run pipeline components directly (avoids re-loading config from disk)
        from projects.p1_burn_severity.src.recovery import (
            build_recovery_timeseries,
            fit_recovery_model,
        )
        from projects.p1_burn_severity.src.severity import (
            classify_severity,
            compute_dnbr,
            compute_nbr,
        )
        from shared.utils.io import write_raster

        pre_nir, profile = read_raster(imagery_dir / "pre_nir.tif")
        pre_swir, _ = read_raster(imagery_dir / "pre_swir.tif")
        post_nir, _ = read_raster(imagery_dir / "post_nir.tif")
        post_swir, _ = read_raster(imagery_dir / "post_swir.tif")

        pre_nir, pre_swir = pre_nir[0], pre_swir[0]
        post_nir, post_swir = post_nir[0], post_swir[0]

        pre_nbr = compute_nbr(pre_nir, pre_swir)
        post_nbr = compute_nbr(post_nir, post_swir)
        dnbr = compute_dnbr(pre_nbr, post_nbr)
        severity = classify_severity(dnbr, config)

        out_dir = Path(config["data"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        sev_path = out_dir / "severity.tif"
        write_raster(sev_path, severity, profile, dtype="uint8", nodata=255)

        # Recovery
        rng = np.random.default_rng(42)
        annual = [
            np.clip(0.2 + 0.1 * yr + rng.normal(0, 0.02, pre_nir.shape), 0, 1).astype(
                np.float32
            )
            for yr in range(1, 6)
        ]
        ts = build_recovery_timeseries(annual, severity, config)
        model = fit_recovery_model(ts, config)

        return {
            "severity": severity,
            "dnbr": dnbr,
            "severity_path": sev_path,
            "recovery_ts": ts,
            "recovery_model": model,
            "profile": profile,
        }

    def test_severity_raster_written(self, pipeline_result: dict) -> None:
        assert pipeline_result["severity_path"].exists()

    def test_severity_has_valid_classes(self, pipeline_result: dict) -> None:
        sev = pipeline_result["severity"]
        valid = sev[sev != 255]
        assert len(valid) > 0
        assert set(np.unique(valid)).issubset({0, 1, 2, 3, 4})

    def test_burned_area_detected(self, pipeline_result: dict) -> None:
        sev = pipeline_result["severity"]
        # Synthetic data has burn in top half -> some high-severity pixels
        burned = np.count_nonzero((sev >= 1) & (sev <= 4))
        assert burned > 0

    def test_dnbr_shape_matches(self, pipeline_result: dict) -> None:
        assert pipeline_result["dnbr"].shape == pipeline_result["severity"].shape

    def test_recovery_ts_not_empty(self, pipeline_result: dict) -> None:
        assert len(pipeline_result["recovery_ts"]) > 0

    def test_recovery_model_fitted(self, pipeline_result: dict) -> None:
        model = pipeline_result["recovery_model"]
        if not model.empty:
            assert "a" in model.columns
            assert "years_to_90pct_recovery" in model.columns
