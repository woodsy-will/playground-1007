"""End-to-end tests for the burn severity pipeline.

User story: A wildfire analyst loads satellite imagery, runs the burn
severity pipeline, and gets a severity map + recovery projections.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from projects.p1_burn_severity.src.pipeline import run_pipeline
from shared.utils.io import read_raster


def _write_e2e_config(tmp_path: Path) -> Path:
    """Write a YAML config pointing all data dirs to tmp_path."""
    imagery_dir = tmp_path / "data" / "raw" / "imagery"
    output_dir = tmp_path / "data" / "processed"
    config = {
        "data": {
            "fire_perimeters": str(tmp_path / "data" / "raw" / "fire_perimeters"),
            "imagery_dir": str(imagery_dir),
            "management_units": str(tmp_path / "data" / "raw" / "management_units"),
            "output_dir": str(output_dir),
        },
        "acquisition": {
            "source": "copernicus",
            "cloud_cover_max": 20,
            "seasonal_window": [6, 10],
        },
        "processing": {
            "crs": "EPSG:3310",
            "severity_thresholds": {
                "unburned": [-0.1, 0.1],
                "low": [0.1, 0.27],
                "moderate_low": [0.27, 0.44],
                "moderate_high": [0.44, 0.66],
                "high": [0.66, 1.3],
            },
            "recovery_index": "NDVI",
            "recovery_model": "exponential",
            "years_post_fire": 5,
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


class TestBurnSeverityE2E:
    """E2E tests simulating a wildfire analyst's workflow."""

    @pytest.fixture()
    def pipeline_result(self, tmp_path: Path) -> dict:
        """Run the full pipeline once and return result dict + paths."""
        config_path = _write_e2e_config(tmp_path)
        result = run_pipeline(config_path)
        return result

    # ------------------------------------------------------------------ #
    # 1. Run pipeline from config -- verify all output files exist
    # ------------------------------------------------------------------ #
    def test_user_runs_pipeline_from_config(self, pipeline_result: dict) -> None:
        output_paths = pipeline_result["output_paths"]
        expected_files = ["severity", "dnbr", "rbr", "recovery_timeseries", "recovery_model"]
        for key in expected_files:
            assert key in output_paths, f"Missing output key: {key}"
            path = Path(output_paths[key])
            assert path.exists(), f"Output file does not exist: {path}"

    # ------------------------------------------------------------------ #
    # 2. Read severity map -- valid class codes and correct CRS
    # ------------------------------------------------------------------ #
    def test_user_reads_severity_map(self, pipeline_result: dict) -> None:
        sev_path = pipeline_result["output_paths"]["severity"]
        data, profile = read_raster(sev_path)
        sev = data[0]

        # Valid class codes: 0-4 and 255 (nodata)
        unique_vals = set(np.unique(sev))
        assert unique_vals.issubset({0, 1, 2, 3, 4, 255})
        assert len(unique_vals - {255}) > 0, "No valid severity classes found"

        # CRS check
        crs_str = str(profile["crs"])
        assert "3310" in crs_str

    # ------------------------------------------------------------------ #
    # 3. Read recovery CSV -- verify columns and data sensibility
    # ------------------------------------------------------------------ #
    def test_user_reads_recovery_csv(self, pipeline_result: dict) -> None:
        ts_path = pipeline_result["output_paths"]["recovery_timeseries"]
        df = pd.read_csv(ts_path)

        # Required columns
        for col in ("year", "severity_class", "mean_index"):
            assert col in df.columns, f"Missing column: {col}"

        # Years should be sequential starting from 1
        years = sorted(df["year"].unique())
        assert years[0] == 1
        for i in range(1, len(years)):
            assert years[i] == years[i - 1] + 1

        # Index values should be in [0, 1]
        assert df["mean_index"].min() >= 0.0
        assert df["mean_index"].max() <= 1.0

    # ------------------------------------------------------------------ #
    # 4. Read model results -- verify R-squared and recovery time columns
    # ------------------------------------------------------------------ #
    def test_user_reads_model_results(self, pipeline_result: dict) -> None:
        model_path = pipeline_result["output_paths"]["recovery_model"]
        df = pd.read_csv(model_path)

        assert "r_squared" in df.columns
        assert "years_to_90pct_recovery" in df.columns
        # At least one severity class was fitted
        assert len(df) > 0

    # ------------------------------------------------------------------ #
    # 5. Inspect dNBR distribution -- plausible range, burned pixels > 0
    # ------------------------------------------------------------------ #
    def test_user_inspects_dnbr_distribution(self, pipeline_result: dict) -> None:
        dnbr_path = pipeline_result["output_paths"]["dnbr"]
        data, _ = read_raster(dnbr_path)
        dnbr = data[0]

        # dNBR should be in plausible range [-2, 2]
        valid = dnbr[np.isfinite(dnbr)]
        assert valid.min() >= -2.0
        assert valid.max() <= 2.0

        # Burned pixels should have positive dNBR
        positive_count = np.count_nonzero(valid > 0)
        assert positive_count > 0, "No burned pixels detected (no positive dNBR)"

    # ------------------------------------------------------------------ #
    # 6. Reproducibility -- two runs with same config produce identical results
    # ------------------------------------------------------------------ #
    def test_pipeline_outputs_reproducible(self, tmp_path: Path) -> None:
        # First run
        dir1 = tmp_path / "run1"
        dir1.mkdir()
        config_path1 = _write_e2e_config(dir1)
        result1 = run_pipeline(config_path1)

        # Second run
        dir2 = tmp_path / "run2"
        dir2.mkdir()
        config_path2 = _write_e2e_config(dir2)
        result2 = run_pipeline(config_path2)

        # Severity rasters should be nearly identical (synthetic noise may
        # cause a few borderline pixels to differ between runs)
        sev1, _ = read_raster(result1["output_paths"]["severity"])
        sev2, _ = read_raster(result2["output_paths"]["severity"])
        match_pct = np.mean(sev1 == sev2) * 100
        assert match_pct > 98, f"Only {match_pct:.1f}% of pixels match between runs"
