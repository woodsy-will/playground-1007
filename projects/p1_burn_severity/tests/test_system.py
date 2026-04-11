"""System tests for the P1 burn severity and recovery pipeline.

Verifies the complete system end-to-end: raw satellite bands are ingested,
burn severity is classified, and recovery models with time-to-recovery
estimates are produced.  All tests run against synthetic data via the
``run_pipeline`` entry point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from projects.p1_burn_severity.src.pipeline import run_pipeline
from shared.data.generate_synthetic import generate_synthetic_burn_rasters
from shared.utils.io import read_raster

# -----------------------------------------------------------------------
# Shared fixture: run the full pipeline once per test class
# -----------------------------------------------------------------------


@pytest.fixture()
def pipeline_config_path(tmp_path: Path) -> Path:
    """Create synthetic imagery and a config YAML pointing to tmp_path."""
    imagery_dir = tmp_path / "data" / "raw" / "imagery"
    output_dir = tmp_path / "data" / "processed"
    generate_synthetic_burn_rasters(imagery_dir)

    cfg = {
        "data": {
            "imagery_dir": str(imagery_dir),
            "output_dir": str(output_dir),
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
    config_path.write_text(yaml.dump(cfg))
    return config_path


@pytest.fixture()
def pipeline_result(pipeline_config_path: Path) -> dict:
    """Execute the full pipeline and return the result dict."""
    return run_pipeline(pipeline_config_path)


# -----------------------------------------------------------------------
# System test class
# -----------------------------------------------------------------------


class TestBurnSeveritySystem:
    """System-level tests for the P1 burn severity pipeline."""

    def test_system_produces_all_outputs(self, pipeline_result: dict) -> None:
        """The pipeline must produce severity raster, dNBR, recovery
        timeseries, and recovery model outputs."""
        paths = pipeline_result["output_paths"]
        for key in ("severity", "dnbr", "recovery_timeseries", "recovery_model"):
            assert key in paths, f"Missing output key: {key}"
            assert Path(paths[key]).exists(), f"Output file missing: {paths[key]}"

    def test_severity_classification_valid_range(self, pipeline_result: dict) -> None:
        """Severity raster must contain only valid class codes (0-4, 255)."""
        sev_path = pipeline_result["output_paths"]["severity"]
        sev_data, _ = read_raster(sev_path)
        sev = sev_data[0]
        unique_vals = set(np.unique(sev))
        valid_codes = {0, 1, 2, 3, 4, 255}
        assert unique_vals.issubset(valid_codes), (
            f"Invalid severity codes found: {unique_vals - valid_codes}"
        )

    def test_recovery_model_quality(self, pipeline_result: dict) -> None:
        """At least one severity class must have R-squared > 0."""
        recovery_records = pipeline_result["summary"]["recovery_model"]
        assert len(recovery_records) > 0, "No recovery models were fitted"
        r2_values = [rec["r_squared"] for rec in recovery_records]
        assert any(r2 > 0 for r2 in r2_values), (
            f"No severity class has R^2 > 0: {r2_values}"
        )

    def test_dnbr_values_physically_plausible(self, pipeline_result: dict) -> None:
        """dNBR values must lie within [-2, 2] (physically realistic)."""
        dnbr_path = pipeline_result["output_paths"]["dnbr"]
        dnbr_data, _ = read_raster(dnbr_path)
        dnbr = dnbr_data[0]
        finite = dnbr[np.isfinite(dnbr)]
        assert finite.min() >= -2.0, f"dNBR min {finite.min()} below -2"
        assert finite.max() <= 2.0, f"dNBR max {finite.max()} above 2"

    def test_burned_area_less_than_total(self, pipeline_result: dict) -> None:
        """Burned pixel count must be less than total pixel count."""
        summary = pipeline_result["summary"]
        total = summary["total_pixels"]
        burned = sum(
            count for cls, count in summary["class_counts"].items() if cls != 0
        )
        assert burned < total, (
            f"Burned pixels ({burned}) >= total pixels ({total})"
        )
        assert burned > 0, "No burned pixels detected"

    def test_system_reproducible(self, pipeline_config_path: Path) -> None:
        """Running the pipeline twice with identical inputs produces
        identical outputs."""
        result_a = run_pipeline(pipeline_config_path)
        result_b = run_pipeline(pipeline_config_path)

        # Compare severity class counts
        assert result_a["summary"]["class_counts"] == result_b["summary"]["class_counts"]

        # Compare mean dNBR
        assert result_a["summary"]["mean_dnbr"] == pytest.approx(
            result_b["summary"]["mean_dnbr"]
        )

        # Compare recovery model parameters
        for rec_a, rec_b in zip(
            result_a["summary"]["recovery_model"],
            result_b["summary"]["recovery_model"],
        ):
            assert rec_a["severity_class"] == rec_b["severity_class"]
            assert rec_a["a"] == pytest.approx(rec_b["a"])
            assert rec_a["b"] == pytest.approx(rec_b["b"])
