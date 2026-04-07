"""Tests for CHM generation module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shared.utils.io import read_raster, write_raster


class TestGenerateCHM:
    """Verify CHM = DSM - DTM with smoothing and clamping."""

    def test_chm_output_exists(self, chm_path: Path, dtm_path: Path, default_config: dict):
        """CHM generation from pre-built DSM and DTM produces a valid raster."""
        from projects.p3_itc_delineation.src.chm import generate_chm

        # Use synthetic CHM as a stand-in DSM (heights above ground)
        # Create a fake DSM = DTM + CHM
        chm_data, chm_profile = read_raster(chm_path)
        dtm_data, dtm_profile = read_raster(dtm_path)

        dsm = dtm_data[0] + chm_data[0]
        dsm_path = Path(default_config["data"]["output_dir"]) / "test_dsm.tif"
        dsm_path.parent.mkdir(parents=True, exist_ok=True)
        write_raster(dsm_path, dsm, dtm_profile)

        result = generate_chm(dsm_path, dtm_path, default_config)
        assert result.exists()

    def test_chm_non_negative(self, chm_path: Path, dtm_path: Path, default_config: dict):
        """All CHM values must be >= 0 after clamping."""
        from projects.p3_itc_delineation.src.chm import generate_chm

        chm_data, chm_profile = read_raster(chm_path)
        dtm_data, dtm_profile = read_raster(dtm_path)

        dsm = dtm_data[0] + chm_data[0]
        dsm_path = Path(default_config["data"]["output_dir"]) / "test_dsm2.tif"
        dsm_path.parent.mkdir(parents=True, exist_ok=True)
        write_raster(dsm_path, dsm, dtm_profile)

        result_path = generate_chm(dsm_path, dtm_path, default_config)
        result_data, _ = read_raster(result_path)
        assert np.all(result_data >= 0), "CHM contains negative values"

    def test_chm_smoothed(self, chm_path: Path, dtm_path: Path, default_config: dict):
        """Smoothed CHM should have lower max than raw DSM-DTM difference."""
        from projects.p3_itc_delineation.src.chm import generate_chm

        chm_data, chm_profile = read_raster(chm_path)
        dtm_data, dtm_profile = read_raster(dtm_path)

        raw_diff = (dtm_data[0] + chm_data[0]) - dtm_data[0]
        raw_max = float(np.max(raw_diff))

        dsm = dtm_data[0] + chm_data[0]
        dsm_path = Path(default_config["data"]["output_dir"]) / "test_dsm3.tif"
        dsm_path.parent.mkdir(parents=True, exist_ok=True)
        write_raster(dsm_path, dsm, dtm_profile)

        result_path = generate_chm(dsm_path, dtm_path, default_config)
        result_data, _ = read_raster(result_path)
        smoothed_max = float(np.max(result_data))

        # Gaussian smoothing should reduce peak values
        assert smoothed_max <= raw_max + 0.01
