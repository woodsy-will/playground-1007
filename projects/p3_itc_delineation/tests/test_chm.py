"""Tests for CHM generation module."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from shared.utils.io import read_raster, write_raster


class TestGenerateCHM:
    """Verify CHM = DSM - DTM with smoothing and clamping."""

    def test_chm_output_exists(self, chm_path: Path, dtm_path: Path, default_config: dict):
        """CHM generation from pre-built DSM and DTM produces a valid raster."""
        from projects.p3_itc_delineation.src.chm import generate_chm

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

    def test_generate_chm_from_laz_file(
        self, chm_path: Path, dtm_path: Path, default_config: dict
    ):
        """CHM generation from a .laz file triggers the LAZ branch and
        calls _generate_dsm_from_laz, which imports pdal internally."""
        from projects.p3_itc_delineation.src.chm import generate_chm

        output_dir = Path(default_config["data"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create a fake .laz file (just needs to exist with the right extension)
        laz_file = output_dir / "test_input.laz"
        laz_file.write_bytes(b"FAKE_LAZ_DATA")

        # Pre-create the DSM file that _generate_dsm_from_laz would produce.
        # Use the CHM data added to DTM data as a valid DSM raster.
        chm_data, chm_profile = read_raster(chm_path)
        dtm_data, _ = read_raster(dtm_path)
        dsm = dtm_data[0] + chm_data[0]
        dsm_path = output_dir / "dsm.tif"
        write_raster(dsm_path, dsm, chm_profile)

        # Mock pdal module so the import inside _generate_dsm_from_laz works
        mock_pdal = MagicMock()
        mock_pipeline = MagicMock()
        mock_pdal.Pipeline.return_value = mock_pipeline

        with patch.dict(sys.modules, {"pdal": mock_pdal}):
            result = generate_chm(laz_file, dtm_path, default_config)

        # The LAZ branch should have been triggered: pdal.Pipeline was called
        mock_pdal.Pipeline.assert_called_once()
        mock_pipeline.execute.assert_called_once()

        # Output CHM should exist and contain non-negative values
        assert result.exists()
        result_data, _ = read_raster(result)
        assert np.all(result_data >= 0)
