"""Tests for ground point classification using PDAL SMRF filter.

These tests mock the ``pdal`` dependency so they always run, even when
the native pdal package is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestClassifyGround:
    """Tests for classify_ground()."""

    def test_classify_ground_returns_path(self, tmp_path: Path, default_config: dict) -> None:
        """classify_ground should return a Path to the classified LAZ file."""
        laz_path = tmp_path / "test_cloud.laz"
        laz_path.touch()
        default_config["data"]["output_dir"] = str(tmp_path / "output")

        mock_pdal = MagicMock()
        with patch.dict(sys.modules, {"pdal": mock_pdal}):
            output_dir = tmp_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "test_cloud_classified.laz").touch()

            from projects.p3_itc_delineation.src.ground_classify import classify_ground

            result = classify_ground(laz_path, default_config)

        assert isinstance(result, Path)
        assert result.exists()
        assert result.name == "test_cloud_classified.laz"

    def test_pdal_pipeline_called(self, tmp_path: Path, default_config: dict) -> None:
        """classify_ground should construct a pdal Pipeline and call execute()."""
        laz_path = tmp_path / "test_cloud.laz"
        laz_path.touch()
        default_config["data"]["output_dir"] = str(tmp_path / "output")

        mock_pdal = MagicMock()
        mock_pipeline_instance = MagicMock()
        mock_pdal.Pipeline.return_value = mock_pipeline_instance

        with patch.dict(sys.modules, {"pdal": mock_pdal}):
            output_dir = tmp_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "test_cloud_classified.laz").touch()

            from projects.p3_itc_delineation.src.ground_classify import classify_ground

            classify_ground(laz_path, default_config)

        mock_pdal.Pipeline.assert_called_once()
        mock_pipeline_instance.execute.assert_called_once()

    def test_smrf_parameters_in_pipeline(self, tmp_path: Path, default_config: dict) -> None:
        """classify_ground should include SMRF parameters in the pipeline JSON."""
        laz_path = tmp_path / "test_cloud.laz"
        laz_path.touch()
        default_config["data"]["output_dir"] = str(tmp_path / "output")
        default_config["processing"]["smrf_cell"] = 2.0
        default_config["processing"]["smrf_slope"] = 0.25
        default_config["processing"]["smrf_window"] = 20.0

        mock_pdal = MagicMock()
        with patch.dict(sys.modules, {"pdal": mock_pdal}):
            output_dir = tmp_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "test_cloud_classified.laz").touch()

            from projects.p3_itc_delineation.src.ground_classify import classify_ground

            classify_ground(laz_path, default_config)

        call_args = mock_pdal.Pipeline.call_args
        pipeline_json_str = call_args[0][0]
        assert "filters.smrf" in pipeline_json_str
        assert "2.0" in pipeline_json_str  # smrf_cell
        assert "0.25" in pipeline_json_str  # smrf_slope
        assert "20.0" in pipeline_json_str  # smrf_window

    def test_output_uses_stem_name(self, tmp_path: Path, default_config: dict) -> None:
        """Output filename should be derived from the input LAZ stem."""
        laz_path = tmp_path / "my_lidar_data.laz"
        laz_path.touch()
        default_config["data"]["output_dir"] = str(tmp_path / "output")

        mock_pdal = MagicMock()
        with patch.dict(sys.modules, {"pdal": mock_pdal}):
            output_dir = tmp_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "my_lidar_data_classified.laz").touch()

            from projects.p3_itc_delineation.src.ground_classify import classify_ground

            result = classify_ground(laz_path, default_config)

        assert result.name == "my_lidar_data_classified.laz"
