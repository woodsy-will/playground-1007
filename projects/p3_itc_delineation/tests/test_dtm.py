"""Tests for DTM generation from classified LiDAR point clouds.

These tests mock the ``pdal`` dependency so they always run, even when
the native pdal package is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestGenerateDTM:
    """Tests for generate_dtm()."""

    def test_generate_dtm_returns_path(self, tmp_path: Path, default_config: dict) -> None:
        """generate_dtm should return a Path object."""
        classified_laz = tmp_path / "test_classified.laz"
        classified_laz.touch()
        default_config["data"]["output_dir"] = str(tmp_path / "output")

        mock_pdal = MagicMock()
        with patch.dict(sys.modules, {"pdal": mock_pdal}):
            output_dir = tmp_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "dtm.tif").touch()

            from projects.p3_itc_delineation.src.dtm import generate_dtm

            result = generate_dtm(classified_laz, default_config)

        assert isinstance(result, Path)

    def test_output_file_exists(self, tmp_path: Path, default_config: dict) -> None:
        """generate_dtm should produce an output file at the expected path."""
        classified_laz = tmp_path / "test_classified.laz"
        classified_laz.touch()
        default_config["data"]["output_dir"] = str(tmp_path / "output")

        mock_pdal = MagicMock()
        with patch.dict(sys.modules, {"pdal": mock_pdal}):
            output_dir = tmp_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "dtm.tif").touch()

            from projects.p3_itc_delineation.src.dtm import generate_dtm

            result = generate_dtm(classified_laz, default_config)

        assert result.exists()
        assert result.name == "dtm.tif"

    def test_pdal_pipeline_called(self, tmp_path: Path, default_config: dict) -> None:
        """generate_dtm should construct a pdal Pipeline and call execute()."""
        classified_laz = tmp_path / "test_classified.laz"
        classified_laz.touch()
        default_config["data"]["output_dir"] = str(tmp_path / "output")

        mock_pdal = MagicMock()
        mock_pipeline_instance = MagicMock()
        mock_pdal.Pipeline.return_value = mock_pipeline_instance

        with patch.dict(sys.modules, {"pdal": mock_pdal}):
            output_dir = tmp_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "dtm.tif").touch()

            from projects.p3_itc_delineation.src.dtm import generate_dtm

            generate_dtm(classified_laz, default_config)

        mock_pdal.Pipeline.assert_called_once()
        mock_pipeline_instance.execute.assert_called_once()

    def test_custom_resolution(self, tmp_path: Path, default_config: dict) -> None:
        """generate_dtm should honour the dtm_resolution config parameter."""
        classified_laz = tmp_path / "test_classified.laz"
        classified_laz.touch()
        default_config["data"]["output_dir"] = str(tmp_path / "output")
        default_config["processing"]["dtm_resolution"] = 2.5

        mock_pdal = MagicMock()
        with patch.dict(sys.modules, {"pdal": mock_pdal}):
            output_dir = tmp_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "dtm.tif").touch()

            from projects.p3_itc_delineation.src.dtm import generate_dtm

            result = generate_dtm(classified_laz, default_config)

        assert isinstance(result, Path)
        # The pipeline JSON passed to pdal.Pipeline should contain the resolution
        call_args = mock_pdal.Pipeline.call_args
        pipeline_json_str = call_args[0][0]
        assert "2.5" in pipeline_json_str
