"""Tests for DTM generation from classified LiDAR point clouds.

These tests require whiteboxtools, which is not available in CI.
They are automatically skipped if the dependency is missing.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

whiteboxtools = pytest.importorskip("whiteboxtools")


class TestGenerateDTM:
    """Tests for generate_dtm()."""

    def test_generate_dtm_returns_path(self, tmp_path: Path, default_config: dict) -> None:
        """generate_dtm should return a Path object."""
        from projects.p3_itc_delineation.src.dtm import generate_dtm

        classified_laz = tmp_path / "test_classified.laz"
        classified_laz.touch()
        default_config["data"]["output_dir"] = str(tmp_path / "output")

        # Mock pdal.Pipeline to avoid requiring real LAZ data
        mock_pipeline = MagicMock()
        with patch("projects.p3_itc_delineation.src.dtm.pdal") as mock_pdal:
            mock_pdal.Pipeline.return_value = mock_pipeline
            # Create the expected output file so the path exists
            output_dir = tmp_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "dtm.tif").touch()

            result = generate_dtm(classified_laz, default_config)

        assert isinstance(result, Path)

    def test_output_file_exists(self, tmp_path: Path, default_config: dict) -> None:
        """generate_dtm should produce an output file at the expected path."""
        from projects.p3_itc_delineation.src.dtm import generate_dtm

        classified_laz = tmp_path / "test_classified.laz"
        classified_laz.touch()
        default_config["data"]["output_dir"] = str(tmp_path / "output")

        mock_pipeline = MagicMock()
        with patch("projects.p3_itc_delineation.src.dtm.pdal") as mock_pdal:
            mock_pdal.Pipeline.return_value = mock_pipeline
            output_dir = tmp_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "dtm.tif").touch()

            result = generate_dtm(classified_laz, default_config)

        assert result.exists()
        assert result.name == "dtm.tif"
