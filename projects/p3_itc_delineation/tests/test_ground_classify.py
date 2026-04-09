"""Tests for ground point classification using PDAL SMRF filter.

These tests require pdal, which is not available in CI.
They are automatically skipped if the dependency is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pdal = pytest.importorskip("pdal")


class TestClassifyGround:
    """Tests for classify_ground()."""

    def test_classify_ground_returns_path(self, tmp_path: Path, default_config: dict) -> None:
        """classify_ground should return a Path to the classified LAZ file."""
        from projects.p3_itc_delineation.src.ground_classify import classify_ground

        laz_path = tmp_path / "test_cloud.laz"
        laz_path.touch()
        default_config["data"]["output_dir"] = str(tmp_path / "output")

        mock_pdal = MagicMock()
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "pdal", mock_pdal)
            output_dir = tmp_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "test_cloud_classified.laz").touch()

            result = classify_ground(laz_path, default_config)

        assert isinstance(result, Path)
        assert result.exists()
        assert result.name == "test_cloud_classified.laz"
