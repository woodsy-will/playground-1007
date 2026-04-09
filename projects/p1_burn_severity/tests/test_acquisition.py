"""Tests for P1 scene acquisition via STAC API.

These tests mock external dependencies (pystac_client, httpx/urllib) so they
run without network access or optional packages.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from projects.p1_burn_severity.src.acquisition import (
    _HAS_PYSTAC,
    _STAC_ENDPOINTS,
    download_scene,
    search_scenes,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stac_item(item_id: str, cloud_cover: float = 5.0):
    """Return a mock STAC item with the expected interface."""
    asset_b8a = SimpleNamespace(href=f"https://example.com/{item_id}/B8A.tif")
    asset_b12 = SimpleNamespace(href=f"https://example.com/{item_id}/B12.tif")
    asset_scl = SimpleNamespace(href=f"https://example.com/{item_id}/SCL.tif")

    item = MagicMock()
    item.id = item_id
    item.datetime = "2024-07-15T10:30:00Z"
    item.properties = {"eo:cloud_cover": cloud_cover}
    item.assets = {"B8A": asset_b8a, "B12": asset_b12, "SCL": asset_scl}
    item.bbox = [-121.0, 37.0, -120.0, 38.0]
    return item


# ---------------------------------------------------------------------------
# TestSearchScenes
# ---------------------------------------------------------------------------


class TestSearchScenes:
    """Tests for ``search_scenes``."""

    BBOX = (-121.5, 37.0, -120.5, 38.0)
    DATE_RANGE = ("2024-06-01", "2024-09-30")

    def test_raises_without_pystac(self):
        """search_scenes should raise RuntimeError when pystac_client is not installed."""
        if _HAS_PYSTAC:
            pytest.skip("pystac_client IS installed; cannot test missing-dep path")

        with pytest.raises(RuntimeError, match="pystac_client is required"):
            search_scenes(self.BBOX, self.DATE_RANGE, config={})

    @patch("projects.p1_burn_severity.src.acquisition._HAS_PYSTAC", True)
    @patch("projects.p1_burn_severity.src.acquisition.STACClient", create=True)
    def test_returns_items(self, mock_stac_cls):
        """search_scenes should return a list of scene dicts from STAC."""
        items = [_make_stac_item("scene_A"), _make_stac_item("scene_B", 12.0)]

        mock_client = MagicMock()
        mock_search = MagicMock()
        mock_search.items.return_value = iter(items)
        mock_client.search.return_value = mock_search
        mock_stac_cls.open.return_value = mock_client

        config = {"acquisition": {"cloud_cover_max": 20, "source": "element84"}}
        result = search_scenes(self.BBOX, self.DATE_RANGE, config)

        assert len(result) == 2
        assert result[0]["id"] == "scene_A"
        assert result[1]["cloud_cover"] == 12.0
        assert "B8A" in result[0]["assets"]
        mock_stac_cls.open.assert_called_once_with(_STAC_ENDPOINTS["element84"])

    @patch("projects.p1_burn_severity.src.acquisition._HAS_PYSTAC", True)
    @patch("projects.p1_burn_severity.src.acquisition.STACClient", create=True)
    def test_no_results(self, mock_stac_cls):
        """search_scenes should return an empty list when STAC yields nothing."""
        mock_client = MagicMock()
        mock_search = MagicMock()
        mock_search.items.return_value = iter([])
        mock_client.search.return_value = mock_search
        mock_stac_cls.open.return_value = mock_client

        config = {"acquisition": {"cloud_cover_max": 5, "source": "copernicus"}}
        result = search_scenes(self.BBOX, self.DATE_RANGE, config)
        assert result == []

    @patch("projects.p1_burn_severity.src.acquisition._HAS_PYSTAC", True)
    @patch("projects.p1_burn_severity.src.acquisition.STACClient", create=True)
    def test_missing_config_uses_defaults(self, mock_stac_cls):
        """When config lacks acquisition keys, defaults should be used."""
        items = [_make_stac_item("default_scene")]

        mock_client = MagicMock()
        mock_search = MagicMock()
        mock_search.items.return_value = iter(items)
        mock_client.search.return_value = mock_search
        mock_stac_cls.open.return_value = mock_client

        result = search_scenes(self.BBOX, self.DATE_RANGE, config={})

        assert len(result) == 1
        assert result[0]["id"] == "default_scene"
        mock_stac_cls.open.assert_called_once_with(_STAC_ENDPOINTS["copernicus"])
        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs["query"] == {"eo:cloud_cover": {"lte": 20}}


# ---------------------------------------------------------------------------
# TestDownloadScene
# ---------------------------------------------------------------------------


class TestDownloadScene:
    """Tests for ``download_scene``."""

    def _scene_meta(self) -> dict:
        return {
            "id": "S2A_20240715",
            "assets": {
                "B8A": "https://example.com/B8A.tif",
                "B12": "https://example.com/B12.tif",
                "SCL": "https://example.com/SCL.tif",
            },
        }

    def test_writes_file_urllib_fallback(self, tmp_path):
        """download_scene should write band files via urllib when httpx is absent."""
        import urllib.request

        scene = self._scene_meta()

        def _fake_retrieve(url, dest):
            Path(dest).write_bytes(b"FAKE_TIFF_DATA")

        with (
            patch.dict(sys.modules, {"httpx": None}),
            patch.object(urllib.request, "urlretrieve", side_effect=_fake_retrieve),
        ):
            result = download_scene(scene, tmp_path, config={})

        assert "nir" in result
        assert "swir" in result
        assert "scl" in result
        for path in result.values():
            assert path.exists()

    def test_connection_error_handled(self, tmp_path):
        """download_scene should catch exceptions and return empty dict."""
        scene = self._scene_meta()

        mock_httpx = MagicMock()
        mock_httpx.stream.side_effect = ConnectionError("Connection refused")

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            result = download_scene(scene, tmp_path, config={})

        assert len(result) == 0
