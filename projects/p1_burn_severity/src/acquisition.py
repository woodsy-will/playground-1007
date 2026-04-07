"""Scene acquisition via STAC API for Sentinel-2 L2A imagery.

Searches the Copernicus / Element84 STAC catalogue for cloud-filtered
Sentinel-2 scenes and downloads NIR (B8A), SWIR (B12), and SCL bands.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from shared.utils.logging import get_logger

logger = get_logger("p1.acquisition")

try:
    from pystac_client import Client as STACClient  # type: ignore[import-untyped]

    _HAS_PYSTAC = True
except ImportError:
    _HAS_PYSTAC = False
    logger.warning("pystac_client not installed; STAC search disabled.")


# Copernicus Data Space STAC endpoint (public)
_STAC_ENDPOINTS: dict[str, str] = {
    "copernicus": "https://catalogue.dataspace.copernicus.eu/stac",
    "element84": "https://earth-search.aws.element84.com/v1",
}

# Sentinel-2 L2A band names used in this project
_BAND_MAP: dict[str, str] = {
    "nir": "B8A",
    "swir": "B12",
    "scl": "SCL",
}


def search_scenes(
    bbox: tuple[float, float, float, float],
    date_range: tuple[str, str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Search for Sentinel-2 L2A scenes matching spatial/temporal criteria.

    Parameters
    ----------
    bbox : tuple
        Bounding box as (west, south, east, north) in EPSG:4326.
    date_range : tuple
        Start and end dates as ISO-8601 strings (``"YYYY-MM-DD"``).
    config : dict
        Project configuration (must include ``acquisition.cloud_cover_max``).

    Returns
    -------
    list[dict]
        List of scene metadata dicts with keys ``id``, ``datetime``,
        ``cloud_cover``, ``assets``, and ``bbox``.
    """
    if not _HAS_PYSTAC:
        raise RuntimeError(
            "pystac_client is required for STAC search. "
            "Install with: pip install pystac-client"
        )

    acq = config.get("acquisition", {})
    cloud_max = acq.get("cloud_cover_max", 20)
    source = acq.get("source", "copernicus")
    endpoint = _STAC_ENDPOINTS.get(source, _STAC_ENDPOINTS["element84"])

    client = STACClient.open(endpoint)
    date_str = f"{date_range[0]}/{date_range[1]}"

    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_str,
        query={"eo:cloud_cover": {"lte": cloud_max}},
        max_items=50,
    )

    scenes: list[dict[str, Any]] = []
    for item in search.items():
        scenes.append(
            {
                "id": item.id,
                "datetime": str(item.datetime),
                "cloud_cover": item.properties.get("eo:cloud_cover"),
                "assets": {k: v.href for k, v in item.assets.items()},
                "bbox": item.bbox,
            }
        )

    logger.info("Found %d scenes with cloud cover <= %d%%", len(scenes), cloud_max)
    return scenes


def download_scene(
    scene_meta: dict[str, Any],
    output_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, Path]:
    """Download NIR, SWIR, and SCL bands for a single scene.

    Parameters
    ----------
    scene_meta : dict
        Scene metadata dict returned by :func:`search_scenes`.
    output_dir : str or Path
        Directory to write downloaded bands.
    config : dict
        Project configuration.

    Returns
    -------
    dict[str, Path]
        Mapping of band key (``nir``, ``swir``, ``scl``) to local file path.
    """
    try:
        import httpx  # type: ignore[import-untyped]
    except ImportError:
        import urllib.request

        httpx = None  # type: ignore[assignment]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = scene_meta.get("assets", {})
    scene_id = scene_meta.get("id", "unknown")
    downloaded: dict[str, Path] = {}

    for key, band_name in _BAND_MAP.items():
        href = assets.get(band_name) or assets.get(band_name.lower())
        if href is None:
            logger.warning("Band %s not found in scene %s assets", band_name, scene_id)
            continue

        dest = output_dir / f"{scene_id}_{band_name}.tif"
        if dest.exists():
            logger.info("Band %s already downloaded: %s", band_name, dest)
            downloaded[key] = dest
            continue

        logger.info("Downloading %s for scene %s ...", band_name, scene_id)
        try:
            if httpx is not None:
                with httpx.stream("GET", href, follow_redirects=True) as resp:
                    resp.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in resp.iter_bytes(chunk_size=8192):
                            f.write(chunk)
            else:
                urllib.request.urlretrieve(href, dest)  # noqa: S310
            downloaded[key] = dest
        except Exception:
            logger.exception("Failed to download %s for scene %s", band_name, scene_id)

    return downloaded
