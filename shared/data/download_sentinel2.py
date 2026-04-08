"""Download Sentinel-2 L2A scenes via STAC (Copernicus / Planetary Computer).

Usage:
    python -m shared.data.download_sentinel2 \
        --bbox -120.5 38.8 -120.3 39.0 \
        --start-date 2021-08-01 \
        --end-date 2021-10-01 \
        --output-dir data/raw/sentinel2
"""

import argparse
from pathlib import Path

import requests
from tqdm import tqdm

try:
    from pystac_client import Client
except ImportError:
    Client = None


STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

# Bands needed for burn severity analysis
REQUIRED_BANDS = ["B04", "B08", "B8A", "B11", "B12", "SCL"]


def search_scenes(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 20.0,
    max_items: int = 10,
) -> list[dict]:
    """Search for Sentinel-2 L2A scenes.

    Parameters
    ----------
    bbox : tuple
        (west, south, east, north) in WGS84.
    start_date : str
        ISO date string (YYYY-MM-DD).
    end_date : str
        ISO date string.
    max_cloud_cover : float
        Maximum cloud cover percentage.
    max_items : int
        Maximum scenes to return.

    Returns
    -------
    list[dict]
        Scene metadata with asset hrefs.
    """
    if Client is None:
        raise ImportError("pystac-client required: pip install pystac-client")

    client = Client.open(STAC_URL)
    search = client.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        max_items=max_items,
    )

    scenes = []
    for item in search.items():
        scene = {
            "id": item.id,
            "datetime": str(item.datetime),
            "cloud_cover": item.properties.get("eo:cloud_cover", None),
            "assets": {},
        }
        for band in REQUIRED_BANDS:
            if band in item.assets:
                scene["assets"][band] = item.assets[band].href
        scenes.append(scene)
    return scenes


def download_band(href: str, output_dir: Path, filename: str) -> Path:
    """Download a single band asset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / filename
    if dest.exists():
        return dest

    resp = requests.get(href, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=filename) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    return dest


def download_scene(
    scene: dict,
    output_dir: str | Path,
    bands: list[str] | None = None,
) -> dict[str, Path]:
    """Download bands for a single scene.

    Parameters
    ----------
    scene : dict
        Scene metadata from search_scenes().
    output_dir : str or Path
        Output directory.
    bands : list[str], optional
        Bands to download. Default: all REQUIRED_BANDS.

    Returns
    -------
    dict[str, Path]
        Mapping of band name to downloaded file path.
    """
    output_dir = Path(output_dir) / scene["id"]
    bands = bands or REQUIRED_BANDS
    paths = {}
    for band in bands:
        if band in scene["assets"]:
            path = download_band(
                scene["assets"][band],
                output_dir,
                f"{band}.tif",
            )
            paths[band] = path
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Sentinel-2 scenes")
    parser.add_argument("--bbox", nargs=4, type=float, required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--max-cloud-cover", type=float, default=20.0)
    parser.add_argument("--output-dir", default="data/raw/sentinel2")
    parser.add_argument("--max-items", type=int, default=5)
    args = parser.parse_args()

    scenes = search_scenes(
        tuple(args.bbox), args.start_date, args.end_date,
        args.max_cloud_cover, args.max_items,
    )
    print(f"Found {len(scenes)} scenes")
    for scene in scenes:
        paths = download_scene(scene, args.output_dir)
        for band, path in paths.items():
            print(f"  {band}: {path}")
