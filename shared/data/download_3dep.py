"""Download USGS 3DEP LiDAR tiles for a bounding box.

Uses the USGS 3DEP STAC endpoint (via pystac-client) to discover and
download LAZ tiles intersecting a given area of interest.

Usage:
    python -m shared.data.download_3dep \
        --bbox -120.5 38.8 -120.3 39.0 \
        --output-dir data/raw/lidar
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
COLLECTION = "3dep-lidar-copc"


def search_3dep_tiles(
    bbox: tuple[float, float, float, float],
    max_items: int = 10,
) -> list[dict]:
    """Search the 3DEP STAC catalog for LiDAR tiles.

    Parameters
    ----------
    bbox : tuple
        (west, south, east, north) in WGS84 degrees.
    max_items : int
        Maximum number of items to return.

    Returns
    -------
    list[dict]
        List of dicts with 'id', 'href', and 'bbox' keys.
    """
    if Client is None:
        raise ImportError("pystac-client is required: pip install pystac-client")

    client = Client.open(STAC_URL)
    search = client.search(
        collections=[COLLECTION],
        bbox=bbox,
        max_items=max_items,
    )

    tiles = []
    for item in search.items():
        asset = item.assets.get("data") or next(iter(item.assets.values()))
        tiles.append({
            "id": item.id,
            "href": asset.href,
            "bbox": item.bbox,
        })
    return tiles


def download_tile(href: str, output_dir: Path, filename: str | None = None) -> Path:
    """Download a single LAZ/COPC tile.

    Parameters
    ----------
    href : str
        URL to the tile.
    output_dir : Path
        Directory to save the tile.
    filename : str, optional
        Override filename. Defaults to URL basename.

    Returns
    -------
    Path
        Path to downloaded file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = href.split("/")[-1].split("?")[0]
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


def download_3dep(
    bbox: tuple[float, float, float, float],
    output_dir: str | Path,
    max_tiles: int = 5,
) -> list[Path]:
    """Search and download 3DEP LiDAR tiles for a bounding box.

    Parameters
    ----------
    bbox : tuple
        (west, south, east, north) in WGS84 degrees.
    output_dir : str or Path
        Download directory.
    max_tiles : int
        Maximum number of tiles to download.

    Returns
    -------
    list[Path]
        Paths to downloaded files.
    """
    output_dir = Path(output_dir)
    tiles = search_3dep_tiles(bbox, max_items=max_tiles)

    paths = []
    for tile in tiles:
        path = download_tile(tile["href"], output_dir)
        paths.append(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download USGS 3DEP LiDAR tiles")
    parser.add_argument("--bbox", nargs=4, type=float, required=True,
                        help="west south east north (WGS84)")
    parser.add_argument("--output-dir", default="data/raw/lidar")
    parser.add_argument("--max-tiles", type=int, default=5)
    args = parser.parse_args()

    paths = download_3dep(tuple(args.bbox), args.output_dir, args.max_tiles)
    for p in paths:
        print(f"  Downloaded: {p}")
