"""Download WorldClim v2 bioclimatic rasters.

Fetches 30-second (~1 km) resolution BioClim variables from the
WorldClim website.

Usage:
    python -m shared.data.download_worldclim \
        --variables bio1 bio12 \
        --output-dir data/raw/worldclim
"""

import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

WORLDCLIM_BASE = "https://biogeo.ucdavis.edu/data/worldclim/v2.1/base"

# Standard bioclimatic variables
BIOCLIM_VARS = [f"bio{i}" for i in range(1, 20)]


def download_file(url: str, dest: Path) -> Path:
    """Download a file with progress bar."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    return dest


def download_worldclim(
    variables: list[str] | None = None,
    resolution: str = "30s",
    output_dir: str | Path = "data/raw/worldclim",
) -> list[Path]:
    """Download WorldClim v2.1 bioclimatic variable rasters.

    Parameters
    ----------
    variables : list[str], optional
        Variable names (e.g., ['bio1', 'bio12']). Default: bio1 through bio19.
    resolution : str
        Spatial resolution: '30s', '2.5m', '5m', or '10m'.
    output_dir : str or Path
        Output directory.

    Returns
    -------
    list[Path]
        Paths to downloaded/extracted GeoTIFF files.
    """
    output_dir = Path(output_dir)
    variables = variables or BIOCLIM_VARS

    # WorldClim distributes bioclim as a single zip per resolution
    zip_url = f"{WORLDCLIM_BASE}/wc2.1_{resolution}_bio.zip"
    zip_path = output_dir / f"wc2.1_{resolution}_bio.zip"

    download_file(zip_url, zip_path)

    # Extract
    extract_dir = output_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # Find requested variable TIFFs
    paths = []
    for var in variables:
        # WorldClim naming: wc2.1_30s_bio_1.tif
        var_num = var.replace("bio", "")
        matches = list(extract_dir.rglob(f"*bio_{var_num}.tif"))
        if matches:
            paths.append(matches[0])
        else:
            print(f"  Warning: {var} not found in archive")

    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download WorldClim bioclim vars")
    parser.add_argument("--variables", nargs="+", default=None,
                        help="e.g., bio1 bio12")
    parser.add_argument("--resolution", default="30s",
                        choices=["30s", "2.5m", "5m", "10m"])
    parser.add_argument("--output-dir", default="data/raw/worldclim")
    args = parser.parse_args()

    paths = download_worldclim(args.variables, args.resolution, args.output_dir)
    for p in paths:
        print(f"  {p}")
