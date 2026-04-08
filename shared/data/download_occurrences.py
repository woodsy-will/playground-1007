"""Download species occurrence records from GBIF.

Uses the GBIF occurrence API to fetch point records for a given taxon
within a bounding box, then outputs a GeoPackage.

Usage:
    python -m shared.data.download_occurrences \
        --species "Pekania pennanti" \
        --bbox -122.0 36.0 -118.0 40.0 \
        --output data/raw/occurrences.gpkg
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from pyproj import CRS
from shapely.geometry import Point

GBIF_API = "https://api.gbif.org/v1"
CRS_4326 = CRS.from_epsg(4326)
CRS_3310 = CRS.from_epsg(3310)


def search_species_key(species_name: str) -> int | None:
    """Look up the GBIF taxon key for a species name.

    Parameters
    ----------
    species_name : str
        Scientific name (e.g., "Pekania pennanti").

    Returns
    -------
    int or None
        GBIF species key, or None if not found.
    """
    resp = requests.get(
        f"{GBIF_API}/species/match",
        params={"name": species_name},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("usageKey")


def download_occurrences(
    species_name: str,
    bbox: tuple[float, float, float, float] | None = None,
    limit: int = 300,
) -> gpd.GeoDataFrame:
    """Fetch occurrence records from GBIF.

    Parameters
    ----------
    species_name : str
        Scientific name.
    bbox : tuple, optional
        (west, south, east, north) in WGS84. If None, no spatial filter.
    limit : int
        Maximum records to fetch.

    Returns
    -------
    GeoDataFrame
        Occurrence points in EPSG:3310.
    """
    taxon_key = search_species_key(species_name)
    if taxon_key is None:
        raise ValueError(f"Species not found in GBIF: {species_name}")

    params = {
        "taxonKey": taxon_key,
        "hasCoordinate": True,
        "limit": min(limit, 300),
        "offset": 0,
    }
    if bbox:
        w, s, e, n = bbox
        params["decimalLatitude"] = f"{s},{n}"
        params["decimalLongitude"] = f"{w},{e}"

    records = []
    while len(records) < limit:
        resp = requests.get(
            f"{GBIF_API}/occurrence/search",
            params=params,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            break
        records.extend(results)
        if data.get("endOfRecords", True):
            break
        params["offset"] += len(results)

    if not records:
        return gpd.GeoDataFrame(columns=["species", "source", "year", "geometry"])

    df = pd.DataFrame(records)
    gdf = gpd.GeoDataFrame(
        {
            "species": df.get("species", species_name),
            "source": "GBIF",
            "year": df.get("year"),
            "gbif_id": df.get("key"),
        },
        geometry=[
            Point(row["decimalLongitude"], row["decimalLatitude"])
            for _, row in df.iterrows()
            if "decimalLongitude" in row and "decimalLatitude" in row
        ],
        crs=CRS_4326,
    )

    # Reproject to EPSG:3310
    return gdf.to_crs(CRS_3310)


def save_occurrences(
    species_name: str,
    output_path: str | Path,
    bbox: tuple[float, float, float, float] | None = None,
    limit: int = 300,
) -> Path:
    """Download and save occurrence records to GeoPackage.

    Parameters
    ----------
    species_name : str
        Scientific name.
    output_path : str or Path
        Output GeoPackage path.
    bbox : tuple, optional
        Spatial filter in WGS84.
    limit : int
        Maximum records.

    Returns
    -------
    Path
        Path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdf = download_occurrences(species_name, bbox, limit)
    gdf.to_file(output_path, driver="GPKG")
    print(f"Saved {len(gdf)} records to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GBIF occurrences")
    parser.add_argument("--species", default="Pekania pennanti")
    parser.add_argument("--bbox", nargs=4, type=float, default=None)
    parser.add_argument("--output", default="data/raw/occurrences.gpkg")
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()

    bbox = tuple(args.bbox) if args.bbox else None
    save_occurrences(args.species, args.output, bbox, args.limit)
