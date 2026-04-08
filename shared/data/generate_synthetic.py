"""Generate synthetic test data for all four portfolio projects.

Creates small, deterministic datasets suitable for unit and integration tests.
All outputs use EPSG:3310 (California Albers) and write to a configurable
output directory.

Usage:
    python -m shared.data.generate_synthetic --output-dir tests/fixtures
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely.geometry import Point, box

from shared.utils.io import make_profile, write_raster

CRS_3310 = CRS.from_epsg(3310)

# Reproducible random state
RNG = np.random.default_rng(42)

# Small AOI in California Albers (meters) — ~200m x 200m
AOI_XMIN, AOI_YMIN = -200_000.0, -50_000.0
AOI_XMAX, AOI_YMAX = AOI_XMIN + 200.0, AOI_YMIN + 200.0
AOI_BOUNDS = (AOI_XMIN, AOI_YMIN, AOI_XMAX, AOI_YMAX)


# ---------------------------------------------------------------------------
# P3 — LiDAR / ITC helpers
# ---------------------------------------------------------------------------


def generate_synthetic_lidar(
    output_dir: Path,
    n_trees: int = 5,
    pts_per_tree: int = 200,
    ground_pts: int = 500,
) -> Path:
    """Create a synthetic LiDAR point cloud as CSV (x, y, z, classification).

    Produces a flat ground plane at z=500 m with conical trees.  This is a
    simplified CSV substitute for LAZ — suitable for testing processing logic
    without requiring PDAL I/O.

    Returns path to the CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ground points — flat plane at z = 500
    gx = RNG.uniform(AOI_XMIN, AOI_XMAX, ground_pts)
    gy = RNG.uniform(AOI_YMIN, AOI_YMAX, ground_pts)
    gz = np.full(ground_pts, 500.0) + RNG.normal(0, 0.05, ground_pts)
    g_class = np.full(ground_pts, 2, dtype=int)  # ground class

    # Tree points — conical shapes
    tree_centers = []
    all_x, all_y, all_z, all_cls = [gx], [gy], [gz], [g_class]

    for i in range(n_trees):
        cx = AOI_XMIN + (i + 1) * (AOI_XMAX - AOI_XMIN) / (n_trees + 1)
        cy = AOI_YMIN + (AOI_YMAX - AOI_YMIN) / 2 + RNG.uniform(-20, 20)
        tree_height = RNG.uniform(15, 35)
        crown_radius = tree_height * 0.15

        tree_centers.append({"x": cx, "y": cy, "height": tree_height})

        # Points on a cone
        r = RNG.uniform(0, crown_radius, pts_per_tree)
        theta = RNG.uniform(0, 2 * np.pi, pts_per_tree)
        tx = cx + r * np.cos(theta)
        ty = cy + r * np.sin(theta)
        # Height decreases linearly with distance from center
        tz = 500.0 + tree_height * (1 - r / crown_radius) + RNG.normal(0, 0.1, pts_per_tree)
        t_class = np.full(pts_per_tree, 1, dtype=int)  # unclassified

        all_x.append(tx)
        all_y.append(ty)
        all_z.append(tz)
        all_cls.append(t_class)

    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    z = np.concatenate(all_z)
    cls = np.concatenate(all_cls)

    df = pd.DataFrame({"x": x, "y": y, "z": z, "classification": cls})
    csv_path = output_dir / "synthetic_lidar.csv"
    df.to_csv(csv_path, index=False)

    # Also save tree reference for validation
    ref_df = pd.DataFrame(tree_centers)
    ref_df.to_csv(output_dir / "synthetic_tree_reference.csv", index=False)

    return csv_path


def generate_synthetic_cruise_plots(output_dir: Path, n_trees: int = 5) -> Path:
    """Create synthetic cruise plot data matching the synthetic LiDAR trees."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_path = output_dir / "synthetic_tree_reference.csv"
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
    else:
        # Fallback — generate standalone
        ref = pd.DataFrame({
            "x": [AOI_XMIN + (i + 1) * 200 / (n_trees + 1) for i in range(n_trees)],
            "y": [AOI_YMIN + 100] * n_trees,
            "height": RNG.uniform(15, 35, n_trees),
        })

    cruise = pd.DataFrame({
        "stem_x": ref["x"] + RNG.normal(0, 0.5, len(ref)),
        "stem_y": ref["y"] + RNG.normal(0, 0.5, len(ref)),
        "dbh_inches": RNG.uniform(8, 30, len(ref)),
        "height_ft": ref["height"] * 3.28084,
        "species": RNG.choice(
            ["ABCO", "PIPO", "CADE", "PSME", "ABMA"], len(ref)
        ),
        "diameter_class": pd.cut(
            RNG.uniform(8, 30, len(ref)),
            bins=[0, 12, 20, 100],
            labels=["small", "medium", "large"],
        ),
    })
    path = output_dir / "cruise_plots.csv"
    cruise.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# P3 — CHM / DTM rasters
# ---------------------------------------------------------------------------


def generate_synthetic_chm(output_dir: Path, n_trees: int = 5) -> Path:
    """Create a synthetic CHM raster with Gaussian-shaped tree crowns."""
    output_dir.mkdir(parents=True, exist_ok=True)
    resolution = 1.0
    profile = make_profile(AOI_BOUNDS, resolution)

    height = profile["height"]
    width = profile["width"]
    chm = np.zeros((height, width), dtype=np.float32)

    # Place Gaussian peaks for each tree
    for i in range(n_trees):
        cx_pix = int((i + 1) * width / (n_trees + 1))
        cy_pix = height // 2 + int(RNG.uniform(-10, 10))
        tree_ht = float(RNG.uniform(15, 35))
        sigma = float(RNG.uniform(2, 4))

        yy, xx = np.mgrid[0:height, 0:width]
        gauss = tree_ht * np.exp(
            -((xx - cx_pix) ** 2 + (yy - cy_pix) ** 2) / (2 * sigma ** 2)
        )
        chm = np.maximum(chm, gauss.astype(np.float32))

    path = output_dir / "synthetic_chm.tif"
    write_raster(path, chm, profile)
    return path


def generate_synthetic_dtm(output_dir: Path) -> Path:
    """Create a flat synthetic DTM at elevation 500 m."""
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = make_profile(AOI_BOUNDS, 1.0)
    dtm = np.full((profile["height"], profile["width"]), 500.0, dtype=np.float32)
    path = output_dir / "synthetic_dtm.tif"
    write_raster(path, dtm, profile)
    return path


# ---------------------------------------------------------------------------
# P1 — Burn severity rasters
# ---------------------------------------------------------------------------


def generate_synthetic_burn_rasters(output_dir: Path) -> dict[str, Path]:
    """Create synthetic pre/post NIR and SWIR rasters for burn severity testing.

    Returns dict of paths: {pre_nir, pre_swir, post_nir, post_swir,
    fire_perimeter}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    resolution = 10.0  # Sentinel-2 native
    profile = make_profile(AOI_BOUNDS, resolution)
    h, w = profile["height"], profile["width"]

    # Pre-fire: healthy vegetation (high NIR, low SWIR)
    pre_nir = RNG.uniform(0.3, 0.5, (h, w)).astype(np.float32)
    pre_swir = RNG.uniform(0.05, 0.15, (h, w)).astype(np.float32)

    # Post-fire: burn scar in upper half
    post_nir = pre_nir.copy()
    post_swir = pre_swir.copy()
    burn_rows = h // 2
    post_nir[:burn_rows, :] = RNG.uniform(0.05, 0.15, (burn_rows, w))
    post_swir[:burn_rows, :] = RNG.uniform(0.2, 0.4, (burn_rows, w))

    paths = {}
    for name, data in [
        ("pre_nir", pre_nir),
        ("pre_swir", pre_swir),
        ("post_nir", post_nir),
        ("post_swir", post_swir),
    ]:
        p = output_dir / f"{name}.tif"
        write_raster(p, data, profile)
        paths[name] = p

    # Fire perimeter polygon
    perim = gpd.GeoDataFrame(
        {"fire_name": ["Synthetic Fire"]},
        geometry=[box(AOI_XMIN, AOI_YMIN, AOI_XMAX, AOI_YMAX)],
        crs=CRS_3310,
    )
    perim_path = output_dir / "fire_perimeter.gpkg"
    perim.to_file(perim_path, driver="GPKG")
    paths["fire_perimeter"] = perim_path

    return paths


# ---------------------------------------------------------------------------
# P4 — Species occurrences and environmental predictors
# ---------------------------------------------------------------------------


def generate_synthetic_occurrences(
    output_dir: Path, n_presence: int = 50, n_background: int = 200,
) -> Path:
    """Create synthetic species occurrence points."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Presence points clustered in center of AOI
    px = RNG.normal(AOI_XMIN + 100, 30, n_presence)
    py = RNG.normal(AOI_YMIN + 100, 30, n_presence)

    gdf = gpd.GeoDataFrame(
        {
            "species": "Pekania pennanti",
            "source": RNG.choice(["GBIF", "CNDDB"], n_presence),
            "presence": 1,
        },
        geometry=[Point(x, y) for x, y in zip(px, py)],
        crs=CRS_3310,
    )
    path = output_dir / "occurrences.gpkg"
    gdf.to_file(path, driver="GPKG")
    return path


def generate_synthetic_predictors(output_dir: Path) -> dict[str, Path]:
    """Create synthetic environmental predictor rasters."""
    output_dir.mkdir(parents=True, exist_ok=True)
    resolution = 30.0
    profile = make_profile(AOI_BOUNDS, resolution)
    h, w = profile["height"], profile["width"]

    paths = {}
    predictors = {
        "elevation": RNG.uniform(500, 2500, (h, w)),
        "slope": RNG.uniform(0, 60, (h, w)),
        "tpi": RNG.normal(0, 5, (h, w)),
        "bio1_mean_temp": RNG.uniform(5, 20, (h, w)),
        "bio12_annual_precip": RNG.uniform(400, 2000, (h, w)),
        "canopy_cover": RNG.uniform(0, 100, (h, w)),
    }

    for name, data in predictors.items():
        p = output_dir / f"{name}.tif"
        write_raster(p, data.astype(np.float32), profile)
        paths[name] = p

    return paths


# ---------------------------------------------------------------------------
# P2 — GeoPackage with spatial layers
# ---------------------------------------------------------------------------


def generate_synthetic_geopackage(output_dir: Path) -> Path:
    """Create a synthetic GeoPackage with forestry management layers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gpkg_path = output_dir / "forest_management.gpkg"

    # Harvest units
    units = gpd.GeoDataFrame(
        {
            "unit_id": [1, 2, 3],
            "unit_name": ["Unit A", "Unit B", "Unit C"],
            "acres": [40.0, 25.0, 60.0],
            "prescription": ["clearcut", "selection", "shelterwood"],
        },
        geometry=[
            box(AOI_XMIN, AOI_YMIN, AOI_XMIN + 80, AOI_YMIN + 80),
            box(AOI_XMIN + 90, AOI_YMIN, AOI_XMIN + 140, AOI_YMIN + 80),
            box(AOI_XMIN + 150, AOI_YMIN, AOI_XMAX, AOI_YMIN + 80),
        ],
        crs=CRS_3310,
    )
    units.to_file(gpkg_path, layer="harvest_units", driver="GPKG")

    # Streams
    from shapely.geometry import LineString
    streams = gpd.GeoDataFrame(
        {
            "stream_id": [1, 2],
            "stream_class": ["I", "III"],
            "name": ["Bear Creek", "Unnamed Trib"],
        },
        geometry=[
            LineString([(AOI_XMIN, AOI_YMIN + 50), (AOI_XMAX, AOI_YMIN + 50)]),
            LineString([(AOI_XMIN + 100, AOI_YMIN), (AOI_XMIN + 100, AOI_YMAX)]),
        ],
        crs=CRS_3310,
    )
    streams.to_file(gpkg_path, layer="streams", driver="GPKG")

    # Sensitive habitats
    habitats = gpd.GeoDataFrame(
        {
            "habitat_id": [1],
            "species": ["Pekania pennanti"],
            "status": ["candidate"],
        },
        geometry=[box(AOI_XMIN + 60, AOI_YMIN + 20, AOI_XMIN + 120, AOI_YMIN + 60)],
        crs=CRS_3310,
    )
    habitats.to_file(gpkg_path, layer="sensitive_habitats", driver="GPKG")

    return gpkg_path


# ---------------------------------------------------------------------------
# Main — generate all synthetic data
# ---------------------------------------------------------------------------


def generate_all(output_dir: str | Path) -> dict[str, Path]:
    """Generate all synthetic datasets for testing.

    Parameters
    ----------
    output_dir : str or Path
        Root output directory.

    Returns
    -------
    dict[str, Path]
        Mapping of dataset names to file paths.
    """
    output_dir = Path(output_dir)
    paths = {}

    # P3 — LiDAR
    p3_dir = output_dir / "p3"
    paths["lidar_csv"] = generate_synthetic_lidar(p3_dir)
    paths["cruise_plots"] = generate_synthetic_cruise_plots(p3_dir)
    paths["chm"] = generate_synthetic_chm(p3_dir)
    paths["dtm"] = generate_synthetic_dtm(p3_dir)

    # P1 — Burn severity
    p1_dir = output_dir / "p1"
    paths.update(generate_synthetic_burn_rasters(p1_dir))

    # P4 — Habitat suitability
    p4_dir = output_dir / "p4"
    paths["occurrences"] = generate_synthetic_occurrences(p4_dir)
    paths.update(generate_synthetic_predictors(p4_dir))

    # P2 — GeoPackage
    p2_dir = output_dir / "p2"
    paths["geopackage"] = generate_synthetic_geopackage(p2_dir)

    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic test data")
    parser.add_argument(
        "--output-dir",
        default="tests/fixtures",
        help="Output directory for synthetic data",
    )
    args = parser.parse_args()
    results = generate_all(args.output_dir)
    for name, path in sorted(results.items()):
        print(f"  {name}: {path}")
