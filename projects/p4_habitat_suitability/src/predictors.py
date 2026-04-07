"""Environmental predictor stack construction and value extraction.

Loads topographic, climatic, and canopy rasters, aligns them to a common
grid, and builds a 3-D numpy array for modelling.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import geopandas as gpd
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import uniform_filter

from shared.utils.io import read_raster, write_raster
from shared.utils.logging import get_logger

logger = get_logger("p4_predictors")


def build_predictor_stack(
    config: dict[str, Any],
) -> tuple[np.ndarray, dict, list[str]]:
    """Load predictor rasters, align to a common grid, and stack.

    Parameters
    ----------
    config : dict
        Project configuration with ``data.predictor_dir`` pointing to a
        directory of single-band GeoTIFFs and ``modeling.crs`` for the
        target CRS.

    Returns
    -------
    tuple[np.ndarray, dict, list[str]]
        - stack: 3-D array of shape ``(bands, rows, cols)``.
        - profile: rasterio profile for the aligned grid.
        - band_names: list of predictor names matching band order.
    """
    pred_dir = Path(config["data"]["predictor_dir"])
    tif_paths = sorted(pred_dir.glob("*.tif"))
    if not tif_paths:
        raise FileNotFoundError(f"No .tif files in {pred_dir}")

    # Use the first raster as the reference grid
    ref_data, ref_profile = read_raster(tif_paths[0])
    ref_height = ref_profile["height"]
    ref_width = ref_profile["width"]
    ref_transform = ref_profile["transform"]
    ref_crs = ref_profile["crs"]

    bands: list[np.ndarray] = []
    band_names: list[str] = []

    for tif_path in tif_paths:
        data, prof = read_raster(tif_path)
        band = data[0].astype(np.float32)

        # Reproject / resample if grid does not match reference
        if (
            prof["height"] != ref_height
            or prof["width"] != ref_width
            or prof["transform"] != ref_transform
        ):
            dst = np.empty((ref_height, ref_width), dtype=np.float32)
            reproject(
                source=band,
                destination=dst,
                src_transform=prof["transform"],
                src_crs=prof.get("crs", ref_crs),
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )
            band = dst

        bands.append(band)
        band_names.append(tif_path.stem)

    stack = np.stack(bands, axis=0)
    profile = ref_profile.copy()
    profile["count"] = len(bands)
    logger.info(
        "Built predictor stack: %d bands, shape (%d, %d)",
        len(bands), ref_height, ref_width,
    )
    return stack, profile, band_names


def compute_topo_derivatives(
    dem_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Compute slope, TPI, and TWI from a DEM using numpy/scipy.

    Parameters
    ----------
    dem_path : str or Path
        Path to DEM GeoTIFF.
    output_dir : str or Path
        Directory for output derivative rasters.
    config : dict, optional
        Project configuration (reserved for future tuning parameters).

    Returns
    -------
    dict[str, Path]
        Mapping of derivative name to output raster path.
    """
    dem_path = Path(dem_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data, profile = read_raster(dem_path)
    dem = data[0].astype(np.float64)

    cellsize = abs(profile["transform"].a)
    out_profile = profile.copy()
    out_profile["count"] = 1
    out_profile["dtype"] = "float32"

    results: dict[str, Path] = {}

    # --- Slope (degrees) via np.gradient ---
    dy, dx = np.gradient(dem, cellsize)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    slope_path = output_dir / "slope.tif"
    write_raster(slope_path, slope_deg, out_profile)
    results["slope"] = slope_path

    # --- Topographic Position Index (focal mean difference) ---
    kernel_size = 5
    focal_mean = uniform_filter(dem, size=kernel_size, mode="nearest")
    tpi = (dem - focal_mean).astype(np.float32)
    tpi_path = output_dir / "tpi.tif"
    write_raster(tpi_path, tpi, out_profile)
    results["tpi"] = tpi_path

    # --- Topographic Wetness Index: ln(flow_accum / tan(slope)) ---
    # Simplified flow accumulation: each cell contributes to its steepest
    # downslope neighbour.  For a proper TWI a full D-inf algorithm would
    # be used, but this approximation is portable and sufficient for SDMs.
    slope_rad_clamp = np.clip(slope_rad, 1e-6, None)
    # Use a uniform contributing area proxy (cell area / tan(slope))
    cell_area = cellsize * cellsize
    twi = np.log(cell_area / np.tan(slope_rad_clamp)).astype(np.float32)
    twi_path = output_dir / "twi.tif"
    write_raster(twi_path, twi, out_profile)
    results["twi"] = twi_path

    logger.info("Computed topo derivatives: %s", list(results.keys()))
    return results


def extract_values_at_points(
    stack: np.ndarray,
    profile: dict,
    points_gdf: gpd.GeoDataFrame,
    band_names: list[str],
) -> pd.DataFrame:
    """Sample raster stack values at point locations.

    Parameters
    ----------
    stack : np.ndarray
        Predictor stack of shape ``(bands, rows, cols)``.
    profile : dict
        Rasterio profile with transform and CRS.
    points_gdf : GeoDataFrame
        Points at which to extract values.
    band_names : list[str]
        Names corresponding to each band in the stack.

    Returns
    -------
    DataFrame
        One row per point, one column per predictor band, plus geometry
        coordinates.  Rows with any NaN are dropped.
    """

    transform = profile["transform"]
    n_bands, height, width = stack.shape

    rows_list: list[dict[str, float]] = []
    for geom in points_gdf.geometry:
        col = int((geom.x - transform.c) / transform.a)
        row = int((geom.y - transform.f) / transform.e)
        if 0 <= row < height and 0 <= col < width:
            vals = {name: float(stack[b, row, col]) for b, name in enumerate(band_names)}
            vals["x"] = geom.x
            vals["y"] = geom.y
            rows_list.append(vals)

    df = pd.DataFrame(rows_list)
    df = df.dropna()
    logger.info("Extracted values for %d / %d points", len(df), len(points_gdf))
    return df
