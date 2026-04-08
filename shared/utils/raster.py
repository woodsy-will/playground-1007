"""Raster processing utilities — reproject, resample, mosaic, mask.

Used across P1 (burn severity), P3 (CHM), and P4 (predictor stacks).
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

from shared.utils.crs import DEFAULT_CRS


def reproject_raster(
    src_path: str | Path,
    dst_path: str | Path,
    dst_crs: str | int = DEFAULT_CRS,
    resolution: float | None = None,
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    """Reproject a raster to a target CRS and optional resolution.

    Parameters
    ----------
    src_path : str or Path
        Input raster path.
    dst_path : str or Path
        Output raster path.
    dst_crs : str or int
        Target CRS. Default EPSG:3310.
    resolution : float, optional
        Target pixel size. If None, auto-computed.
    resampling : Resampling
        Resampling method.

    Returns
    -------
    Path
        Path to reprojected raster.
    """
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=resolution,
        )
        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                )
    return dst_path


def apply_nodata_mask(
    data: np.ndarray,
    nodata: float,
    fill: float = np.nan,
) -> np.ndarray:
    """Replace nodata values with a fill value.

    Parameters
    ----------
    data : np.ndarray
        Raster data array.
    nodata : float
        Nodata sentinel value to mask.
    fill : float
        Replacement value. Default ``np.nan``.

    Returns
    -------
    np.ndarray
        Masked copy of the array.
    """
    out = data.astype(np.float64, copy=True)
    out[data == nodata] = fill
    return out


def clip_raster_to_bounds(
    data: np.ndarray,
    profile: dict,
    bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, dict]:
    """Clip a raster array to spatial bounds using array slicing.

    Parameters
    ----------
    data : np.ndarray
        Raster array of shape (bands, rows, cols) or (rows, cols).
    profile : dict
        Rasterio profile with transform.
    bounds : tuple
        (xmin, ymin, xmax, ymax) in the raster's CRS.

    Returns
    -------
    tuple[np.ndarray, dict]
        Clipped (data, updated_profile).
    """
    transform = profile["transform"]
    xmin, ymin, xmax, ymax = bounds

    col_start = max(0, int((xmin - transform.c) / transform.a))
    col_end = min(profile["width"], int((xmax - transform.c) / transform.a))
    row_start = max(0, int((transform.f - ymax) / abs(transform.e)))
    row_end = min(profile["height"], int((transform.f - ymin) / abs(transform.e)))

    if data.ndim == 3:
        clipped = data[:, row_start:row_end, col_start:col_end]
    else:
        clipped = data[row_start:row_end, col_start:col_end]

    new_transform = rasterio.transform.from_origin(
        transform.c + col_start * transform.a,
        transform.f + row_start * transform.e,
        abs(transform.a),
        abs(transform.e),
    )

    new_profile = profile.copy()
    new_profile.update(
        width=col_end - col_start,
        height=row_end - row_start,
        transform=new_transform,
    )
    return clipped, new_profile


def resample_raster(
    src_path: str | Path,
    dst_path: str | Path,
    target_resolution: float,
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    """Resample a raster to a new resolution.

    Parameters
    ----------
    src_path : str or Path
        Input raster path.
    dst_path : str or Path
        Output raster path.
    target_resolution : float
        Target pixel size in CRS units.
    resampling : Resampling
        Resampling method.

    Returns
    -------
    Path
        Path to resampled raster.
    """
    return reproject_raster(
        src_path, dst_path,
        resolution=target_resolution,
        resampling=resampling,
    )
