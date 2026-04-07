"""I/O helpers for raster and vector data.

Wraps rasterio and geopandas for consistent CRS handling and format defaults.
"""

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds

from shared.utils.crs import DEFAULT_CRS


def read_raster(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read a raster file and return data array with metadata.

    Parameters
    ----------
    path : str or Path
        Path to raster file (GeoTIFF).

    Returns
    -------
    tuple[np.ndarray, dict]
        (data array [bands, rows, cols], rasterio profile dict).
    """
    with rasterio.open(path) as src:
        data = src.read()
        profile = dict(src.profile)
    return data, profile


def write_raster(
    path: str | Path,
    data: np.ndarray,
    profile: dict,
    **overrides: Any,
) -> Path:
    """Write a numpy array to a GeoTIFF raster.

    Parameters
    ----------
    path : str or Path
        Output file path.
    data : np.ndarray
        Array of shape (bands, rows, cols) or (rows, cols).
    profile : dict
        Rasterio profile dict (crs, transform, dtype, etc.).
    **overrides
        Additional profile keys to override (e.g., dtype, compress).

    Returns
    -------
    Path
        Path to written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        data = data[np.newaxis, ...]

    profile.update(
        driver="GTiff",
        count=data.shape[0],
        height=data.shape[1],
        width=data.shape[2],
        dtype=data.dtype.name,
    )
    profile.update(overrides)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def make_profile(
    bounds: tuple[float, float, float, float],
    resolution: float,
    crs: Any = DEFAULT_CRS,
    dtype: str = "float32",
    nodata: float = -9999.0,
) -> dict:
    """Create a rasterio profile from spatial bounds and resolution.

    Parameters
    ----------
    bounds : tuple
        (xmin, ymin, xmax, ymax) in CRS units.
    resolution : float
        Pixel size in CRS units (meters for EPSG:3310).
    crs : Any
        Coordinate reference system. Default EPSG:3310.
    dtype : str
        Numpy dtype string.
    nodata : float
        Nodata value.

    Returns
    -------
    dict
        Rasterio-compatible profile.
    """
    xmin, ymin, xmax, ymax = bounds
    width = int(np.ceil((xmax - xmin) / resolution))
    height = int(np.ceil((ymax - ymin) / resolution))
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

    return {
        "driver": "GTiff",
        "dtype": dtype,
        "width": width,
        "height": height,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }


def read_vector(path: str | Path, layer: str | None = None) -> gpd.GeoDataFrame:
    """Read a vector dataset (GeoPackage, Shapefile, GeoJSON).

    Parameters
    ----------
    path : str or Path
        Path to vector file.
    layer : str, optional
        Layer name for multi-layer formats (GeoPackage).

    Returns
    -------
    GeoDataFrame
    """
    return gpd.read_file(path, layer=layer)


def write_vector(
    gdf: gpd.GeoDataFrame,
    path: str | Path,
    layer: str | None = None,
    driver: str | None = None,
) -> Path:
    """Write a GeoDataFrame to disk.

    Parameters
    ----------
    gdf : GeoDataFrame
        Data to write.
    path : str or Path
        Output file path.
    layer : str, optional
        Layer name (for GeoPackage).
    driver : str, optional
        OGR driver. Auto-detected from extension if None.

    Returns
    -------
    Path
        Path to written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if driver is None:
        ext = path.suffix.lower()
        driver = {
            ".gpkg": "GPKG",
            ".shp": "ESRI Shapefile",
            ".geojson": "GeoJSON",
        }.get(ext, "GPKG")

    gdf.to_file(path, layer=layer, driver=driver)
    return path


def list_files(directory: str | Path, pattern: str = "*.laz") -> list[Path]:
    """List files matching a glob pattern in a directory.

    Parameters
    ----------
    directory : str or Path
        Directory to search.
    pattern : str
        Glob pattern. Default ``*.laz``.

    Returns
    -------
    list[Path]
        Sorted list of matching file paths.
    """
    return sorted(Path(directory).glob(pattern))
