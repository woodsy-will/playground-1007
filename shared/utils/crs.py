"""CRS validation and reprojection utilities.

Default CRS: EPSG:3310 (California Albers NAD 83)
"""

from pyproj import CRS

DEFAULT_CRS = CRS.from_epsg(3310)


def validate_crs(crs_input: str | int | CRS, expected: CRS = DEFAULT_CRS) -> bool:
    """Check if a CRS matches the expected CRS.

    Parameters
    ----------
    crs_input : str, int, or CRS
        CRS to validate.
    expected : CRS
        Expected CRS. Defaults to EPSG:3310.

    Returns
    -------
    bool
        True if CRS matches expected.
    """
    return CRS(crs_input) == expected
