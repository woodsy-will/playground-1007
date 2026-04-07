"""Sierra Nevada allometric equations for tree biometrics.

Sources:
    - Pillsbury, N.H., & Kirkley, M.L. (1984). PNW-GTR-414.
    - FIA regional volume equations.
"""

import numpy as np
from numpy.typing import ArrayLike


# Basal area constant: BA (sq ft) = BA_CONSTANT * DBH (inches)^2
BA_CONSTANT = 0.005454


def basal_area_sqft(dbh_inches: ArrayLike) -> np.ndarray:
    """Compute basal area in square feet from DBH in inches.

    Parameters
    ----------
    dbh_inches : array-like
        Diameter at breast height in inches.

    Returns
    -------
    np.ndarray
        Basal area in square feet.
    """
    dbh = np.asarray(dbh_inches, dtype=np.float64)
    return BA_CONSTANT * dbh ** 2


def dbh_from_crown_diameter(crown_diameter_m: ArrayLike) -> np.ndarray:
    """Estimate DBH (inches) from crown diameter (meters).

    Uses the inverse of Pillsbury & Kirkley (1984) linear crown-DBH
    relationship for Sierra Nevada mixed-conifer:
        crown_diameter_ft = a + b * DBH_inches
    With published coefficients a=3.0, b=0.25 (generic mixed-conifer).

    Parameters
    ----------
    crown_diameter_m : array-like
        Crown diameter in meters.

    Returns
    -------
    np.ndarray
        Estimated DBH in inches.
    """
    cd_m = np.asarray(crown_diameter_m, dtype=np.float64)
    cd_ft = cd_m * 3.28084  # meters to feet
    a, b = 3.0, 0.25
    dbh = (cd_ft - a) / b
    return np.maximum(dbh, 0.0)


def stem_volume_cuft(dbh_inches: ArrayLike, height_ft: ArrayLike) -> np.ndarray:
    """Estimate total stem volume (cubic feet) from DBH and height.

    Uses the combined-variable equation form common to FIA regional
    volume tables for Sierra Nevada conifers:
        volume = b1 * DBH^2 * height
    With b1 = 0.002 (approximate for mixed-conifer, Pillsbury & Kirkley).

    Parameters
    ----------
    dbh_inches : array-like
        Diameter at breast height in inches.
    height_ft : array-like
        Total tree height in feet.

    Returns
    -------
    np.ndarray
        Estimated stem volume in cubic feet.
    """
    dbh = np.asarray(dbh_inches, dtype=np.float64)
    ht = np.asarray(height_ft, dtype=np.float64)
    b1 = 0.002
    vol = b1 * dbh ** 2 * ht
    return np.maximum(vol, 0.0)
