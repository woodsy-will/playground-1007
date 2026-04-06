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
