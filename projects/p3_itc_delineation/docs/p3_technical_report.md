# P3 Technical Report: Individual Tree Crown Delineation & Biometrics

## 1. Objective

Develop an automated, LiDAR-based workflow to delineate individual tree crowns
(ITC) and estimate per-tree biometric attributes across Sierra Nevada
mixed-conifer forests.  The pipeline converts raw airborne LiDAR point clouds
into a georeferenced tree inventory that includes location, height, crown area,
estimated DBH, and stem volume for every detectable tree.

## 2. Study Area

The target landscape encompasses mixed-conifer stands in the central and
southern Sierra Nevada of California, spanning elevations of 1,200--2,400 m.
Dominant species include white fir (*Abies concolor*), ponderosa pine (*Pinus
ponderosa*), incense-cedar (*Calocedrus decurrens*), Douglas-fir (*Pseudotsuga
menziesii*), and red fir (*Abies magnifica*).  All spatial data are projected
to California Albers NAD83 (EPSG:3310).

## 3. Data Sources

### 3.1 LiDAR Point Clouds

Airborne LiDAR data are sourced from the USGS 3D Elevation Program (3DEP)
Quality Level 2 acquisitions:

| Parameter               | Specification              |
|-------------------------|----------------------------|
| Nominal pulse density   | >= 2 pts/m^2               |
| Vertical accuracy (RMSE)| <= 10 cm (open terrain)    |
| Horizontal accuracy     | <= 1/2 * NPS               |
| Classification          | ASPRS LAS 1.4 classes      |
| Format                  | LAZ (compressed LAS)       |

### 3.2 Field Cruise Data

Validation stems come from fixed-radius (1/10-acre) cruise plots with the
following attributes recorded per tree: stem location (GPS), DBH (inches),
total height (feet), species code, and diameter class (small / medium / large).

## 4. Methods

### 4.1 Ground Classification (SMRF)

Raw point clouds are classified using the Simple Morphological Filter (SMRF)
algorithm (Pingel et al. 2013), implemented via PDAL.  Default parameters:

- Cell size: 1.0 m
- Slope threshold: 0.15
- Maximum window size: 18.0 m

Ground-classified points (ASPRS class 2) are retained for DTM generation.

### 4.2 Digital Terrain Model (DTM)

Ground points are interpolated onto a 1.0 m regular grid using inverse-distance
weighting (IDW) via `writers.gdal`.  The resulting DTM serves as the base
elevation surface.

### 4.3 Canopy Height Model (CHM)

A Digital Surface Model (DSM) is derived from first-return points (maximum
value per cell).  The CHM is computed as:

    CHM = DSM - DTM

Negative values (artefacts from interpolation) are clamped to zero.  A Gaussian
smoothing kernel (sigma = 0.67 m) suppresses spurious micro-peaks while
preserving crown morphology.

### 4.4 Treetop Detection -- Variable-Window Local Maxima

Individual treetops are identified as local maxima in the CHM using a
height-adaptive sliding window.  The window radius scales linearly with
canopy height:

    window_m = 1.5 + 0.15 * height_m

This avoids over-segmentation in tall, broad-crowned trees and
under-segmentation in dense understory.  Only pixels exceeding the minimum
tree height threshold (5.0 m) are considered.

### 4.5 Crown Segmentation -- Marker-Controlled Watershed

Detected treetops seed a marker-controlled watershed segmentation on the
inverted CHM surface (Beucher & Lantuejoul 1979; Meyer & Beucher 1990).
The inverted CHM turns canopy peaks into basin bottoms, causing the
watershed to "flood" outward from each treetop marker until crown boundaries
(saddle points) are reached.

Segments smaller than 4.0 m^2 are discarded as likely noise.  Label regions
are vectorised into georeferenced polygons.

### 4.6 Allometric Estimation

Crown polygons are converted to biometric attributes using published
allometric relationships for Sierra Nevada mixed-conifer (Pillsbury &
Kirkley 1984):

- **Crown diameter** (m) is estimated from polygon area assuming a circular
  crown: `d = 2 * sqrt(area / pi)`.
- **DBH** (inches) is back-calculated from crown diameter using the inverse
  linear equation: `DBH = (crown_diameter_ft - 3.0) / 0.25`.
- **Stem volume** (cubic feet) uses the combined-variable form:
  `V = 0.002 * DBH^2 * height_ft`.
- **Basal area** (square feet) follows the standard formula:
  `BA = 0.005454 * DBH^2`.

## 5. Validation Approach

Predicted tree locations are matched to field-measured stems using
nearest-neighbour assignment with a maximum match distance of 3.0 m.
Metrics computed:

| Metric           | Definition                                      |
|------------------|-------------------------------------------------|
| Detection rate   | Proportion of reference stems with a match      |
| Omission rate    | 1 - detection rate                              |
| Commission rate  | Proportion of predicted trees without a match   |
| RMSE (height)    | Root-mean-square error of height (metres)       |
| RMSE (DBH)       | Root-mean-square error of DBH (inches)          |

Metrics are reported globally and stratified by species group and diameter
class.

## 6. Expected Deliverables

1. **Classified LiDAR** -- ground-classified LAZ files
2. **DTM** -- 1.0 m GeoTIFF digital terrain model
3. **CHM** -- 1.0 m smoothed GeoTIFF canopy height model
4. **Treetop points** -- GeoPackage of detected treetop locations with heights
5. **Crown polygons** -- GeoPackage of delineated crowns with areas and
   diameters
6. **Tree inventory** -- GeoPackage with full biometric attributes (height,
   DBH, volume, basal area) per tree
7. **Validation report** -- detection rates, RMSE tables by strata, and
   diagnostic plots
8. **Reproducible pipeline** -- config-driven Python modules callable via
   `run_pipeline(config_path)`

## 7. Key References

- Beucher, S., & Lantuejoul, C. (1979). Use of watersheds in contour
  detection. *International Workshop on Image Processing*.
- Meyer, F., & Beucher, S. (1990). Morphological segmentation. *Journal of
  Visual Communication and Image Representation*, 1(1), 21--46.
- Pillsbury, N. H., & Kirkley, M. L. (1984). Equations for total, wood,
  and saw-log volume for thirteen California hardwoods. *USDA Forest Service
  PNW Research Note PNW-414*.
- Pingel, T. J., Clarke, K. C., & McBride, W. A. (2013). An improved simple
  morphological filter for the terrain classification of airborne LIDAR data.
  *ISPRS Journal of Photogrammetry and Remote Sensing*, 77, 21--30.
- Popescu, S. C., & Wynne, R. H. (2004). Seeing the trees in the forest:
  using lidar and multispectral data fusion with local filtering and variable
  window size for estimating tree height. *Photogrammetric Engineering &
  Remote Sensing*, 70(5), 589--604.
- USGS 3D Elevation Program (3DEP). https://www.usgs.gov/3d-elevation-program
