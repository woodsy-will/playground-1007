# P1 Technical Report: Automated Post-Wildfire Burn Severity & Recovery Tracker

## 1. Objective & Study Area Context

This project implements an automated, reproducible pipeline for quantifying
wildfire burn severity and tracking post-fire vegetation recovery across the
Sierra Nevada bioregion of California. The system ingests multi-temporal
Sentinel-2 imagery, computes standard spectral burn indices, classifies
severity into ecologically meaningful categories, and models long-term
vegetation recovery trajectories.

**Study area:** Sierra Nevada mountain range, California. The region
experiences frequent mixed-severity wildfires in conifer-dominated forests
(Pinus ponderosa, Abies concolor, Pseudotsuga menziesii) with fire return
intervals of 5-30 years depending on elevation and aspect.

**Coordinate reference system:** EPSG:3310 (California Albers Equal Area,
NAD 83) ensures equal-area analysis suitable for area-based severity
statistics.

## 2. Data Sources

### 2.1 Sentinel-2 L2A Imagery

- **Platform:** ESA Copernicus Sentinel-2A/B
- **Product level:** L2A (bottom-of-atmosphere reflectance via Sen2Cor)
- **Spatial resolution:** 10 m (visible/NIR), 20 m (SWIR, red-edge, SCL)
- **Bands used:**
  - B8A (Vegetation Red-Edge, 865 nm, 20 m) -- Near-Infrared
  - B12 (SWIR, 2190 nm, 20 m) -- Short-Wave Infrared
  - SCL (Scene Classification Layer) -- Cloud/shadow mask
- **Cloud cover filter:** <= 20% scene-level cloud cover
- **Seasonal window:** June-October compositing to minimize phenological
  variation and snow cover
- **Access:** Copernicus Data Space Ecosystem STAC API

### 2.2 Fire Perimeter Data

- **Source:** CAL FIRE FRAP (Fire and Resource Assessment Program)
- **Format:** GeoPackage with fire name, discovery date, containment date,
  and perimeter geometry
- **Supplement:** NIFC InciWeb perimeters for active fires

## 3. Methods

### 3.1 Normalized Burn Ratio (NBR)

The Normalized Burn Ratio exploits the contrast between NIR reflectance
(high in healthy vegetation) and SWIR reflectance (high in bare/burned
soil):

    NBR = (NIR - SWIR) / (NIR + SWIR)

Healthy vegetation produces NBR values near +0.6 to +0.9, while severely
burned areas drop to -0.2 to +0.1 (Key & Benson, 2006).

### 3.2 Differenced NBR (dNBR)

Temporal change is captured by differencing pre-fire and post-fire NBR:

    dNBR = NBR_pre - NBR_post

Positive dNBR indicates vegetation loss (burn). Negative values may
indicate increased vegetation growth.

### 3.3 Relativized Burn Ratio (RBR)

RBR normalises dNBR by pre-fire conditions, reducing bias in areas with
sparse pre-fire vegetation:

    RBR = dNBR / (NBR_pre + 1.001)

This index was recommended by Parks et al. (2014) for heterogeneous
landscapes.

### 3.4 Severity Classification

dNBR is classified into five categories using thresholds calibrated for
Sierra Nevada mixed-conifer forests (Miller & Thode, 2007):

| Class          | dNBR Range     | Code |
|----------------|---------------|------|
| Unburned       | -0.10 to 0.10 | 0    |
| Low            | 0.10 to 0.27  | 1    |
| Moderate-Low   | 0.27 to 0.44  | 2    |
| Moderate-High  | 0.44 to 0.66  | 3    |
| High           | 0.66 to 1.30  | 4    |

These thresholds are configurable via the project YAML configuration.

### 3.5 Cloud Masking

The Sentinel-2 Scene Classification Layer (SCL) is used to mask
cloud-contaminated pixels before index computation. The following SCL
classes are masked:

- Class 3: Cloud shadow
- Class 8: Cloud medium probability
- Class 9: Cloud high probability
- Class 10: Thin cirrus

### 3.6 Vegetation Recovery Modeling

Post-fire recovery is tracked using annual NDVI composites. Mean NDVI is
computed per severity class for each post-fire year, and an exponential
recovery model is fitted:

    NDVI(t) = a * (1 - exp(-b * t)) + c

Where:
- `a` = asymptotic recovery amplitude
- `b` = recovery rate constant
- `c` = baseline NDVI (immediate post-fire)
- `t` = years since fire

Time-to-90%-recovery is estimated as t_90 = -ln(0.1) / b.

The model is fitted via non-linear least squares (scipy.optimize.curve_fit)
independently for each severity class. This allows comparison of recovery
trajectories -- high-severity areas are expected to recover more slowly than
low-severity areas.

## 4. Dashboard Design

An interactive Plotly Dash dashboard provides three views:

1. **Severity Map** -- Heatmap of classified burn severity with a
   five-colour palette (green to red).
2. **Recovery Time Series** -- Line chart of mean NDVI by severity class
   over post-fire years, with standard deviation bands.
3. **Summary Statistics Panel** -- Table of pixel counts and area
   percentages per severity class.

The dashboard is served locally and reads outputs directly from the
processed data directory.

## 5. Pipeline Architecture

The pipeline is orchestrated by `pipeline.py` with the following stages:

1. **Acquisition** -- STAC search and band download (or synthetic fallback)
2. **Preprocessing** -- SCL cloud masking, reprojection to EPSG:3310, clip
   to fire perimeter
3. **Severity Analysis** -- NBR, dNBR, RBR computation and classification
4. **Recovery Tracking** -- Annual NDVI time series, exponential model fit
5. **Output** -- GeoTIFF severity map, CSV time series, CSV model params

All stages are configuration-driven and use shared I/O utilities for
consistent CRS handling and file format defaults.

## 6. Key References

- Key, C.H. & Benson, N.C. (2006). Landscape Assessment (LA): Sampling
  and Analysis Methods. USDA Forest Service General Technical Report
  RMRS-GTR-164-CD.

- Miller, J.D. & Thode, A.E. (2007). Quantifying burn severity in a
  heterogeneous landscape with a relative version of the delta Normalized
  Burn Ratio (dNBR). Remote Sensing of Environment, 109(1), 66-80.

- Parks, S.A., Dillon, G.K., & Miller, C. (2014). A New Metric for
  Quantifying Burn Severity: The Relativized Burn Ratio. Remote Sensing,
  6(3), 1827-1844.

- ESA (2021). Sentinel-2 User Handbook. European Space Agency.

- Monitoring Trends in Burn Severity (MTBS). https://www.mtbs.gov/
