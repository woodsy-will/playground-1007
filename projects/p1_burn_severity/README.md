# P1 — Automated Post-Wildfire Burn Severity & Recovery Tracker

## Objective

Automated pipeline ingesting pre/post-fire multispectral imagery to classify burn severity via dNBR and track vegetation recovery over consecutive years using NDVI time series.

## Data Requirements

- **Imagery:** Sentinel-2 L2A (Copernicus Data Space API) or Landsat 8/9 (USGS EarthExplorer)
- **Fire Perimeters:** MTBS or CAL FIRE FRAP
- **Management Units:** Optional — for zonal statistics overlay
- **CRS:** EPSG:3310

## Pipeline

```
Fire selection → API image acquisition (pre/post/annual)
                        │
              NBR = (NIR - SWIR2) / (NIR + SWIR2)
                        │
              dNBR = NBR_pre - NBR_post → severity classification
                        │
              Annual NDVI extraction → recovery curve fitting
                        │
              Zonal statistics by management unit
                        │
              Interactive dashboard + white paper
```

## Severity Thresholds (USGS/BAER)

| Class | dNBR Range |
|---|---|
| Unburned | -0.1 to 0.1 |
| Low | 0.1 to 0.27 |
| Moderate-Low | 0.27 to 0.44 |
| Moderate-High | 0.44 to 0.66 |
| High | 0.66 to 1.3 |

## Key References

- Key, C.H., & Benson, N.C. (2006). Landscape Assessment (LA). USDA Forest Service RMRS-GTR-164-CD.
- Parks, S.A., et al. (2019). Mean Composite Fire Severity Metrics. *Remote Sensing*, 11(15), 1735.

## Status

🔲 Not started
