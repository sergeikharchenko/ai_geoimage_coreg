# AI GeoImage Coregistration

**A Python package for automated georeferencing of historical satellite imagery (e.g., Hexagon KH-9) using Deep Learning.**

This tool uses state-of-the-art feature matching algorithms (**LoFTR** and **SuperPoint+SuperGlue**) to find corresponding points between a raw, unreferenced image and a modern georeferenced map. It then automatically warps the raw image using Polynomial or Thin Plate Spline (TPS) transformations.

## Features

- **Automated GCP Generation**: No manual clicking required. Uses AI to find thousands of tie points.
- **Robust to Temporal Changes**: Handles season changes, urban development, and appearance differences between 1970s and 2020s imagery.
- **Large Image Support**: Automatically tiles gigapixel images to process them piece by piece.
- **Smart Filtering**: Uses RANSAC and Spatial Grid Binning to remove outliers and ensure geometric consistency.
- **Warping**: Outputs both Polynomial (Order 2/3) and TPS (Thin Plate Spline) GeoTIFFs.

---

## Prerequisites (Important)

Before installing this package, **GDAL must be installed** on your system.

### Ubuntu / Debian
```bash
sudo apt update
sudo apt install libgdal-dev gdal-bin
```

### Conda (Recommended for Windows/Linux/Mac)
```bash
conda install -c conda-forge gdal
```

---

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/sergeikharchenko/ai_geoimage_coreg.git
```

### Optional: SuperGlue Setup
By default, the tool uses **LoFTR**, which downloads its own weights automatically.
If you want to use the **SuperGlue** algorithm (high precision for detailed structures), you must clone the official repository due to license restrictions:

```bash
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
export PYTHONPATH=$PYTHONPATH:$(pwd)/SuperGluePretrainedNetwork
```

---

## Data Preparation (Reference Image)

To get the best results, you need a good **Reference Image**.

1.  **Format**: It must be a **GeoTIFF** (`.tif`) with a valid coordinate system (CRS).
2.  **Resolution**: Try to match the resolution of your input image. For Hexagon (~1-2m/px), standard satellite imagery works well.
3.  **Source**: The easiest way to get a reference is using **QGIS**:
    *   Open QGIS.
    *   Add a basemap (e.g., *ESRI Satellite*, *Google Satellite*, or *Bing Aerial*).
    *   Zoom to the area covering your historical image.
    *   Go to **Project -> Import/Export -> Export Map to Image**.
    *   **Crucial**: Check "Append georeference information (embedded)" or "Save World File".
    *   Increase the resolution (DPI) until the "Output width/height" in pixels is sufficient (e.g., 5000-10000 px).
    *   Save as `.tif`.

---

## Usage

Run the tool from the command line:

```bash
geobind --input /path/to/raw_hexagon.tif --ref /path/to/reference_map.tif --output result_name
```

### Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--input` | **Required** | Path to the raw, unreferenced historical image (TIFF). |
| `--ref` | **Required** | Path to the georeferenced modern image (TIFF). |
| `--algo` | `loftr` | Matching algorithm: `loftr` (robust) or `superglue` (precise). |
| `--output` | `georef` | Prefix for output files. |
| `--tile-size` | `800` | Processing tile size in pixels. Reduce to 600 if running out of GPU memory. |
| `--step-size` | `400` | Sliding window step. Lower value = more points but slower processing. |
| `--margin` | `400` | Search margin on reference image. Increase if initial coarse alignment is poor. |
| `--conf` | `0.75` | Neural network confidence threshold (0.0 - 1.0). |
| `--grid` | `100` | Spatial binning size (px). Keeps only the best point per grid cell to avoid clustering. |
| `--ransac` | `500.0` | RANSAC threshold (meters/pixels). Increase for mountainous terrain or high distortion. |

### Examples

**1. Standard run (recommended for most cases):**
```bash
geobind --input KH9_strip.tif --ref modern_map.tif
```

**2. High Precision (slower):**
Use a smaller step to find more points and a finer grid for better distribution. Good for detailed urban areas.
```bash
geobind --input KH9.tif --ref modern.tif --step-size 200 --grid 50
```

**3. Difficult Terrain (Mountains/Distortion):**
If valid points are being deleted because the image is very warped (panoramic distortion), increase the RANSAC threshold.
```bash
geobind --input KH9.tif --ref modern.tif --ransac 1000.0
```
---

## Algorithms

*   **LoFTR (Detector-free)**: Recommended default. It works exceptionally well on images with significant differences (e.g., 50-year gap). It produces dense matches even in low-texture areas.
*   **SuperPoint + SuperGlue**: Excellent for urban areas with distinct corners and buildings. Requires separate installation (see above).

## License

This code is provided as-is.
*   **LoFTR** is Apache 2.0 licensed.
*   **SuperGlue** has a non-commercial license (CC-BY-NC-4.0).

---
*Author: Sergei Kharchenko*
