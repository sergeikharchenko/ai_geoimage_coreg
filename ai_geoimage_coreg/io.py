import rasterio
import numpy as np
import torch
from rasterio.windows import Window
from osgeo import gdal

def read_and_resize(path, max_size):
    """Reads a GeoTIFF and resizes it to max_size on the long edge."""
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        scale = max_size / max(h, w)
        
        new_h = int((h * scale) // 8 * 8)
        new_w = int((w * scale) // 8 * 8)
        
        print(f"Loading {path}: Original {w}x{h} -> Resized {new_w}x{new_h}")
        
        img = src.read(
            1, 
            out_shape=(new_h, new_w),
            resampling=rasterio.enums.Resampling.bilinear
        )
    return img, (w, h), (new_w, new_h)

def read_tile(path, x, y, w, h):
    """Reads a specific window (crop) from the TIFF."""
    with rasterio.open(path) as src:
        # Check boundaries
        if x + w > src.width or y + h > src.height:
            return None
            
        window = Window(x, y, w, h)
        img = src.read(1, window=window)
        
        # Skip empty areas (black borders)
        if np.mean(img) < 10:
            return None
    return img

def to_tensor(img_np, device):
    """Normalizes and converts numpy array to torch tensor (B, C, H, W)."""
    return torch.from_numpy(img_np).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)

def warp_image(input_path, output_path, reference_path, gcps, method='poly', poly_order=3):
    """
    Warps the image using GDAL based on Ground Control Points (GCPs).
    method: 'poly' or 'tps'
    """
    # Get projection from reference
    ds_ref = gdal.Open(reference_path)
    target_wkt = ds_ref.GetProjection()
    ds_ref = None

    # Convert GCP dictionaries to GDAL objects
    gdal_gcps = [gdal.GCP(p['mapX'], p['mapY'], 0, p['pixelX'], p['pixelY']) for p in gcps]

    src_ds = gdal.Open(input_path)
    temp_vrt = gdal.Translate('', src_ds, format='VRT', outputSRS=target_wkt, GCPs=gdal_gcps)

    # Warp options
    options = {
        'format': 'GTiff',
        'dstSRS': target_wkt,
        'dstAlpha': True,
        'resampleAlg': gdal.GRA_Cubic,
        'creationOptions': ["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_NEEDED", "NUM_THREADS=ALL_CPUS"],
        'errorThreshold': 0.125 if method == 'poly' else 0.5
    }

    if method == 'tps':
        options['tps'] = True
    else:
        options['tps'] = False
        options['polynomialOrder'] = poly_order

    print(f"Starting GDAL Warp ({method})...")
    warp_opts = gdal.WarpOptions(**options)
    gdal.Warp(output_path, temp_vrt, options=warp_opts)
    print(f"Saved: {output_path}")