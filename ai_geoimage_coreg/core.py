import torch
import pandas as pd
import numpy as np
import argparse
from .io import read_and_resize, read_tile, to_tensor, warp_image
from .matching import LoFTRMatcher, SuperGlueWrapper
from .geometry import compute_coarse_homography, project_center, filter_points_grid, filter_points_ransac
from .utils import generate_tiles

def run_pipeline(path_hex, path_ref, algorithm='loftr', step_size=400, tile_size=800, output_prefix='result'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- STEP 1: COARSE MATCHING ---
    print("--- Step 1: Coarse Matching (Global) ---")
    img_hex_small, size_hex_orig, size_hex_small = read_and_resize(path_hex, 1200)
    img_ref_small, size_ref_orig, size_ref_small = read_and_resize(path_ref, 1200)

    # Use LoFTR for coarse step always (it's robust)
    coarse_matcher = LoFTRMatcher(device)
    t_hex = to_tensor(img_hex_small, device)
    t_ref = to_tensor(img_ref_small, device)

    mkpts0, mkpts1, _ = coarse_matcher.match(t_hex, t_ref)
    
    if len(mkpts0) < 10:
        raise ValueError("Not enough coarse matches found! Try rotating the image.")

    H, s_hex, s_ref = compute_coarse_homography(mkpts0, mkpts1, size_hex_orig, size_hex_small, size_ref_orig, size_ref_small)
    print("Coarse Homography calculated.")

    # --- STEP 2: FINE MATCHING (SLIDING WINDOW) ---
    print(f"--- Step 2: Fine Matching using {algorithm.upper()} ---")
    
    if algorithm == 'superglue':
        matcher = SuperGlueWrapper(device)
        conf_thr = 0.9
    else:
        matcher = coarse_matcher # Reuse LoFTR
        conf_thr = 0.75

    final_gcp = []
    
    # Get image dimensions from file directly without loading full image
    import rasterio
    with rasterio.open(path_hex) as src:
        h_full, w_full = src.height, src.width
    with rasterio.open(path_ref) as src:
        ref_w, ref_h = src.width, src.height
        # Need transform to convert pixel -> geo
        ref_transform = src.transform

    search_margin = 400

    # Iterate tiles
    for x_hex, y_hex, w, h in generate_tiles(w_full, h_full, tile_size, step_size):
        
        # 1. Project center
        center_x, center_y = x_hex + w/2, y_hex + h/2
        pred_x, pred_y = project_center(center_x, center_y, H, s_hex, s_ref)
        
        if not (0 <= pred_x < ref_w and 0 <= pred_y < ref_h):
            continue

        # 2. Read Crops
        img_hex_crop = read_tile(path_hex, x_hex, y_hex, w, h)
        if img_hex_crop is None: continue

        # Calculate ref window
        ref_size = tile_size + search_margin
        ref_x_start = int(pred_x - ref_size/2)
        ref_y_start = int(pred_y - ref_size/2)
        
        # Clamp
        ref_x_start = max(0, min(ref_x_start, ref_w - ref_size))
        ref_y_start = max(0, min(ref_y_start, ref_h - ref_size))

        img_ref_crop = read_tile(path_ref, ref_x_start, ref_y_start, ref_size, ref_size)
        if img_ref_crop is None: continue

        # 3. Match
        try:
            t_hex_c = to_tensor(img_hex_crop, device)
            t_ref_c = to_tensor(img_ref_crop, device)
            
            kpts0, kpts1, conf = matcher.match(t_hex_c, t_ref_c, conf_thr=conf_thr)
        except RuntimeError:
            torch.cuda.empty_cache()
            continue

        if len(kpts0) < 5: continue

        # 4. Globalize coordinates
        for i in range(len(kpts0)):
            gx_hex = x_hex + kpts0[i][0]
            gy_hex = y_hex + kpts0[i][1]
            
            gx_ref = ref_x_start + kpts1[i][0]
            gy_ref = ref_y_start + kpts1[i][1]
            
            # Pixel to Geo
            geo_x, geo_y = rasterio.transform.xy(ref_transform, gy_ref, gx_ref, offset='center')

            final_gcp.append({
                'mapX': geo_x,
                'mapY': geo_y,
                'pixelX': gx_hex,
                'pixelY': gy_hex,
                'confidence': conf[i]
            })

        print(f"Tile {x_hex},{y_hex}: {len(kpts0)} matches.")
        torch.cuda.empty_cache()

    print(f"Total raw points: {len(final_gcp)}")
    if not final_gcp:
        print("No points found.")
        return

    df = pd.DataFrame(final_gcp)
    df.to_csv(f'{output_prefix}_raw.csv', index=False)

    # --- STEP 3: FILTERING ---
    print("--- Step 3: Filtering ---")
    df_grid = filter_points_grid(df, grid_size=100)
    df_clean = filter_points_ransac(df_grid, threshold=500.0)
    
    print(f"Clean points remaining: {len(df_clean)}")
    df_clean.to_csv(f'{output_prefix}_clean.csv', index=False)

    # --- STEP 4: WARPING ---
    print("--- Step 4: Warping ---")
    
    # Convert clean DF to list of dicts
    gcps = df_clean.to_dict('records')
    
    # Polynomial (Robust)
    warp_image(path_hex, f'{output_prefix}_poly.tif', path_ref, gcps, method='poly')
    
    # TPS (Elastic)
    warp_image(path_hex, f'{output_prefix}_tps.tif', path_ref, gcps, method='tps')
    
    print("Done.")

def main_cli():
    parser = argparse.ArgumentParser(description="Geobind: AI Georeferencing Tool")
    parser.add_argument('--input', required=True, help='Path to unreferenced image (Hexagon)')
    parser.add_argument('--ref', required=True, help='Path to georeferenced image')
    parser.add_argument('--algo', default='loftr', choices=['loftr', 'superglue'], help='Matching algorithm')
    parser.add_argument('--output', default='georef', help='Output filename prefix')
    
    args = parser.parse_args()
    
    run_pipeline(args.input, args.ref, args.algo, output_prefix=args.output)