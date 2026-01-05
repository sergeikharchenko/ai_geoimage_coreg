import torch
import pandas as pd
import numpy as np
import argparse
import rasterio
from .io import read_and_resize, read_tile, to_tensor, warp_image
from .matching import LoFTRMatcher, SuperGlueWrapper
from .geometry import compute_coarse_homography, project_center, filter_points_grid, filter_points_ransac
from .utils import generate_tiles

def run_pipeline(path_hex, path_ref, algorithm='loftr', output_prefix='result',
                 tile_size=800, step_size=400, search_margin=400,
                 conf_thresh=0.75, ransac_thresh=500.0, grid_size=100):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Settings: Tile={tile_size}, Step={step_size}, Margin={search_margin}, Conf={conf_thresh}")

    # --- STEP 1: COARSE MATCHING (Global) ---
    print("--- Step 1: Coarse Matching (Global) ---")
    # Для грубого поиска размер фиксируем (1200), это обычно менять не надо
    img_hex_small, size_hex_orig, size_hex_small = read_and_resize(path_hex, 1200)
    img_ref_small, size_ref_orig, size_ref_small = read_and_resize(path_ref, 1200)

    coarse_matcher = LoFTRMatcher(device)
    t_hex = to_tensor(img_hex_small, device)
    t_ref = to_tensor(img_ref_small, device)

    # Грубый поиск всегда делаем с дефолтным порогом (например 0.7), чтобы точно найти зацепку
    mkpts0, mkpts1, _ = coarse_matcher.match(t_hex, t_ref, conf_thr=0.7)
    
    if len(mkpts0) < 10:
        raise ValueError("Not enough coarse matches found! Try rotating the image.")

    H, s_hex, s_ref = compute_coarse_homography(mkpts0, mkpts1, size_hex_orig, size_hex_small, size_ref_orig, size_ref_small)
    print("Coarse Homography calculated.")

    # --- STEP 2: FINE MATCHING (Sliding Window) ---
    print(f"--- Step 2: Fine Matching using {algorithm.upper()} ---")
    
    if algorithm == 'superglue':
        matcher = SuperGlueWrapper(device)
    else:
        matcher = coarse_matcher # Reuse LoFTR instance

    final_gcp = []
    
    with rasterio.open(path_hex) as src:
        h_full, w_full = src.height, src.width
    with rasterio.open(path_ref) as src:
        ref_w, ref_h = src.width, src.height
        ref_transform = src.transform

    # Используем переданные параметры (step_size, tile_size)
    for x_hex, y_hex, w, h in generate_tiles(w_full, h_full, tile_size, step_size):
        
        center_x, center_y = x_hex + w/2, y_hex + h/2
        pred_x, pred_y = project_center(center_x, center_y, H, s_hex, s_ref)
        
        if not (0 <= pred_x < ref_w and 0 <= pred_y < ref_h):
            continue

        # Читаем тайл Hexagon
        img_hex_crop = read_tile(path_hex, x_hex, y_hex, w, h)
        if img_hex_crop is None: continue

        # Вычисляем окно на референсе с учетом margin
        ref_size = tile_size + search_margin
        ref_x_start = int(pred_x - ref_size/2)
        ref_y_start = int(pred_y - ref_size/2)
        
        ref_x_start = max(0, min(ref_x_start, ref_w - ref_size))
        ref_y_start = max(0, min(ref_y_start, ref_h - ref_size))

        img_ref_crop = read_tile(path_ref, ref_x_start, ref_y_start, ref_size, ref_size)
        if img_ref_crop is None: continue

        try:
            t_hex_c = to_tensor(img_hex_crop, device)
            t_ref_c = to_tensor(img_ref_crop, device)
            
            # Передаем conf_thresh, заданный пользователем
            kpts0, kpts1, conf = matcher.match(t_hex_c, t_ref_c, conf_thr=conf_thresh)
        except RuntimeError:
            torch.cuda.empty_cache()
            continue

        if len(kpts0) < 5: continue

        for i in range(len(kpts0)):
            gx_hex = x_hex + kpts0[i][0]
            gy_hex = y_hex + kpts0[i][1]
            gx_ref = ref_x_start + kpts1[i][0]
            gy_ref = ref_y_start + kpts1[i][1]
            
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
        print("No points found. Try increasing --margin or lowering --conf.")
        return

    df = pd.DataFrame(final_gcp)
    df.to_csv(f'{output_prefix}_raw.csv', index=False)

    # --- STEP 3: FILTERING ---
    print(f"--- Step 3: Filtering (Grid={grid_size}, RANSAC={ransac_thresh}) ---")
    
    # Передаем параметры фильтрации
    df_grid = filter_points_grid(df, grid_size=grid_size)
    df_clean = filter_points_ransac(df_grid, threshold=ransac_thresh)
    
    print(f"Clean points remaining: {len(df_clean)}")
    df_clean.to_csv(f'{output_prefix}_clean.csv', index=False)

    # --- STEP 4: WARPING ---
    print("--- Step 4: Warping ---")
    gcps = df_clean.to_dict('records')
    
    if len(gcps) < 10:
        print("Not enough points for warping (>10 required).")
        return

    warp_image(path_hex, f'{output_prefix}_poly.tif', path_ref, gcps, method='poly')
    warp_image(path_hex, f'{output_prefix}_tps.tif', path_ref, gcps, method='tps')
    
    print("Done.")

def main_cli():
    parser = argparse.ArgumentParser(description="Geobind: AI Georeferencing Tool")
    
    # Обязательные пути
    parser.add_argument('--input', required=True, help='Path to unreferenced image (Hexagon)')
    parser.add_argument('--ref', required=True, help='Path to georeferenced image')
    
    # Основные настройки
    parser.add_argument('--algo', default='loftr', choices=['loftr', 'superglue'], help='Matching algorithm')
    parser.add_argument('--output', default='georef', help='Output filename prefix')
    
    # --- НОВЫЕ ПАРАМЕТРЫ ---
    parser.add_argument('--tile-size', type=int, default=800, help='Tile size for processing (default: 800)')
    parser.add_argument('--step-size', type=int, default=400, help='Stride/Overlap step (default: 400)')
    parser.add_argument('--margin', type=int, default=400, help='Search margin on reference image (default: 400)')
    parser.add_argument('--conf', type=float, default=0.75, help='Confidence threshold 0.0-1.0 (default: 0.75)')
    
    parser.add_argument('--grid', type=int, default=100, help='Grid size for spatial binning (default: 100)')
    parser.add_argument('--ransac', type=float, default=500.0, help='RANSAC threshold in meters/pixels (default: 500.0)')
    
    args = parser.parse_args()
    
    run_pipeline(
        path_hex=args.input,
        path_ref=args.ref,
        algorithm=args.algo,
        output_prefix=args.output,
        # Передача новых аргументов
        tile_size=args.tile_size,
        step_size=args.step_size,
        search_margin=args.margin,
        conf_thresh=args.conf,
        ransac_thresh=args.ransac,
        grid_size=args.grid
    )
