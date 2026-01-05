import cv2
import numpy as np
import pandas as pd

def compute_coarse_homography(mkpts0, mkpts1, size_hex_orig, size_hex_small, size_ref_orig, size_ref_small):
    """Calculates Homography and scaling factors from coarse match."""
    H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 50.0)
    
    scale_hex = (size_hex_orig[0] / size_hex_small[0], size_hex_orig[1] / size_hex_small[1])
    scale_ref = (size_ref_orig[0] / size_ref_small[0], size_ref_orig[1] / size_ref_small[1])
    
    return H, scale_hex, scale_ref

def project_center(x, y, H, s_hex, s_ref):
    """Projects a point from Image Space -> Reference Space using Homography."""
    # Scale down to model space
    pt_small = np.array([x / s_hex[0], y / s_hex[1], 1.0])
    # Apply Homography
    res = H @ pt_small
    res /= res[2]
    # Scale up to reference space
    final_x = res[0] * s_ref[0]
    final_y = res[1] * s_ref[1]
    return final_x, final_y

def filter_points_grid(df, grid_size=50):
    """Spatially thins points to avoid clumping."""
    df['grid_x'] = (df['pixelX'] / grid_size).astype(int)
    df['grid_y'] = (df['pixelY'] / grid_size).astype(int)
    
    # Sort by confidence
    df = df.sort_values('confidence', ascending=False)
    
    # Keep one per grid cell
    df_grid = df.drop_duplicates(subset=['grid_x', 'grid_y'], keep='first')
    return df_grid

def filter_points_ransac(df, threshold=500.0):
    """Applies RANSAC to remove geometric outliers."""
    if len(df) < 4:
        return df
        
    src_pts = df[['pixelX', 'pixelY']].values
    dst_pts = df[['mapX', 'mapY']].values
    
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
    
    df['is_inlier'] = mask.ravel()
    return df[df['is_inlier'] == 1].copy()