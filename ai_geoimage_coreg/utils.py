import os
import torch
import numpy as np

def download_weights(url, dest_path):
    """Downloads model weights if they don't exist."""
    if not os.path.exists(dest_path):
        print(f"Downloading weights to {dest_path}...")
        torch.hub.download_url_to_file(url, dest_path)

def generate_tiles(width, height, tile_size, step_size):
    """
    Generator for sliding window coordinates.
    Yields: (x, y, tile_width, tile_height)
    """
    for y in range(0, height - tile_size + 1, step_size):
        for x in range(0, width - tile_size + 1, step_size):
            yield x, y, tile_size, tile_size

def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()