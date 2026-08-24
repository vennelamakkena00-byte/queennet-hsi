"""
Spatial 2D Classification map generation.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Tuple, List

def generate_classification_map(
    spatial_shape: Tuple[int, int],
    coords: List[Tuple[int, int]],
    predictions: List[int],
    save_path: Optional[str | Path] = None,
    title: str = "QueenNet Spatial Classification Map"
) -> np.ndarray:
    """
    Construct 2D classification map from sample coordinates and predicted labels.
    
    Args:
        spatial_shape: (Height, Width)
        coords: List of (row, col) coordinates.
        predictions: Predicted class indices (0-indexed).
        save_path: Optional path to save figure.
        
    Returns:
        cmap_array: 2D numpy array of shape (H, W).
    """
    h, w = spatial_shape
    cmap_array = np.zeros((h, w), dtype=np.int32)
    
    for (r, c), pred in zip(coords, predictions):
        cmap_array[r, c] = pred + 1 # +1 so 0 remains unclassified/background
        
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cmap_array, cmap="tab20", interpolation="nearest")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")
    plt.tight_layout()
    
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(p), dpi=300)
        plt.close(fig)
        
    return cmap_array
