"""
Spectral normalization and preprocessing functions.
"""
import numpy as np

def normalize_spectral(image: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize HSI cube across spectral bands.
    
    Args:
        image: np.ndarray of shape (H, W, B)
        method: "minmax" or "standard"
        
    Returns:
        Normalized array of shape (H, W, B), float32.
    """
    img = image.astype(np.float32)
    if method == "minmax":
        min_val = img.min()
        max_val = img.max()
        if max_val - min_val > 1e-8:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = np.zeros_like(img)
    elif method == "standard":
        mean = img.mean(axis=(0, 1), keepdims=True)
        std = img.std(axis=(0, 1), keepdims=True) + 1e-8
        img = (img - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")
        
    return img

def remove_background(gt: np.ndarray, bg_class: int = 0) -> np.ndarray:
    """Return boolean mask indicating labeled (non-background) pixels."""
    return gt != bg_class
