"""
MATLAB .mat and synthetic HSI data loading utilities.
"""
import os
import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

DATASET_KEY_MAPPINGS = {
    "indian_pines": {
        "image_keys": ["indian_pines", "Indian_pines", "indian_pines_corrected", "data"],
        "gt_keys": ["indian_pines_gt", "Indian_pines_gt", "gt"]
    },
    "pavia_university": {
        "image_keys": ["paviaU", "PaviaU", "pavia", "data"],
        "gt_keys": ["paviaU_gt", "PaviaU_gt", "pavia_gt", "gt"]
    },
    "whu_hi_honghu": {
        "image_keys": ["WHU_Hi_HongHu", "whu_hi_honghu", "honghu", "data"],
        "gt_keys": ["WHU_Hi_HongHu_gt", "whu_hi_honghu_gt", "honghu_gt", "gt"]
    }
}

def detect_mat_key(mat_dict: Dict[str, Any], candidates: list[str]) -> str:
    """Find the first matching key from candidate list or available non-private keys."""
    for key in candidates:
        if key in mat_dict:
            return key
    # Fallback to non-dunder keys
    data_keys = [k for k in mat_dict.keys() if not k.startswith("__")]
    if len(data_keys) == 1:
        return data_keys[0]
    raise KeyError(f"None of candidates {candidates} found in keys: {list(mat_dict.keys())}")

def load_hsi_data(
    image_path: str | Path,
    gt_path: str | Path,
    dataset_name: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load HSI image array and ground-truth label map from .mat files.
    
    Returns:
        image: np.ndarray of shape (H, W, B), float32
        gt: np.ndarray of shape (H, W), int64
    """
    img_p = Path(image_path)
    gt_p = Path(gt_path)
    
    if not img_p.exists() or not gt_p.exists():
        raise FileNotFoundError(f"Missing data files: {img_p} or {gt_p}")
        
    img_mat = sio.loadmat(str(img_p))
    gt_mat = sio.loadmat(str(gt_p))
    
    cand_img = DATASET_KEY_MAPPINGS.get(dataset_name, {}).get("image_keys", ["data"]) if dataset_name else ["data"]
    cand_gt = DATASET_KEY_MAPPINGS.get(dataset_name, {}).get("gt_keys", ["gt"]) if dataset_name else ["gt"]
    
    img_key = detect_mat_key(img_mat, cand_img)
    gt_key = detect_mat_key(gt_mat, cand_gt)
    
    img = np.asarray(img_mat[img_key], dtype=np.float32)
    gt = np.asarray(gt_mat[gt_key], dtype=np.int64)
    
    if np.isnan(img).any() or np.isinf(img).any():
        raise ValueError(f"NaN or Inf values detected in HSI image: {img_p}")
        
    return img, gt

def generate_synthetic_hsi(
    height: int = 64,
    width: int = 64,
    bands: int = 103,
    num_classes: int = 9,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic HSI cube and ground-truth map for testing and verification.
    
    Returns:
        image: (height, width, bands) float32
        gt: (height, width) int64 with values in [0, num_classes] (0 is background)
    """
    rng = np.random.default_rng(seed)
    
    # Generate spatial ground truth regions
    gt = np.zeros((height, width), dtype=np.int64)
    grid_h = height // 3
    grid_w = width // 3
    
    cls_idx = 1
    for i in range(3):
        for j in range(3):
            if cls_idx <= num_classes:
                gt[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w] = cls_idx
                cls_idx += 1
                
    # Generate spectral signatures with distinct means per class
    spectral_bases = rng.uniform(0.1, 1.0, size=(num_classes + 1, bands)).astype(np.float32)
    
    image = np.zeros((height, width, bands), dtype=np.float32)
    for c in range(num_classes + 1):
        mask = (gt == c)
        if mask.any():
            noise = rng.normal(0.0, 0.05, size=(mask.sum(), bands)).astype(np.float32)
            image[mask] = np.clip(spectral_bases[c] + noise, 0.0, 1.5)
            
    return image, gt
