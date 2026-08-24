"""
Data verification script for QueenNet HSI pipeline.
"""
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

from queennet.seed import set_seed
from queennet.data.loaders import generate_synthetic_hsi
from queennet.data.preprocessing import normalize_spectral, remove_background
from queennet.data.splits import split_hsi_samples
from queennet.data.patches import HSIPatchDataset

def main():
    print("=" * 60)
    print("QUEENNET DATA PIPELINE VERIFICATION")
    print("=" * 60)
    set_seed(42)
    
    # 1. Generate synthetic dataset
    print("[1] Generating synthetic HSI dataset (64x64, 103 bands, 9 classes)...")
    image, gt = generate_synthetic_hsi(height=64, width=64, bands=103, num_classes=9, seed=42)
    print(f"    Raw Image Shape: {image.shape}, Dtype: {image.dtype}")
    print(f"    Raw Ground-truth Shape: {gt.shape}, Unique labels: {np.unique(gt)}")
    
    assert not np.isnan(image).any(), "Found NaNs in raw image!"
    assert not np.isinf(image).any(), "Found Infs in raw image!"
    
    # 2. Normalization
    print("[2] Normalizing spectral bands...")
    norm_image = normalize_spectral(image, method="minmax")
    assert norm_image.min() >= 0.0 and norm_image.max() <= 1.0, "Normalization range invalid!"
    print(f"    Normalized range: [{norm_image.min():.4f}, {norm_image.max():.4f}]")
    
    # 3. Deterministic 2-class split test
    print("[3] Performing deterministic 2-class split...")
    split_info = split_hsi_samples(
        gt,
        train_samples_per_class=20,
        val_samples_per_class=10,
        test_samples_per_class=20,
        selected_classes=[1, 2],
        seed=42
    )
    
    tr_coords, tr_lbls = split_info["train"]
    va_coords, va_lbls = split_info["val"]
    te_coords, te_lbls = split_info["test"]
    
    print(f"    Train samples: {len(tr_coords)} (classes: {np.bincount(tr_lbls)})")
    print(f"    Val samples:   {len(va_coords)} (classes: {np.bincount(va_lbls)})")
    print(f"    Test samples:  {len(te_coords)} (classes: {np.bincount(te_lbls)})")
    
    # Verify non-overlapping coordinates
    tr_set = set(tr_coords)
    va_set = set(va_coords)
    te_set = set(te_coords)
    
    assert len(tr_set.intersection(va_set)) == 0, "Train and Val coordinates overlap!"
    assert len(tr_set.intersection(te_set)) == 0, "Train and Test coordinates overlap!"
    assert len(va_set.intersection(te_set)) == 0, "Val and Test coordinates overlap!"
    print("    [PASS] No data leakage: Train, Val, and Test coordinate sets are strictly disjoint.")
    
    # 4. Patch dataset and DataLoader test
    print("[4] Testing patch extraction and PyTorch DataLoader...")
    patch_size = 32
    train_dataset = HSIPatchDataset(norm_image, tr_coords, tr_lbls, patch_size=patch_size)
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    batch_patches, batch_labels = next(iter(loader))
    print(f"    Batch Patches Shape: {batch_patches.shape} (Expected: [4, 103, 32, 32])")
    print(f"    Batch Labels:        {batch_labels.tolist()}")
    
    assert batch_patches.shape == (4, 103, patch_size, patch_size), f"Unexpected patch shape: {batch_patches.shape}"
    assert not torch.isnan(batch_patches).any(), "Patch tensor contains NaNs!"
    
    print("\n[SUCCESS] Data pipeline fully verified!")
    print("=" * 60)

if __name__ == "__main__":
    main()
