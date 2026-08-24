"""
Unit tests for HSI data loading, normalization, patch extraction, and splitting.
"""
import numpy as np
import pytest
import torch
from queennet.data.loaders import generate_synthetic_hsi
from queennet.data.preprocessing import normalize_spectral, remove_background
from queennet.data.splits import split_hsi_samples
from queennet.data.patches import extract_patches, HSIPatchDataset

def test_synthetic_data_generation():
    img, gt = generate_synthetic_hsi(height=32, width=32, bands=20, num_classes=3, seed=42)
    assert img.shape == (32, 32, 20)
    assert gt.shape == (32, 32)
    assert not np.isnan(img).any()
    assert set(np.unique(gt)).issubset({0, 1, 2, 3})

def test_spectral_normalization():
    img = np.array([[[1.0, 5.0], [2.0, 10.0]]], dtype=np.float32)
    norm = normalize_spectral(img, method="minmax")
    assert norm.min() == 0.0
    assert norm.max() == 1.0

def test_split_no_data_leakage():
    _, gt = generate_synthetic_hsi(height=32, width=32, bands=10, num_classes=2, seed=42)
    split_info = split_hsi_samples(gt, train_samples_per_class=10, val_samples_per_class=5, test_samples_per_class=10, seed=42)
    tr_set = set(split_info["train"][0])
    va_set = set(split_info["val"][0])
    te_set = set(split_info["test"][0])
    assert len(tr_set.intersection(va_set)) == 0
    assert len(tr_set.intersection(te_set)) == 0
    assert len(va_set.intersection(te_set)) == 0

def test_hsi_patch_dataset():
    img, gt = generate_synthetic_hsi(height=32, width=32, bands=15, num_classes=2, seed=42)
    split_info = split_hsi_samples(gt, train_samples_per_class=5, val_samples_per_class=2, test_samples_per_class=5, seed=42)
    coords, lbls = split_info["train"]
    dataset = HSIPatchDataset(img, coords, lbls, patch_size=16)
    assert len(dataset) == len(coords)
    patch, label = dataset[0]
    assert patch.shape == (15, 16, 16)
    assert isinstance(label.item(), int)
