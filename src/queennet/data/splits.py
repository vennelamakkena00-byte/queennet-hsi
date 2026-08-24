"""
Deterministic dataset splitting with per-class constraints and non-overlapping sets.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional

def split_hsi_samples(
    gt: np.ndarray,
    train_samples_per_class: Optional[int] = 20,
    val_samples_per_class: Optional[int] = 10,
    test_samples_per_class: Optional[int] = None,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    selected_classes: Optional[List[int]] = None,
    seed: int = 42
) -> Dict[str, Tuple[List[Tuple[int, int]], List[int]]]:
    """
    Deterministically split HSI coordinates into train, validation, and test sets.
    Labels are mapped to 0-indexed contiguous labels [0, num_classes-1].
    
    Returns:
        dict with keys "train", "val", "test", each mapping to (coords, labels_0indexed).
    """
    rng = np.random.default_rng(seed)
    
    unique_classes = np.unique(gt)
    valid_classes = [c for c in unique_classes if c != 0] # exclude background
    
    if selected_classes is not None:
        valid_classes = [c for c in valid_classes if c in selected_classes]
        
    valid_classes.sort()
    class_to_idx = {c: i for i, c in enumerate(valid_classes)}
    
    train_coords, train_labels = [], []
    val_coords, val_labels = [], []
    test_coords, test_labels = [], []
    
    for cls in valid_classes:
        rows, cols = np.where(gt == cls)
        coords = list(zip(rows, cols))
        rng.shuffle(coords)
        
        total_n = len(coords)
        if total_n == 0:
            continue
            
        if train_samples_per_class is not None:
            n_tr = min(train_samples_per_class, total_n - 2) if total_n > 2 else 1
            n_va = min(val_samples_per_class or 0, total_n - n_tr - 1) if total_n - n_tr > 1 else 0
            n_te = min(test_samples_per_class, total_n - n_tr - n_va) if test_samples_per_class else (total_n - n_tr - n_va)
        elif train_ratio is not None:
            n_tr = max(1, int(total_n * train_ratio))
            v_ratio = val_ratio if val_ratio is not None else 0.1
            n_va = max(1, int(total_n * v_ratio))
            n_te = total_n - n_tr - n_va
        else:
            n_tr = min(20, total_n // 3)
            n_va = min(10, total_n // 6)
            n_te = total_n - n_tr - n_va
            
        tr_c = coords[:n_tr]
        va_c = coords[n_tr:n_tr+n_va]
        te_c = coords[n_tr+n_va:n_tr+n_va+n_te]
        
        idx_label = class_to_idx[cls]
        
        train_coords.extend(tr_c)
        train_labels.extend([idx_label] * len(tr_c))
        
        val_coords.extend(va_c)
        val_labels.extend([idx_label] * len(va_c))
        
        test_coords.extend(te_c)
        test_labels.extend([idx_label] * len(te_c))
        
    return {
        "train": (train_coords, train_labels),
        "val": (val_coords, val_labels),
        "test": (test_coords, test_labels),
        "class_mapping": class_to_idx,
        "num_classes": len(valid_classes)
    }
