"""
Spatial-spectral patch extraction and PyTorch Dataset module.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List

def extract_patches(
    image: np.ndarray,
    patch_size: int = 32,
    pad_mode: str = "reflect"
) -> np.ndarray:
    """
    Pad image to support patch extraction for all border pixels.
    
    Args:
        image: (H, W, B)
        patch_size: int
        pad_mode: 'reflect', 'constant', or 'symmetric'
        
    Returns:
        padded_image: (H + 2*margin, W + 2*margin, B)
    """
    margin = patch_size // 2
    pad_width = ((margin, margin), (margin, margin), (0, 0))
    if pad_mode == "reflect":
        return np.pad(image, pad_width, mode="reflect")
    return np.pad(image, pad_width, mode="constant", constant_values=0)

class HSIPatchDataset(Dataset):
    """
    PyTorch Dataset yielding (patch, label) pairs.
    Patch shape: (B, P, P) float32 tensor suitable for 2D/3D CNN and QueenNet learnable projection.
    """
    def __init__(
        self,
        image: np.ndarray,
        coords: List[Tuple[int, int]],
        labels: List[int],
        patch_size: int = 32,
        pad_mode: str = "reflect"
    ):
        self.image = image
        self.coords = coords
        self.labels = labels
        self.patch_size = patch_size
        self.margin = patch_size // 2
        self.padded_image = extract_patches(image, patch_size=patch_size, pad_mode=pad_mode)
        
    def __len__(self) -> int:
        return len(self.coords)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        r, c = self.coords[idx]
        # In padded coordinates:
        r_pad = r + self.margin
        c_pad = c + self.margin
        
        r_start = r_pad - self.margin
        r_end = r_start + self.patch_size
        c_start = c_pad - self.margin
        c_end = c_start + self.patch_size
        
        # Patch: (H_p, W_p, B) -> Transpose to (B, H_p, W_p) for PyTorch Conv / Projection
        patch = self.padded_image[r_start:r_end, c_start:c_end, :]
        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return patch_tensor, label_tensor
