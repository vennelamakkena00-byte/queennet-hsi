"""
Classical MLP Baseline for HSI Classification comparison.
"""
import torch
import torch.nn as nn

class ClassicalMLPBaseline(nn.Module):
    """
    Classical Multi-Layer Perceptron baseline.
    """
    def __init__(self, in_bands: int = 103, hidden_dim: int = 64, num_classes: int = 9):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.net = nn.Sequential(
            nn.Linear(in_bands, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x).flatten(1)
        return self.net(pooled)
