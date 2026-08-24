"""
Loss functions for QueenNet training.
"""
import torch
import torch.nn as nn

def get_loss_fn(name: str = "cross_entropy") -> nn.Module:
    """Return configured classification loss."""
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unknown loss function: {name}")
