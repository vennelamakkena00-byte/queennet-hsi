"""
Early stopping handler with patience and best model checkpointing.
"""
import numpy as np
import torch
from pathlib import Path
from typing import Optional

class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after given patience."""
    def __init__(self, patience: int = 20, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        
    def step(self, current_val: float) -> bool:
        """
        Check if validation metric improved.
        
        Returns:
            improved (bool): True if current_val is the best score so far.
        """
        score = -current_val if self.mode == "min" else current_val
        
        if self.best_score is None:
            self.best_score = score
            return True
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.counter = 0
            return True
