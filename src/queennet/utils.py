"""
General helper utilities for device selection, logging, and directory management.
"""
import os
import json
import logging
from pathlib import Path
import torch

def get_device() -> torch.device:
    """Return PyTorch device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def setup_logger(name: str = "queennet", log_file: str | Path | None = None) -> logging.Logger:
    """Configure structured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if log_file:
        ensure_dir(Path(log_file).parent)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger
