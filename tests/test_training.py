"""
Unit tests for Trainer lifecycle, checkpointing, and determinism.
"""
import torch
from torch.utils.data import DataLoader
from queennet.seed import set_seed
from queennet.data.loaders import generate_synthetic_hsi
from queennet.data.preprocessing import normalize_spectral
from queennet.data.splits import split_hsi_samples
from queennet.data.patches import HSIPatchDataset
from queennet.models.queennet import QueenNet
from queennet.training.trainer import Trainer

def test_trainer_lifecycle(tmp_path):
    set_seed(42)
    img, gt = generate_synthetic_hsi(height=32, width=32, bands=10, num_classes=2, seed=42)
    norm = normalize_spectral(img, method="minmax")
    split_info = split_hsi_samples(gt, train_samples_per_class=4, val_samples_per_class=2, seed=42)
    
    train_ds = HSIPatchDataset(norm, split_info["train"][0], split_info["train"][1], patch_size=16)
    val_ds = HSIPatchDataset(norm, split_info["val"][0], split_info["val"][1], patch_size=16)
    
    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=2)
    
    model = QueenNet(in_bands=10, num_classes=2, num_qubits=4, depth_L=1, patch_size=16)
    cfg = {
        "training": {"learning_rate": 0.01, "max_epochs": 2, "patience": 2, "loss": "cross_entropy"},
        "output": {"dir": str(tmp_path)}
    }
    
    trainer = Trainer(model, train_loader, val_loader, cfg, output_dir=tmp_path)
    df = trainer.fit()
    assert len(df) == 2
    assert (tmp_path / "checkpoints" / "best_model.pt").exists()
    assert (tmp_path / "metrics" / "training_history.csv").exists()
