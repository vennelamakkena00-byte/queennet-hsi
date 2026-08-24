"""
Training CLI script for QueenNet HSI classification.
"""
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from queennet.seed import set_seed
from queennet.config import load_config
from queennet.utils import ensure_dir, setup_logger
from queennet.data.loaders import load_hsi_data, generate_synthetic_hsi
from queennet.data.preprocessing import normalize_spectral
from queennet.data.splits import split_hsi_samples
from queennet.data.patches import HSIPatchDataset
from queennet.models.queennet import QueenNet
from queennet.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train QueenNet on HSI datasets")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--epochs", type=int, default=None, help="Override maximum epochs")
    parser.add_argument("--fast-dev", action="store_true", help="Run fast development mode (5 epochs)")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke mode (2 epochs)")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    seed = set_seed(cfg.get("seed", 42))
    
    out_dir = Path(cfg.get("output", {}).get("dir", "outputs/run"))
    ensure_dir(out_dir)
    logger = setup_logger("queennet_train", out_dir / "logs" / "run.log")
    
    logger.info("=" * 60)
    logger.info("STARTING QUEENNET TRAINING RUN")
    logger.info("=" * 60)
    logger.info(f"Loaded config: {args.config}")
    
    d_cfg = cfg.get("dataset", {})
    m_cfg = cfg.get("model", {})
    t_cfg = cfg.get("training", {})
    
    if args.smoke:
        t_cfg["max_epochs"] = 2
    elif args.fast_dev:
        t_cfg["max_epochs"] = 5
    elif args.epochs:
        t_cfg["max_epochs"] = args.epochs
        
    cfg["training"] = t_cfg
    
    # 1. Load Dataset
    d_name = d_cfg.get("name", "synthetic")
    img_path = d_cfg.get("image_path")
    gt_path = d_cfg.get("gt_path")
    
    if img_path and gt_path and Path(img_path).exists() and Path(gt_path).exists():
        logger.info(f"Loading {d_name} dataset from {img_path} and {gt_path}...")
        image, gt = load_hsi_data(img_path, gt_path, dataset_name=d_name)
    else:
        logger.info(f"Local .mat files not found for {d_name}. Generating synthetic HSI cube for experiment pipeline...")
        image, gt = generate_synthetic_hsi(
            height=d_cfg.get("height", 64),
            width=d_cfg.get("width", 64),
            bands=d_cfg.get("in_bands", 103),
            num_classes=d_cfg.get("num_classes", 9),
            seed=seed
        )
        
    norm_image = normalize_spectral(image, method="minmax")
    
    # 2. Split Dataset
    split_info = split_hsi_samples(
        gt,
        train_samples_per_class=d_cfg.get("train_samples_per_class", 20),
        val_samples_per_class=d_cfg.get("val_samples_per_class", 10),
        test_samples_per_class=d_cfg.get("test_samples_per_class", 20),
        selected_classes=d_cfg.get("selected_classes"),
        seed=seed
    )
    
    tr_coords, tr_lbls = split_info["train"]
    va_coords, va_lbls = split_info["val"]
    num_classes = split_info["num_classes"]
    
    logger.info(f"Dataset split: Train samples={len(tr_coords)}, Val samples={len(va_coords)}, Classes={num_classes}")
    
    patch_size = d_cfg.get("patch_size", 32)
    train_ds = HSIPatchDataset(norm_image, tr_coords, tr_lbls, patch_size=patch_size)
    val_ds = HSIPatchDataset(norm_image, va_coords, va_lbls, patch_size=patch_size)
    
    batch_size = t_cfg.get("batch_size", 8)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # 3. Model
    model = QueenNet(
        in_bands=image.shape[-1],
        num_classes=num_classes,
        num_qubits=m_cfg.get("num_qubits", 12),
        depth_L=m_cfg.get("depth_L", 2),
        patch_size=patch_size,
        device_name=m_cfg.get("device_name", "default.qubit"),
        diff_method=m_cfg.get("diff_method", "backprop")
    )
    
    counts = model.count_parameters()
    logger.info(f"Model instantiated: Trainable Parameters = {counts['trainable_parameters']:,} (QC: {counts['quantum_convolution']}, QE: {counts['quantum_encoding']})")
    
    # 4. Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        output_dir=out_dir
    )
    
    history_df = trainer.fit()
    
    # 5. Plot Training Curves
    fig_dir = ensure_dir(out_dir / "figures")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", marker="o")
    ax1.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", marker="s")
    ax1.set_title("Loss vs Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history_df["epoch"], history_df["train_acc"]*100, label="Train Acc (%)", marker="o")
    ax2.plot(history_df["epoch"], history_df["val_acc"]*100, label="Val Acc (%)", marker="s")
    ax2.set_title("Accuracy vs Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    curve_path = fig_dir / "training_curves.png"
    plt.savefig(curve_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved training curves figure to: {curve_path}")
    
    logger.info("=" * 60)
    logger.info("[SUCCESS] Training finished successfully!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
