"""
Evaluation CLI script for QueenNet HSI classification models.
"""
import argparse
import sys
import json
from pathlib import Path
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
from queennet.evaluation.metrics import calculate_metrics, classification_report_dict
from queennet.evaluation.confusion import plot_confusion_matrix
from queennet.evaluation.classification_map import generate_classification_map

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained QueenNet model on test set")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    seed = set_seed(cfg.get("seed", 42))
    
    out_dir = Path(cfg.get("output", {}).get("dir", "outputs/run"))
    ensure_dir(out_dir)
    logger = setup_logger("queennet_eval", out_dir / "logs" / "evaluation.log")
    
    logger.info("=" * 60)
    logger.info("STARTING QUEENNET EVALUATION")
    logger.info("=" * 60)
    
    d_cfg = cfg.get("dataset", {})
    m_cfg = cfg.get("model", {})
    
    # 1. Load Dataset
    d_name = d_cfg.get("name", "synthetic")
    img_path = d_cfg.get("image_path")
    gt_path = d_cfg.get("gt_path")
    
    if img_path and gt_path and Path(img_path).exists() and Path(gt_path).exists():
        image, gt = load_hsi_data(img_path, gt_path, dataset_name=d_name)
    else:
        image, gt = generate_synthetic_hsi(
            height=d_cfg.get("height", 64),
            width=d_cfg.get("width", 64),
            bands=d_cfg.get("in_bands", 103),
            num_classes=d_cfg.get("num_classes", 9),
            seed=seed
        )
        
    norm_image = normalize_spectral(image, method="minmax")
    
    # 2. Split Dataset to extract Test partition
    split_info = split_hsi_samples(
        gt,
        train_samples_per_class=d_cfg.get("train_samples_per_class", 20),
        val_samples_per_class=d_cfg.get("val_samples_per_class", 10),
        test_samples_per_class=d_cfg.get("test_samples_per_class", 20),
        selected_classes=d_cfg.get("selected_classes"),
        seed=seed
    )
    
    te_coords, te_lbls = split_info["test"]
    num_classes = split_info["num_classes"]
    
    patch_size = d_cfg.get("patch_size", 32)
    test_ds = HSIPatchDataset(norm_image, te_coords, te_lbls, patch_size=patch_size)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    
    # 3. Load Model & Weights
    model = QueenNet(
        in_bands=image.shape[-1],
        num_classes=num_classes,
        num_qubits=m_cfg.get("num_qubits", 12),
        depth_L=m_cfg.get("depth_L", 2),
        patch_size=patch_size,
        device_name=m_cfg.get("device_name", "default.qubit"),
        diff_method=m_cfg.get("diff_method", "backprop")
    )
    
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (out_dir / "checkpoints" / "best_model.pt")
    if ckpt_path.exists():
        logger.info(f"Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        logger.warning(f"Checkpoint not found at {ckpt_path}. Evaluating using initialized model.")
        
    model.eval()
    
    # 4. Inference on Test Set
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            logits = model(data)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            
    # 5. Compute Metrics
    metrics = calculate_metrics(all_targets, all_preds)
    report_str = classification_report_dict(all_targets, all_preds)
    
    logger.info("=" * 60)
    logger.info(f"Overall Accuracy (OA): {metrics['oa']:.2f}%")
    logger.info(f"Average Accuracy (AA): {metrics['aa']:.2f}%")
    logger.info(f"Cohen's Kappa:         {metrics['kappa']:.2f}%")
    logger.info("=" * 60)
    logger.info(f"\nClassification Report:\n{report_str}")
    
    # Save Metrics CSV and JSON
    metrics_dir = ensure_dir(out_dir / "metrics")
    results_df = pd.DataFrame([{
        "dataset": d_name,
        "OA": metrics["oa"],
        "AA": metrics["aa"],
        "Kappa": metrics["kappa"],
        "num_test_samples": len(all_targets),
        "parameters": model.count_parameters()["trainable_parameters"]
    }])
    results_df.to_csv(metrics_dir / "evaluation_results.csv", index=False)
    
    # 6. Save Confusion Matrix Figure
    fig_dir = ensure_dir(out_dir / "figures")
    cm_fig_path = fig_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        save_path=cm_fig_path,
        title=f"QueenNet Confusion Matrix ({d_name})"
    )
    logger.info(f"Saved confusion matrix figure to: {cm_fig_path}")
    
    # 7. Generate Spatial Classification Map
    cmap_dir = ensure_dir(out_dir / "classification_maps")
    cmap_path = cmap_dir / "classification_map.png"
    generate_classification_map(
        spatial_shape=(image.shape[0], image.shape[1]),
        coords=te_coords,
        predictions=all_preds,
        save_path=cmap_path,
        title=f"QueenNet Test Spatial Map ({d_name})"
    )
    logger.info(f"Saved classification map to: {cmap_path}")
    logger.info("[SUCCESS] Evaluation completed successfully!")

if __name__ == "__main__":
    main()
