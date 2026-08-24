"""
Quantum convolution depth ablation script (L = 1, 2, 3, 4, 5).
Compares empirical performance and parameter scaling against paper reference benchmarks.
"""
import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from queennet.seed import set_seed
from queennet.config import load_config
from queennet.utils import ensure_dir, setup_logger
from queennet.data.loaders import generate_synthetic_hsi
from queennet.data.preprocessing import normalize_spectral
from queennet.data.splits import split_hsi_samples
from queennet.data.patches import HSIPatchDataset
from queennet.models.queennet import QueenNet
from queennet.training.trainer import Trainer
from queennet.evaluation.metrics import calculate_metrics

PAPER_ABLATION_REF = {
    1: {"paper_oa": 93.37, "paper_aa": 91.39, "paper_kappa": 92.03, "paper_params": 1249},
    2: {"paper_oa": 96.05, "paper_aa": 95.82, "paper_kappa": 95.24, "paper_params": 1393},
    3: {"paper_oa": 95.50, "paper_aa": 94.91, "paper_kappa": 94.60, "paper_params": 1537},
    4: {"paper_oa": 94.97, "paper_aa": 94.54, "paper_kappa": 93.16, "paper_params": 1681},
    5: {"paper_oa": 93.68, "paper_aa": 93.55, "paper_kappa": 92.42, "paper_params": 1825},
}

def main():
    parser = argparse.ArgumentParser(description="Run Quantum Depth L Ablation Study")
    parser.add_argument("--config", type=str, default="configs/ablation.yaml", help="Path to config file")
    parser.add_argument("--smoke", action="store_true", help="Fast smoke mode (fewer samples & epochs)")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    seed = set_seed(cfg.get("seed", 42))
    
    out_dir = ensure_dir(Path(cfg.get("output", {}).get("dir", "outputs/ablation")))
    logger = setup_logger("ablation", out_dir / "logs" / "ablation.log")
    
    logger.info("=" * 65)
    logger.info("QUEENNET QUANTUM DEPTH (L=1..5) ABLATION STUDY")
    logger.info("=" * 65)
    
    d_cfg = cfg.get("dataset", {})
    t_cfg = cfg.get("training", {})
    
    max_epochs = 2 if args.smoke else t_cfg.get("max_epochs", 10)
    
    # 1. Dataset
    image, gt = generate_synthetic_hsi(
        height=64, width=64, bands=d_cfg.get("in_bands", 103),
        num_classes=d_cfg.get("num_classes", 4), seed=seed
    )
    norm_image = normalize_spectral(image, method="minmax")
    
    n_tr = 8 if args.smoke else d_cfg.get("train_samples_per_class", 15)
    n_va = 4 if args.smoke else d_cfg.get("val_samples_per_class", 5)
    n_te = 8 if args.smoke else d_cfg.get("test_samples_per_class", 15)
    
    split_info = split_hsi_samples(
        gt,
        train_samples_per_class=n_tr,
        val_samples_per_class=n_va,
        test_samples_per_class=n_te,
        selected_classes=d_cfg.get("selected_classes", [1, 2, 3, 4]),
        seed=seed
    )
    
    patch_size = d_cfg.get("patch_size", 32)
    train_ds = HSIPatchDataset(norm_image, split_info["train"][0], split_info["train"][1], patch_size=patch_size)
    val_ds = HSIPatchDataset(norm_image, split_info["val"][0], split_info["val"][1], patch_size=patch_size)
    test_ds = HSIPatchDataset(norm_image, split_info["test"][0], split_info["test"][1], patch_size=patch_size)
    
    batch_size = t_cfg.get("batch_size", 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    results = []
    depths = [1, 2, 3, 4, 5]
    
    for L in depths:
        logger.info(f"\n>>> Running Quantum Depth L = {L} ...")
        t0 = time.time()
        
        # Fresh model with fixed seed
        set_seed(seed)
        model = QueenNet(
            in_bands=image.shape[-1],
            num_classes=split_info["num_classes"],
            num_qubits=12,
            depth_L=L,
            patch_size=patch_size
        )
        
        counts = model.count_parameters()
        our_params = counts["trainable_parameters"]
        
        # Sub-trainer
        sub_cfg = cfg.copy()
        sub_cfg["training"]["max_epochs"] = max_epochs
        sub_out = out_dir / f"depth_L{L}"
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=sub_cfg,
            output_dir=sub_out
        )
        
        trainer.fit()
        
        # Test evaluation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for d, t in test_loader:
                logits = model(d)
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.tolist())
                all_targets.extend(t.tolist())
                
        metrics = calculate_metrics(all_targets, all_preds)
        train_time = time.time() - t0
        
        ref = PAPER_ABLATION_REF[L]
        record = {
            "L": L,
            "Our_Trainable_Params": our_params,
            "Paper_Params": ref["paper_params"],
            "Our_OA": metrics["oa"],
            "Paper_OA": ref["paper_oa"],
            "OA_Diff": metrics["oa"] - ref["paper_oa"],
            "Our_AA": metrics["aa"],
            "Paper_AA": ref["paper_aa"],
            "Our_Kappa": metrics["kappa"],
            "Paper_Kappa": ref["paper_kappa"],
            "Training_Time_Sec": train_time
        }
        results.append(record)
        
        logger.info(
            f"L={L} Results: Our OA={metrics['oa']:.2f}% (Paper={ref['paper_oa']}%) | "
            f"Our Params={our_params} (Paper={ref['paper_params']}) | "
            f"Time={train_time:.1f}s"
        )
        
    df = pd.DataFrame(results)
    metrics_dir = ensure_dir(out_dir / "metrics")
    csv_path = metrics_dir / "depth_ablation.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved ablation results table to: {csv_path}")
    
    # Generate Multi-Plot Curves
    fig_dir = ensure_dir(out_dir / "figures")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. OA vs L
    axes[0, 0].plot(df["L"], df["Paper_OA"], "r--o", label="Paper Reported OA")
    axes[0, 0].plot(df["L"], df["Our_OA"], "b-s", label="Our Reproduction OA")
    axes[0, 0].set_title("Overall Accuracy (OA) vs Depth L", fontweight="bold")
    axes[0, 0].set_xlabel("Quantum Convolution Depth (L)")
    axes[0, 0].set_ylabel("OA (%)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. AA vs L
    axes[0, 1].plot(df["L"], df["Paper_AA"], "r--o", label="Paper Reported AA")
    axes[0, 1].plot(df["L"], df["Our_AA"], "g-s", label="Our Reproduction AA")
    axes[0, 1].set_title("Average Accuracy (AA) vs Depth L", fontweight="bold")
    axes[0, 1].set_xlabel("Quantum Convolution Depth (L)")
    axes[0, 1].set_ylabel("AA (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. Kappa vs L
    axes[1, 0].plot(df["L"], df["Paper_Kappa"], "r--o", label="Paper Reported Kappa")
    axes[1, 0].plot(df["L"], df["Our_Kappa"], "m-s", label="Our Reproduction Kappa")
    axes[1, 0].set_title("Cohen's Kappa vs Depth L", fontweight="bold")
    axes[1, 0].set_xlabel("Quantum Convolution Depth (L)")
    axes[1, 0].set_ylabel("Kappa (%)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. Parameters vs L
    axes[1, 1].plot(df["L"], df["Paper_Params"], "r--o", label="Paper Reference Params")
    axes[1, 1].plot(df["L"], df["Our_Trainable_Params"], "c-s", label="Our Trainable Params")
    axes[1, 1].set_title("Trainable Parameters vs Depth L", fontweight="bold")
    axes[1, 1].set_xlabel("Quantum Convolution Depth (L)")
    axes[1, 1].set_ylabel("Parameters Count")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    fig_path = fig_dir / "ablation_depth_curves.png"
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved depth ablation curves figure to: {fig_path}")
    logger.info("[SUCCESS] Full Depth Ablation Study complete!")

if __name__ == "__main__":
    main()
