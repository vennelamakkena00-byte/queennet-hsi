"""
Parameter efficiency comparison module between QueenNet and classical architectures (ViT, SSAN).
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict

# Reference parameter counts reported in the paper (IEEE JSTARS 2026)
PAPER_BENCHMARK_PARAMS = {
    "Vision Transformer (ViT)": 85_170_000,
    "SSAN (Spectral-Spatial Attention)": 6_400_000,
    "QueenNet (L=1, Paper)": 1_249,
    "QueenNet (L=2, Paper)": 1_393,
    "QueenNet (L=3, Paper)": 1_537,
    "QueenNet (L=4, Paper)": 1_681,
    "QueenNet (L=5, Paper)": 1_825,
}

def generate_parameter_comparison_table(our_queennet_params: Optional[int] = None) -> pd.DataFrame:
    """
    Create structured comparison DataFrame of model parameter counts.
    """
    data = []
    for model_name, params in PAPER_BENCHMARK_PARAMS.items():
        data.append({
            "Model Architecture": model_name,
            "Type": "Classical Baseline" if "ViT" in model_name or "SSAN" in model_name else "Quantum Paper Reference",
            "Trainable Parameters": params,
            "Parameters (M/K)": f"{params/1e6:.2f}M" if params >= 1e6 else f"{params/1e3:.2f}k"
        })
        
    if our_queennet_params is not None:
        data.append({
            "Model Architecture": "QueenNet (Our Reproduction, L=2)",
            "Type": "Our Reproduction",
            "Trainable Parameters": our_queennet_params,
            "Parameters (M/K)": f"{our_queennet_params/1e3:.2f}k"
        })
        
    return pd.DataFrame(data)

def plot_parameter_comparison(
    our_params: int = 1393,
    save_path: Optional[str | Path] = None
) -> plt.Figure:
    """
    Plot logarithmic bar chart comparing parameter efficiency.
    """
    models = ["ViT (Classical)", "SSAN (Classical)", "QueenNet Paper (L=2)", "QueenNet Ours (L=2)"]
    params = [85_170_000, 6_400_000, 1393, our_params]
    colors = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, params, color=colors, edgecolor="black", width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("Trainable Parameters (Log Scale)", fontsize=12, fontweight="bold")
    ax.set_title("Model Parameter Efficiency Comparison (IEEE JSTARS 2026)", fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    for bar, val in zip(bars, params):
        label = f"{val/1e6:.2f}M" if val >= 1e6 else f"{val:,}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.3,
            label,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )
        
    # Explanatory note
    plt.figtext(
        0.5, -0.05,
        "Note: Parameter count reflects memory and representational compactness. Classical simulation runtime does not reflect physical quantum speedup.",
        ha="center", fontsize=9, style="italic"
    )
    plt.tight_layout()
    
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(p), dpi=300, bbox_inches="tight")
        
    return fig
