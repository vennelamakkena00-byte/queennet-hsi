"""
Confusion matrix visualization and saving.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List

def plot_confusion_matrix(
    cm: np.ndarray | list,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str | Path] = None,
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """
    Plot and optionally save a styled confusion matrix heatmap.
    """
    cm_arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names else True,
        yticklabels=class_names if class_names else True,
        ax=ax
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(p), dpi=300)
        
    return fig
