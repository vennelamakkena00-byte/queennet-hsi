"""
Evaluation metrics, confusion matrices, and spatial classification maps.
"""
from .metrics import calculate_metrics, classification_report_dict
from .confusion import plot_confusion_matrix
from .classification_map import generate_classification_map

__all__ = [
    "calculate_metrics",
    "classification_report_dict",
    "plot_confusion_matrix",
    "generate_classification_map",
]
