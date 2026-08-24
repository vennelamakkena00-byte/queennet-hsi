"""
Classification metric calculations: OA (Overall Accuracy), AA (Average Accuracy), and Cohen Kappa.
"""
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, classification_report
from typing import Dict, Any

def calculate_metrics(y_true: np.ndarray | list, y_pred: np.ndarray | list) -> Dict[str, Any]:
    """
    Calculate OA, AA, Cohen's Kappa, and Per-Class Accuracies.
    
    Args:
        y_true: Array or list of true labels.
        y_pred: Array or list of predicted labels.
        
    Returns:
        dict containing OA (%), AA (%), Kappa (%), per_class_acc (%), and confusion_matrix.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    
    # 1. Overall Accuracy (OA)
    oa = accuracy_score(y_t, y_p) * 100.0
    
    # 2. Confusion Matrix & Per-Class Accuracy
    cm = confusion_matrix(y_t, y_p)
    class_totals = cm.sum(axis=1)
    class_correct = np.diag(cm)
    
    # Avoid zero-division for empty classes
    per_class_acc = np.where(class_totals > 0, class_correct / class_totals * 100.0, 0.0)
    
    # 3. Average Accuracy (AA)
    aa = float(np.mean(per_class_acc))
    
    # 4. Cohen's Kappa Coefficient
    kappa = cohen_kappa_score(y_t, y_p) * 100.0
    
    return {
        "oa": oa,
        "aa": aa,
        "kappa": kappa,
        "per_class_acc": per_class_acc.tolist(),
        "confusion_matrix": cm.tolist()
    }

def classification_report_dict(y_true: np.ndarray | list, y_pred: np.ndarray | list) -> str:
    """Return formatted sklearn classification report."""
    return classification_report(y_true, y_pred, digits=4, zero_division=0)
