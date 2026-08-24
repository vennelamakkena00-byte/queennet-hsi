"""
Unit tests for evaluation metrics: OA, AA, and Cohen Kappa.
"""
import numpy as np
from queennet.evaluation.metrics import calculate_metrics

def test_metrics_calculation():
    # 10 samples, 2 classes
    # True: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    # Pred: [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
    # 8 correct out of 10 -> OA = 80%
    # Class 0: 4/5 = 80%, Class 1: 4/5 = 80% -> AA = 80%
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
    
    metrics = calculate_metrics(y_true, y_pred)
    assert abs(metrics["oa"] - 80.0) < 1e-4
    assert abs(metrics["aa"] - 80.0) < 1e-4
    assert abs(metrics["kappa"] - 60.0) < 1e-4
    assert len(metrics["per_class_acc"]) == 2
