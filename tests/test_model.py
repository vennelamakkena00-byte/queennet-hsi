"""
Unit tests for complete QueenNet architecture and parameter counting.
"""
import torch
from queennet.models.queennet import QueenNet

def test_queennet_end_to_end_forward():
    model = QueenNet(
        in_bands=20,
        num_classes=3,
        num_qubits=12,
        depth_L=2,
        patch_size=16
    )
    x = torch.randn(2, 20, 16, 16)
    logits = model(x)
    assert logits.shape == (2, 3)
    assert not torch.isnan(logits).any()

def test_queennet_parameter_counting():
    model = QueenNet(
        in_bands=103,
        num_classes=9,
        num_qubits=12,
        depth_L=2,
        patch_size=32
    )
    counts = model.count_parameters()
    assert counts["trainable_parameters"] > 0
    assert counts["quantum_convolution"] == 2 * 12 * 3 # 72 params
    assert counts["classical_projection"] == 103 * 12 + 12 # 1248 params
    assert counts["classical_classifier"] == 12 * 9 + 9 # 117 params
    assert counts["trainable_parameters"] == 1473
