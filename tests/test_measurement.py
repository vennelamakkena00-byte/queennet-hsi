"""
Unit tests for Quantum Measurement (QM) and Classifier module.
"""
import torch
from queennet.models.quantum_measurement import QuantumMeasurement

def test_qm_classifier_forward_and_gradients():
    qm = QuantumMeasurement(num_qubits=12, num_classes=5)
    # Simulated Pauli-X expectations in [-1, 1]
    features = torch.empty(4, 12).uniform_(-1.0, 1.0)
    features.requires_grad = True
    logits = qm(features)
    assert logits.shape == (4, 5)
    loss = logits.sum()
    loss.backward()
    assert qm.classifier.weight.grad is not None
    assert features.grad is not None
