"""
Quantum Measurement (QM) and Classifier Module for QueenNet.
Measures Pauli-X expectation values and maps to classification logits.
"""
import torch
import torch.nn as nn
import pennylane as qml

class QuantumMeasurement(nn.Module):
    """
    Quantum Measurement and Classification Head.
    
    Reads Pauli-X expectations from 12 wires and projects to class logits.
    """
    def __init__(self, num_qubits: int = 12, num_classes: int = 9):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_classes = num_classes
        
        # Classical linear classifier mapping 12 Pauli-X expectations -> num_classes logits
        self.classifier = nn.Linear(num_qubits, num_classes, bias=True)
        
    def get_measurement_ops(self, wires: list[int]):
        """Return list of Pauli-X expectation measurements for QNode."""
        return [qml.expval(qml.PauliX(w)) for w in wires]
        
    def forward(self, quantum_features: torch.Tensor) -> torch.Tensor:
        """
        Map Pauli-X expectation measurements to class logits.
        
        Args:
            quantum_features: (B, 12) tensor of Pauli-X expectations in [-1, 1].
            
        Returns:
            logits: (B, num_classes) raw class logits.
        """
        return self.classifier(quantum_features)
