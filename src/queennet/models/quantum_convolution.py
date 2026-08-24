"""
Quantum Convolution (QC) Module for QueenNet.
Parameterized quantum circuit extracting joint spatial-spectral features
using Hadamard gates, parameterized U3 rotations, and CZ entanglement across L layers.
"""
import torch
import torch.nn as nn
import pennylane as qml

class QuantumConvolution(nn.Module):
    """
    Quantum Convolution circuit layer module.
    
    Supports depth L in [1, 2, 3, 4, 5], default L=2.
    """
    def __init__(self, num_qubits: int = 12, depth_L: int = 2):
        super().__init__()
        self.num_qubits = num_qubits
        self.depth_L = depth_L
        
        # Trainable parameters for L layers: (L, num_qubits, 3) for U3 (alpha, beta, gamma)
        # Initialized with small uniform random values
        self.weights = nn.Parameter(
            0.1 * torch.randn(depth_L, num_qubits, 3, dtype=torch.float32)
        )
        
    def apply_circuit(self, wires: list[int]):
        """
        Apply QC quantum operations across L layers.
        
        Args:
            wires: List of wire indices (0 to num_qubits - 1).
        """
        num_w = len(wires)
        for layer in range(self.depth_L):
            # 1. Hadamard gates on all wires to create superposition
            for w in wires:
                qml.Hadamard(wires=w)
                
            # 2. Parameterized U3 rotations
            for i, w in enumerate(wires):
                alpha = self.weights[layer, i, 0]
                beta = self.weights[layer, i, 1]
                gamma = self.weights[layer, i, 2]
                qml.U3(alpha, beta, gamma, wires=w)
                
            # 3. CZ Entanglement between adjacent qubits (joint feature mixing)
            for i in range(num_w):
                w1 = wires[i]
                w2 = wires[(i + 1) % num_w]
                qml.CZ(wires=[w1, w2])
