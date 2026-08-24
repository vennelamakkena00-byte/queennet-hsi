"""
Quantum Encoding (QE) Module for QueenNet.
Accepts HSI patches, performs learnable projection down to 12 dimensions,
and prepares the 12-qubit quantum state using U3 rotations and CZ entanglement.
"""
import torch
import torch.nn as nn
import pennylane as qml
from typing import Tuple

class QuantumEncoding(nn.Module):
    """
    Quantum Encoding module.
    
    Transforms (B, C, H, W) HSI patch to 12-dimensional quantum features
    via spatial-spectral pooling + linear projection, followed by U3 and CZ encoding.
    """
    def __init__(self, in_bands: int = 103, num_qubits: int = 12):
        super().__init__()
        self.in_bands = in_bands
        self.num_qubits = num_qubits
        
        # Spatial pooling across patch: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Parameter-efficient learnable projection from C spectral bands to 12 quantum features
        self.projection = nn.Linear(in_bands, num_qubits, bias=True)
        
        # Learnable phase parameters for U3 gate angles (phi and lambda)
        self.u3_phi = nn.Parameter(torch.zeros(num_qubits, dtype=torch.float32))
        self.u3_lam = nn.Parameter(torch.zeros(num_qubits, dtype=torch.float32))
        self.scale = nn.Parameter(torch.ones(num_qubits, dtype=torch.float32))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 12-dimensional classical angles from HSI patch.
        
        Args:
            x: (B, C, H, W) tensor
            
        Returns:
            theta: (B, 12) tensor of rotation angles
        """
        # Spatial pooling
        pooled = self.spatial_pool(x).flatten(1) # (B, in_bands)
        # Linear projection
        proj = self.projection(pooled) # (B, num_qubits)
        # Scale and non-linear bounded phase mapping
        theta = torch.tanh(proj) * torch.pi * self.scale
        return theta

    def apply_circuit(self, theta: torch.Tensor, wires: list[int]):
        """
        Apply QE quantum operations on PennyLane wires.
        
        Args:
            theta: 1D tensor of length 12 containing input theta angles for one sample.
            wires: List of wire indices.
        """
        # 1. U3 single-qubit rotations
        for i, w in enumerate(wires):
            qml.U3(theta[i], self.u3_phi[i], self.u3_lam[i], wires=w)
            
        # 2. CZ Entanglement across neighboring qubits in a ring
        for i in range(len(wires)):
            w1 = wires[i]
            w2 = wires[(i + 1) % len(wires)]
            qml.CZ(wires=[w1, w2])
