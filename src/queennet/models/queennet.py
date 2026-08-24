"""
Full QueenNet model assembling QE -> QC -> QM -> Classifier.
"""
import torch
import torch.nn as nn
import pennylane as qml
from typing import Dict, Any

from .quantum_encoding import QuantumEncoding
from .quantum_convolution import QuantumConvolution
from .quantum_measurement import QuantumMeasurement

class QueenNet(nn.Module):
    """
    QueenNet: Quantum-Enhanced Neural Network for Hyperspectral Image Classification.
    
    Paper: IEEE JSTARS 2026 (DOI: 10.1109/JSTARS.2026.3677227).
    """
    def __init__(
        self,
        in_bands: int = 103,
        num_classes: int = 9,
        num_qubits: int = 12,
        depth_L: int = 2,
        patch_size: int = 32,
        diff_method: str = "backprop",
        device_name: str = "default.qubit"
    ):
        super().__init__()
        self.in_bands = in_bands
        self.num_classes = num_classes
        self.num_qubits = num_qubits
        self.depth_L = depth_L
        self.patch_size = patch_size
        self.wires = list(range(num_qubits))
        
        # 1. Quantum Encoding Module
        self.qe = QuantumEncoding(in_bands=in_bands, num_qubits=num_qubits)
        
        # 2. Quantum Convolution Module
        self.qc = QuantumConvolution(num_qubits=num_qubits, depth_L=depth_L)
        
        # 3. Quantum Measurement Module
        self.qm = QuantumMeasurement(num_qubits=num_qubits, num_classes=num_classes)
        
        # 4. Initialize PennyLane quantum device and QNode
        self.dev = qml.device(device_name, wires=self.num_qubits)
        
        # Define QNode with PyTorch interface and backpropagation differentiation
        @qml.qnode(self.dev, interface="torch", diff_method=diff_method)
        def _circuit(theta_sample):
            self.qe.apply_circuit(theta_sample, self.wires)
            self.qc.apply_circuit(self.wires)
            return self.qm.get_measurement_ops(self.wires)
            
        self._circuit = _circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of HSI patches.
        
        Args:
            x: (B, C, H, W) HSI patch tensor.
            
        Returns:
            logits: (B, num_classes) classification logits.
        """
        # Step 1: Classical projection to rotation angles (B, 12)
        thetas = self.qe(x)
        
        # Step 2: Execute quantum circuit for each sample in the batch
        q_features = []
        for i in range(x.shape[0]):
            exp_vals = self._circuit(thetas[i])
            if isinstance(exp_vals, (list, tuple)):
                sample_features = torch.stack(exp_vals)
            else:
                sample_features = exp_vals
            q_features.append(sample_features)
            
        # Ensure float32 dtype matches PyTorch linear classifier
        q_features = torch.stack(q_features).to(dtype=thetas.dtype) # (B, num_qubits)
        
        # Step 3: Classification head to produce logits
        logits = self.qm(q_features) # (B, num_classes)
        return logits

    def count_parameters(self) -> Dict[str, int]:
        """
        Count and categorize trainable parameters into Classical Projection,
        Quantum Circuit (QE + QC), and Classical Classifier.
        """
        proj_params = sum(p.numel() for p in self.qe.projection.parameters() if p.requires_grad)
        qe_quantum_params = (
            self.qe.u3_phi.numel() + self.qe.u3_lam.numel() + self.qe.scale.numel()
        )
        qc_params = sum(p.numel() for p in self.qc.parameters() if p.requires_grad)
        classifier_params = sum(p.numel() for p in self.qm.parameters() if p.requires_grad)
        
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            "classical_projection": proj_params,
            "quantum_encoding": qe_quantum_params,
            "quantum_convolution": qc_params,
            "quantum_total": qe_quantum_params + qc_params,
            "classical_classifier": classifier_params,
            "classical_total": proj_params + classifier_params,
            "trainable_parameters": total_trainable,
            "total_parameters": total_params,
        }
