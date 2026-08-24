"""
Verification script for QueenNet installation and quantum autodiff environment.
"""
import sys
import numpy as np
import torch
import pennylane as qml
import sklearn
import scipy
import matplotlib

from queennet.seed import set_seed

def main():
    print("=" * 60)
    print("QUEENNET ENVIRONMENT VERIFICATION")
    print("=" * 60)
    print(f"Python Version:       {sys.version.split()[0]}")
    print(f"PyTorch Version:      {torch.__version__}")
    print(f"PennyLane Version:    {qml.__version__}")
    print(f"NumPy Version:        {np.__version__}")
    print(f"Scikit-Learn Version: {sklearn.__version__}")
    print(f"SciPy Version:        {scipy.__version__}")
    print(f"Matplotlib Version:   {matplotlib.__version__}")
    
    # Test PennyLane local device
    dev = qml.device("default.qubit", wires=2)
    print(f"PennyLane Device:     {dev.name} ({dev.wires.tolist()} wires)")
    
    # Check deterministic seed setting
    set_seed(42)
    print("Deterministic Seed:   Configured (seed=42)")
    
    # 2-qubit differentiable circuit test
    print("\n--- Verifying PennyLane + PyTorch Gradient Flow ---")
    
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def test_circuit(weights):
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(1))
    
    weights = torch.tensor([0.5, 0.8], requires_grad=True, dtype=torch.float32)
    output = test_circuit(weights)
    loss = (output - 1.0) ** 2
    loss.backward()
    
    print(f"Circuit Forward Output: {output.item():.6f}")
    print(f"Loss Value:             {loss.item():.6f}")
    print(f"Gradients w.r.t weights: {weights.grad.numpy()}")
    
    assert weights.grad is not None, "Gradient was not computed!"
    assert not torch.isnan(weights.grad).any(), "Gradient contains NaNs!"
    assert (weights.grad != 0).any(), "Gradient is all zeros!"
    
    print("\n[SUCCESS] Quantum-Classical Autograd and Environment fully verified!")
    print("=" * 60)

if __name__ == "__main__":
    main()
