"""
Unit tests for Quantum Encoding (QE) module.
"""
import torch
import pennylane as qml
from queennet.models.quantum_encoding import QuantumEncoding

def test_qe_forward_shape_and_gradients():
    qe = QuantumEncoding(in_bands=20, num_qubits=12)
    x = torch.randn(2, 20, 16, 16, requires_grad=True)
    thetas = qe(x)
    assert thetas.shape == (2, 12)
    assert not torch.isnan(thetas).any()
    loss = thetas.sum()
    loss.backward()
    assert qe.projection.weight.grad is not None
    assert (qe.projection.weight.grad != 0).any()

def test_qe_circuit_execution():
    qe = QuantumEncoding(in_bands=10, num_qubits=4)
    dev = qml.device("default.qubit", wires=4)
    wires = list(range(4))
    
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(theta):
        qe.apply_circuit(theta, wires)
        return [qml.expval(qml.PauliZ(w)) for w in wires]
        
    theta = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
    out = circuit(theta)
    stacked = torch.stack(out)
    assert stacked.shape == (4,)
    assert not torch.isnan(stacked).any()
