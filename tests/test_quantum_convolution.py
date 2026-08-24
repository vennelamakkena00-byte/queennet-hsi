"""
Unit tests for Quantum Convolution (QC) parameterized circuit.
"""
import torch
import pennylane as qml
import pytest
from queennet.models.quantum_convolution import QuantumConvolution

@pytest.mark.parametrize("depth_L", [1, 2, 3, 4, 5])
def test_qc_depths_and_gradients(depth_L):
    num_qubits = 4
    wires = list(range(num_qubits))
    qc = QuantumConvolution(num_qubits=num_qubits, depth_L=depth_L)
    dev = qml.device("default.qubit", wires=num_qubits)
    
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit():
        qc.apply_circuit(wires)
        return [qml.expval(qml.PauliZ(w)) for w in wires]
        
    out = torch.stack(circuit())
    assert out.shape == (num_qubits,)
    loss = out.sum()
    loss.backward()
    assert qc.weights.grad is not None
    assert (qc.weights.grad != 0).any()
