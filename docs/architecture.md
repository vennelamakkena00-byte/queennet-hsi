# QueenNet Architecture Documentation

## 1. Overview
**QueenNet** is a Quantum-Enhanced Neural Network specifically designed for Hyperspectral Image (HSI) classification, published in *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing* (Volume 19, 2026, DOI: `10.1109/JSTARS.2026.3677227`).

The core innovation of QueenNet is its hybrid quantum-classical pipeline that maps high-dimensional spatial-spectral HSI patches into an entangled 12-qubit Hilbert space, capturing complex spectral interactions with unprecedented parameter efficiency (~1.3k to 1.8k parameters vs. millions in classical transformers).

```text
HSI Input Patch (B x C x P x P)
             |
             v
   [Quantum Encoding (QE)]
   - Spatial Pooling (P x P -> 1 x 1)
   - Parameter-Efficient Linear Projection (C -> 12)
   - U3 Single-Qubit Rotations (12 qubits)
   - Ring CZ Entanglement Gates
             |
             v
  [Quantum Convolution (QC)]
   - Multi-Layer Parameterized Circuit (L=1..5, default L=2)
   - Hadamard (H) Layer (Superposition)
   - Parameterized U3 Rotations (Trainable Angles)
   - Circular CZ Entanglement
             |
             v
 [Quantum Measurement (QM)]
   - Pauli-X Expectation Values: <X_i> on each wire
   - 12-Dimensional Measured Quantum Feature Vector
             |
             v
    [Linear Classifier]
   - Linear(12, num_classes)
   - Cross-Entropy Loss / Class Logits
```

## 2. Mathematical Formalism

### 2.1 Quantum Encoding (QE)
Let $X \in \mathbb{R}^{B \times P \times P}$ denote a spatial-spectral patch with $B$ bands and patch size $P \times P$.
1. **Spatial Pooling & Linear Projection**:
   $$\mathbf{v} = \mathbf{W}_{proj} \text{Pool}(X) + \mathbf{b}_{proj} \in \mathbb{R}^{12}$$
2. **Rotation Angle Mapping**:
   $$\theta_i = \tanh(v_i) \cdot \pi \cdot s_i, \quad i \in \{0, \dots, 11\}$$
3. **State Preparation**:
   $$|\psi_0\rangle = \left(\prod_{i=0}^{11} \text{CZ}_{i, (i+1)\%12} \right) \left(\bigotimes_{i=0}^{11} U3(\theta_i, \phi_i, \lambda_i)\right) |0\rangle^{\otimes 12}$$

### 2.2 Quantum Convolution (QC)
The QC module applies $L$ parameterized layers:
$$U_{QC}(\boldsymbol{\alpha}, \boldsymbol{\beta}, \boldsymbol{\gamma}) = \prod_{l=1}^L \left[ \left(\prod_{i=0}^{11} \text{CZ}_{i, (i+1)\%12}\right) \left(\bigotimes_{i=0}^{11} U3(\alpha_{l,i}, \beta_{l,i}, \gamma_{l,i})\right) \left(H^{\otimes 12}\right) \right]$$

### 2.3 Quantum Measurement (QM) & Classification
Pauli-X observables are measured on all 12 wires:
$$f_i = \langle \psi_{final} | X_i | \psi_{final} \rangle \in [-1, 1], \quad i = 0 \dots 11$$
The feature vector $\mathbf{f} \in \mathbb{R}^{12}$ is projected to class logits $\mathbf{z} \in \mathbb{R}^C$:
$$\mathbf{z} = \mathbf{W}_{cls} \mathbf{f} + \mathbf{b}_{cls}$$

## 3. Parameter Breakdown (Pavia University, L=2)
- Classical Projection ($103 \times 12 + 12$): **1,248 parameters**
- Quantum Encoding ($12 \times 3$): **36 parameters**
- Quantum Convolution ($2 \times 12 \times 3$): **72 parameters**
- Classical Classifier ($12 \times 9 + 9$): **117 parameters**
- **Total Trainable Parameters**: **1,473 parameters** (~1.4k params)
