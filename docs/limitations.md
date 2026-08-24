# Limitations & Research Transparency Statement

1. **Classical Simulation vs Quantum Hardware**:
   This implementation utilizes PennyLane's `default.qubit` classical state-vector simulator. Classical simulation of a 12-qubit system requires tracking $2^{12} = 4096$ complex state amplitudes. It provides exact quantum mathematical fidelity but runs on classical CPUs without physical quantum computational acceleration.

2. **Parameter Efficiency vs Computation**:
   The primary scientific benefit of QueenNet is **extreme parameter efficiency and expressive quantum Hilbert space representations**, not faster training time on classical hardware.

3. **Resource Sensitivity**:
   Simulating 12 qubits during 300 epochs of training on large datasets (e.g. WHU-Hi-HongHu) can require substantial CPU time. The fast development mode (`--fast-dev`) and smoke test (`--smoke`) allow instant verification on standard laptops and Google Colab free tier.
