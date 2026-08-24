# Reproduction Notes and Paper Comparisons

## 1. Paper Overview
- **Title**: QueenNet: Quantum-Enhanced Neural Network for Hyperspectral Image Classification
- **Journal**: IEEE JSTARS, Volume 19, 2026
- **DOI**: `10.1109/JSTARS.2026.3677227`

## 2. Experimental Setup & Discrepancy Mitigation
1. **Benchmark Datasets**: The paper reports results on WHU-Hi-HongHu and Pavia University. Indian Pines is included in this repository as an educational development benchmark.
2. **Deterministic Splitting**: Split ratios follow standard HSI evaluation protocols (20 train samples per class, 10 val samples per class, remainder test samples) using fixed seeds.
3. **Classical Simulation Reality**: Execution runs via PennyLane's `default.qubit` state-vector simulator on standard CPU/GPU without quantum hardware. While quantum speedup cannot be demonstrated on classical hardware, the parameter count reduction (~1.4k params vs 85M for ViT) is fully replicated.

## 3. Reference Paper Results vs. Ablation
| Quantum Depth (L) | Paper Parameters | Paper OA (%) | Paper AA (%) | Paper Kappa (%) |
|:---:|:---:|:---:|:---:|:---:|
| L=1 | 1,249 | 93.37 | 91.39 | 92.03 |
| L=2 | 1,393 | 96.05 | 95.82 | 95.24 |
| L=3 | 1,537 | 95.50 | 94.91 | 94.60 |
| L=4 | 1,681 | 94.97 | 94.54 | 93.16 |
| L=5 | 1,825 | 93.68 | 93.55 | 92.42 |
