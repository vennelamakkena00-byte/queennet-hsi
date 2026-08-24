# QueenNet: Quantum-Enhanced Neural Network for Hyperspectral Image Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.45+-00ADD8.svg)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Clean, reproducible, zero-cost implementation of:
> **"QueenNet: Quantum-Enhanced Neural Network for Hyperspectral Image Classification"**  
> *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, Volume 19, 2026.  
> DOI: `10.1109/JSTARS.2026.3677227`

---

## 🌟 Highlights & Scientific Benefit
- **Zero-Cost & Free-Tier Compatible**: Runs on standard CPU and Google Colab free tier using PennyLane's `default.qubit` local quantum simulator (no quantum hardware, paid APIs, or cloud billing required).
- **Extreme Parameter Efficiency**: Implements QueenNet with ~**1.47k trainable parameters** vs. **85.17M** for Vision Transformers (ViT) and **6.40M** for SSAN, providing a **~58,000x parameter reduction**.
- **Accurate Model Reproduction**: Full Quantum Encoding (QE) with learnable projection and U3/CZ state preparation across 12 qubits, Parameterized Quantum Convolution (QC, $L=1..5$), and Pauli-X Quantum Measurement (QM) feeding a lightweight linear classifier.
- **Complete Test & Verification Suite**: Fully automated pytest suite, smoke tests, metric calculation (OA, AA, Cohen's Kappa), confusion matrix heatmaps, and spatial classification maps.

---

## 📂 Repository Structure

```text
queennet-hsi/
├── configs/                  # YAML configurations (smoke_test, indian_pines, pavia, ablation)
├── data/                     # Raw HSI storage & processed caches
├── docs/                     # In-depth architectural & reproduction documentation
├── notebooks/                # Google Colab & Jupyter interactive notebooks
├── outputs/                  # Saved checkpoints, metrics CSVs, figures & classification maps
├── scripts/                  # Command-line execution scripts
├── src/queennet/             # Core Python package (data, models, training, evaluation, baselines)
├── tests/                    # Comprehensive pytest unit test suite
├── pyproject.toml            # Project packaging & metadata
├── requirements.txt          # Python dependencies
└── README.md                 # Project guide
```

---

## 🚀 Quickstart & Installation

### 1. Create and Activate Virtual Environment

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows CMD:**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Environment & Autodiff
```bash
python scripts/verify_installation.py
```

---

## 🧪 Verification & Smoke Testing

Run the full automated pytest suite (all 16 tests):
```bash
pytest -q
```

Run the end-to-end quantum smoke test:
```bash
python scripts/smoke_test.py
```

Verify data pipeline integrity:
```bash
python scripts/verify_data.py
```

---

## 🔬 Running Experiments

### Parameter Counting & Comparison
```bash
python scripts/count_parameters.py --config configs/pavia_university.yaml
```

### Indian Pines Benchmark Training & Evaluation
```bash
# Fast mode (2 epochs)
python scripts/train.py --config configs/indian_pines.yaml --smoke

# Full training
python scripts/train.py --config configs/indian_pines.yaml

# Evaluation
python scripts/evaluate.py --config configs/indian_pines.yaml
```

### Pavia University Reproduction Experiment
```bash
python scripts/train.py --config configs/pavia_university.yaml --smoke
python scripts/evaluate.py --config configs/pavia_university.yaml
```

### Quantum Convolution Depth Ablation ($L=1..5$)
```bash
python scripts/run_ablation.py --smoke
```
*Generates comparison curves at `outputs/ablation/figures/ablation_depth_curves.png` and CSV metrics at `outputs/ablation/metrics/depth_ablation.csv`.*

---

## 📊 Parameter Comparison Summary

| Model Architecture | Trainable Parameters | Parameter Scale | Parameter Savings vs. ViT |
|:---|:---:|:---:|:---:|
| **Vision Transformer (ViT)** | 85,170,000 | 85.17M | Reference Baseline |
| **SSAN (Spectral-Spatial Attention)** | 6,400,000 | 6.40M | 13.3x fewer |
| **QueenNet Paper (L=2)** | 1,393 | 1.39k | **61,141x fewer** |
| **QueenNet (Our Reproduction, L=2)** | **1,473** | **1.47k** | **57,820x fewer** |

---

## ⚖️ Research Honesty & Limitations
1. **Classical Simulator**: Uses PennyLane's `default.qubit` state-vector simulator on classical CPUs/GPUs.
2. **Computational Benefit**: The demonstrated scientific benefit is **extreme parameter efficiency and expressive quantum feature representation**, not runtime speedup on classical hardware.
3. **No Paid Services Required**: Designed entirely for zero monetary cost.

---

## 📜 Citation
```bibtex
@article{queennet2026,
  title={QueenNet: Quantum-Enhanced Neural Network for Hyperspectral Image Classification},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={19},
  year={2026},
  doi={10.1109/JSTARS.2026.3677227}
}
```