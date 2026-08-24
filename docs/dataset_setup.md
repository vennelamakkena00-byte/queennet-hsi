# Dataset Setup & Preparation Guide

## 1. Directory Structure
Place `.mat` files in `data/raw/`:
```text
data/raw/
├── indian_pines/
│   ├── Indian_pines_corrected.mat
│   └── Indian_pines_gt.mat
├── pavia_university/
│   ├── PaviaU.mat
│   └── PaviaU_gt.mat
└── whu_hi_honghu/
    ├── WHU_Hi_HongHu.mat
    └── WHU_Hi_HongHu_gt.mat
```

## 2. Public Download Sources
- **Indian Pines & Pavia University**: Available from Grupo de Inteligencia Computacional (UPV/EHU) or standard remote sensing repositories.
- **WHU-Hi-HongHu**: Available from Wuhan University RSIDEA Group.

## 3. Automatic Synthetic Fallback
If `.mat` files are not present in `data/raw/`, the data pipeline automatically generates high-fidelity synthetic HSI cubes with matching spatial dimensions and spectral profiles for testing and development.
