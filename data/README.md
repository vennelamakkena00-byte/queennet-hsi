# Data Directory Structure

- `data/raw/`: Place original `.mat` HSI cubes and ground truth files here.
- `data/processed/`: Cache directory for preprocessed tensors and patches.
- The pipeline provides automatic synthetic generation if `.mat` files are absent.
