"""
End-to-End Quantum Smoke Test for QueenNet.
Verifies forward pass, loss calculation, backward autodiff, optimizer updates,
checkpoint saving/loading, and deterministic reproduction on a 2-class subset.
"""
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from queennet.seed import set_seed
from queennet.config import load_config
from queennet.data.loaders import generate_synthetic_hsi
from queennet.data.preprocessing import normalize_spectral
from queennet.data.splits import split_hsi_samples
from queennet.data.patches import HSIPatchDataset
from queennet.models.queennet import QueenNet
from queennet.utils import ensure_dir

def main():
    print("=" * 65)
    print("QUEENNET QUANTUM GRADIENT & END-TO-END SMOKE TEST")
    print("=" * 65)
    
    # 1. Deterministic seed
    set_seed(42)
    print("[1] Configured deterministic seed=42")
    
    # 2. Generate 2-class synthetic dataset
    print("[2] Generating 2-class synthetic HSI dataset...")
    image, gt = generate_synthetic_hsi(height=64, width=64, bands=103, num_classes=2, seed=42)
    norm_image = normalize_spectral(image, method="minmax")
    
    split_info = split_hsi_samples(
        gt,
        train_samples_per_class=10,
        val_samples_per_class=5,
        test_samples_per_class=10,
        selected_classes=[1, 2],
        seed=42
    )
    
    tr_coords, tr_lbls = split_info["train"]
    va_coords, va_lbls = split_info["val"]
    
    train_dataset = HSIPatchDataset(norm_image, tr_coords, tr_lbls, patch_size=32)
    val_dataset = HSIPatchDataset(norm_image, va_coords, va_lbls, patch_size=32)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"    Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")
    
    # 3. Instantiate QueenNet
    print("[3] Instantiating 12-qubit QueenNet (L=2)...")
    model = QueenNet(
        in_bands=103,
        num_classes=2,
        num_qubits=12,
        depth_L=2,
        patch_size=32
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 4. Multi-step optimization & gradient verification
    print("[4] Running 2 optimization steps and tracking loss & gradients...")
    losses = []
    
    for step in range(2):
        optimizer.zero_grad()
        data_batch, label_batch = next(iter(train_loader))
        
        logits = model(data_batch)
        loss = criterion(logits, label_batch)
        loss.backward()
        
        # Check gradients
        qc_grad = model.qc.weights.grad
        proj_grad = model.qe.projection.weight.grad
        clf_grad = model.qm.classifier.weight.grad
        
        assert proj_grad is not None and (proj_grad != 0).any(), "Classical projection grad is zero/None!"
        assert qc_grad is not None and (qc_grad != 0).any(), "Quantum convolution grad is zero/None!"
        assert clf_grad is not None and (clf_grad != 0).any(), "Classifier head grad is zero/None!"
        
        losses.append(loss.item())
        optimizer.step()
        
        print(f"    Step {step+1}: Loss = {loss.item():.6f} | QC Grad Norm = {qc_grad.norm().item():.6f}")
        
    assert losses[1] != losses[0], "Loss did not change between optimizer steps!"
    print("    [PASS] Gradients correctly flowed through QC, QE, and QM layers.")
    
    # 5. Checkpoint Saving & Reloading
    print("[5] Testing model checkpoint save & reload...")
    ckpt_dir = ensure_dir("outputs/checkpoints")
    ckpt_path = ckpt_dir / "smoke_test_model.pt"
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": losses[-1]
    }, str(ckpt_path))
    print(f"    Saved checkpoint to: {ckpt_path}")
    
    # Reload model
    reloaded_model = QueenNet(
        in_bands=103,
        num_classes=2,
        num_qubits=12,
        depth_L=2,
        patch_size=32
    )
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    reloaded_model.load_state_dict(ckpt["model_state_dict"])
    reloaded_model.eval()
    
    # Test identical inference reproduction
    model.eval()
    test_batch, _ = next(iter(val_loader))
    with torch.no_grad():
        orig_preds = model(test_batch)
        reloaded_preds = reloaded_model(test_batch)
        
    diff = torch.abs(orig_preds - reloaded_preds).max().item()
    print(f"    Max prediction diff between original & reloaded model: {diff:.8f}")
    assert diff < 1e-6, "Reloaded model failed to reproduce identical output!"
    print("    [PASS] Checkpoint reload verified with exact deterministic parity.")
    
    print("\n" + "=" * 65)
    print("[SUCCESS] All Phase 7 Smoke Test criteria passed cleanly!")
    print("=" * 65)

if __name__ == "__main__":
    main()
