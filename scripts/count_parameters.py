"""
Script to count and categorize QueenNet parameters against classical baselines.
"""
import argparse
import yaml
from pathlib import Path
from queennet.config import load_config
from queennet.models.queennet import QueenNet
from queennet.baselines.parameter_comparison import (
    generate_parameter_comparison_table,
    plot_parameter_comparison
)
from queennet.utils import ensure_dir

def main():
    parser = argparse.ArgumentParser(description="Count QueenNet parameters and compare with baselines")
    parser.add_argument("--config", type=str, default="configs/pavia_university.yaml", help="Path to config file")
    parser.add_argument("--depth", type=int, default=2, help="Quantum convolution depth L")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    m_cfg = cfg.get("model", {})
    
    in_bands = m_cfg.get("in_bands", 103)
    num_classes = m_cfg.get("num_classes", 9)
    num_qubits = m_cfg.get("num_qubits", 12)
    depth_L = args.depth
    
    print("=" * 65)
    print("QUEENNET PARAMETER EFFICIENCY ANALYSIS")
    print("=" * 65)
    print(f"Configuration: Bands={in_bands}, Classes={num_classes}, Qubits={num_qubits}, Depth L={depth_L}")
    
    model = QueenNet(
        in_bands=in_bands,
        num_classes=num_classes,
        num_qubits=num_qubits,
        depth_L=depth_L
    )
    
    counts = model.count_parameters()
    
    print("\n--- Detailed Parameter Breakdown ---")
    print(f"  1. Classical Projection (Linear): {counts['classical_projection']:,}")
    print(f"  2. Quantum Encoding (QE Phases):   {counts['quantum_encoding']:,}")
    print(f"  3. Quantum Convolution (QC L={depth_L}): {counts['quantum_convolution']:,}")
    print(f"  4. Classical Classifier Head:     {counts['classical_classifier']:,}")
    print("  " + "-" * 40)
    print(f"  Total Quantum Parameters:          {counts['quantum_total']:,}")
    print(f"  Total Classical Parameters:        {counts['classical_total']:,}")
    print(f"  TOTAL TRAINABLE PARAMETERS:        {counts['trainable_parameters']:,}")
    print("=" * 65)
    
    # Baseline comparison table
    df = generate_parameter_comparison_table(our_queennet_params=counts['trainable_parameters'])
    print("\n--- Benchmark Model Comparison ---")
    print(df.to_string(index=False))
    
    # Save comparison figure
    fig_dir = ensure_dir("outputs/figures")
    fig_path = fig_dir / "parameter_comparison.png"
    plot_parameter_comparison(our_params=counts['trainable_parameters'], save_path=fig_path)
    print(f"\n[SAVED] Parameter comparison figure saved to: {fig_path}")

if __name__ == "__main__":
    main()
