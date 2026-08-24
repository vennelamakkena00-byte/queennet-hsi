"""
Baseline models and parameter comparison utilities.
"""
from .mlp import ClassicalMLPBaseline
from .parameter_comparison import generate_parameter_comparison_table, plot_parameter_comparison

__all__ = [
    "ClassicalMLPBaseline",
    "generate_parameter_comparison_table",
    "plot_parameter_comparison",
]
