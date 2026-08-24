"""
Configuration parser and validator for YAML configs.
"""
import yaml
from pathlib import Path
from typing import Any, Dict

def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to YAML config file.
        
    Returns:
        Dictionary containing configuration parameters.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Ensure default fields exist
    config.setdefault("dataset", {})
    config.setdefault("model", {})
    config.setdefault("training", {})
    config.setdefault("seed", 42)
    
    return config
