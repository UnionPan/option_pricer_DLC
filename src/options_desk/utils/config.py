"""
Configuration loading utilities.
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def get_project_root() -> Path:
    """Get the project root directory."""
    # Assuming this file is in src/options_desk/utils/
    return Path(__file__).parent.parent.parent.parent


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load a configuration file from the config directory.

    Args:
        config_name: Name of the config file (without .yaml extension)

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_path = get_project_root() / "config" / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_data_sources_config() -> Dict[str, Any]:
    """Load data sources configuration."""
    return load_config("data_sources")


def load_hedging_config() -> Dict[str, Any]:
    """Load hedging configuration."""
    return load_config("hedging")
