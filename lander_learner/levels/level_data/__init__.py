"""Utility helpers for loading predefined level data.

Provides a thin wrapper around JSON loading so higher level modules can work
with dictionary payloads that describe level layouts and parameters.
"""

from pathlib import Path
from typing import Any, Dict

import json


def load_level_payload(name: str) -> Dict[str, Any]:
    """Load a JSON payload describing a predefined level.

    Args:
        name (str): Base filename (without extension) of the level preset.

    Returns:
        Dict[str, Any]: Parsed JSON data containing level configuration values.

    Raises:
        FileNotFoundError: If the preset JSON file cannot be located.
        json.JSONDecodeError: If the preset file contains invalid JSON.
    """
    root = Path(__file__).resolve().parent
    candidate = root / f"{name}.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Level preset '{name}' not found in {root}")
    with candidate.open("r", encoding="utf-8") as handle:
        return json.load(handle)
