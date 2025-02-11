import numpy as np
import json
from pathlib import Path
import os
from datetime import datetime

def flatten_state(*arrays):
    """
    Flatten multiple arrays into one.
    """
    return np.concatenate([arr.flatten() for arr in arrays if arr is not None])

def load_scenarios(json_path: Path) -> dict:
    """
    Load the scenarios JSON file from the given path.
    """
    try:
        with json_path.open("r") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading {json_path}: {e}")
    
def adjust_save_path(path: str) -> str:
    """
    Ensure the save path exists, and add default file name if directory is provided.
    """
    if os.path.isdir(path):
        filename = "model_" + datetime.now().strftime("%y%m%d") + ".zip"
        path = os.path.join(path, filename)
    if not path.lower().endswith(".zip"):
        path += ".zip"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def adjust_load_path(path: str) -> str:
    """
    Ensure the load path exists.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return path