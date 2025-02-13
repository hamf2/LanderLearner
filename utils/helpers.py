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
    
def adjust_save_path(path: str, model_type: str = "") -> str:
    """
    Ensure the save path exists, and add default file name if directory is provided.
    Default file name is of the form: "<model_type>_yymmdd_HHMMSS.zip".
    """
    if os.path.isdir(path):
        filename = f"{model_type if model_type else "model"}_{datetime.now().strftime('%y%m%d_%H%M%S')}.zip"
        path = os.path.join(path, filename)
    if not str(path).lower().endswith(".zip"):
        path = str(path) + ".zip"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def adjust_load_path(path: str, model_type: str = "") -> str:
    """
    Ensure the load path exists. If a directory is provided, find the latest zip file for model_type.
    """
    if not model_type:
        model_type = "model"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    if os.path.isdir(path):
        dir_path = Path(path)
        zip_files = list(dir_path.glob(f"{model_type}*.zip"))
        if not zip_files:
            raise FileNotFoundError(f"No {model_type} zip files found in directory: {path}")
        latest_zip = max(zip_files, key=lambda f: f.stat().st_mtime)
        path = str(latest_zip)
    elif not path.lower().endswith(".zip"):
        raise ValueError(f"Invalid model file or directory: {path}")
    return path