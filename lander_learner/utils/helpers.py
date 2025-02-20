import numpy as np
import json
from pathlib import Path
import os
from datetime import datetime


def flatten_state(*arrays):
    """Flattens multiple arrays into a single one.

    This function takes any number of numpy arrays as input, flattens each array (ignoring any that are None),
    and concatenates them into a single 1D numpy array.

    Args:
        *arrays: Variable length argument list of numpy arrays. Any None values are skipped.

    Returns:
        numpy.ndarray: A 1D array resulting from the concatenation of the flattened input arrays.
    """
    return np.concatenate([arr.flatten() for arr in arrays if arr is not None])


def load_scenarios(json_path: Path) -> dict:
    """Loads scenario configurations from a JSON file.

    Reads the JSON file located at the specified path and returns its contents as a dictionary.

    Args:
        json_path (Path): A Path object pointing to the JSON file containing scenario configurations.

    Returns:
        dict: A dictionary of scenario configurations.

    Raises:
        RuntimeError: If an error occurs while opening or parsing the JSON file.
    """
    try:
        with json_path.open("r") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading {json_path}: {e}")


def adjust_save_path(path: str, model_type: str = "") -> str:
    """Adjusts the given path to ensure a valid save file path for the model.

    If the provided path is a directory, a default filename is generated using the format
    "<model_type>_yymmdd_HHMMSS.zip" (or "model_yymmdd_HHMMSS.zip" if model_type is empty) and appended
    to the directory. If the path does not end with ".zip", the extension is appended.
    Additionally, the directory for the file is created if it does not exist.

    Args:
        path (str): The desired save path, which may be a directory or a full file path.
        model_type (str, optional): The model type used in the default filename. Defaults to an empty string.

    Returns:
        str: A valid file path ending with ".zip" suitable for saving the model.
    """
    if os.path.isdir(path):
        filename = f"{model_type if model_type else 'model'}_{datetime.now().strftime('%y%m%d_%H%M%S')}.zip"
        path = os.path.join(path, filename)
    if not str(path).lower().endswith(".zip"):
        path = str(path) + ".zip"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def adjust_load_path(path: str, model_type: str = "") -> str:
    """Adjusts the given path to ensure a valid model file path for loading.

    If the provided path is a directory, the function searches for the latest zip file
    whose name starts with the given model_type (or "model" if model_type is empty) and returns its path.
    If the provided path is a file, it validates that it ends with ".zip".

    Args:
        path (str): The path or directory where the model file is expected.
        model_type (str, optional): The model type prefix to search for if a directory is provided.
            Defaults to an empty string, which is interpreted as "model".

    Returns:
        str: The path to the model file (a .zip file).

    Raises:
        FileNotFoundError: If the specified path does not exist or if no matching zip files are found in a directory.
        ValueError: If the provided path is not a directory and does not end with ".zip".
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
