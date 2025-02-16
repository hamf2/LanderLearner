import os
import tempfile
from pathlib import Path
import numpy as np
import pytest
from lander_learner.utils.helpers import flatten_state, adjust_save_path, adjust_load_path

def test_flatten_state():
    a = np.array([1, 2])
    b = np.array([[3, 4], [5, 6]])
    flattened = flatten_state(a, b)
    # Expected flatten: [1, 2, 3, 4, 5, 6]
    np.testing.assert_array_equal(flattened, np.array([1, 2, 3, 4, 5, 6]))

def test_adjust_save_path_directory(tmp_path):
    # Create a temporary directory to simulate providing a directory path.
    directory = tmp_path / "models"
    directory.mkdir()
    path = str(directory)
    save_path = adjust_save_path(path, model_type="test")
    # Check that the returned path is within the directory and ends with .zip.
    assert save_path.startswith(str(directory))
    assert save_path.endswith(".zip")

def test_adjust_load_path_file(tmp_path):
    # Create a dummy zip file in a temporary directory.
    directory = tmp_path / "models"
    directory.mkdir()
    file_path = directory / "test_210101_000000.zip"
    file_path.write_text("dummy content")
    # adjust_load_path should return the same file if given the file path.
    load_path = adjust_load_path(str(file_path), model_type="test")
    assert load_path == str(file_path)

def test_adjust_load_path_directory(tmp_path):
    # Create multiple dummy zip files and check that the latest is selected.
    directory = tmp_path / "models"
    directory.mkdir()
    file1 = directory / "test_210101_000000.zip"
    file2 = directory / "test_210101_010000.zip"
    file1.write_text("dummy content")
    file2.write_text("dummy content")
    # Ensure file1 has an earlier modification time than file2.
    os.utime(file1, (file1.stat().st_atime, file1.stat().st_mtime - 10))
    load_path = adjust_load_path(str(directory), model_type="test")
    # The latest file (file2) should be returned.
    assert load_path == str(file2)
