import json
import numpy as np
from pathlib import Path
import polars as pl
import subprocess

def test_train_and_artifacts(project_root, pyexe, fast_cache, tmp_path):
    out_dir = tmp_path / "processed"
    cmd = [
        pyexe, str(project_root / "scripts" / "train_model.py"),
        "--cache-dir", str(fast_cache),
        "--out-dir", str(out_dir),
        "--model", "dtree,rf,ridge",
        "--cv", "3",
        "--test-size", "0.2",
        "--random-state", "0",
    ]
    r = subprocess.run(cmd, cwd=project_root, check=True, capture_output=True, text=True)
    # Doit imprimer "Final model" et "Metrics"
    assert "Final model:" in r.stdout
    assert "Metrics:" in r.stdout

    # Artefacts attendus
    assert (out_dir / "model.pkl").exists()
    assert (out_dir / "X_test.npy").exists()
    assert (out_dir / "Y_test.npy").exists()
    assert (out_dir / "Y_pred.npy").exists()

    # Shapes compatibles
    Yt = np.load(out_dir / "Y_test.npy")
    Yp = np.load(out_dir / "Y_pred.npy")
    assert Yt.shape == Yp.shape and Yt.ndim == 2 and Yt.shape[1] >= 2
