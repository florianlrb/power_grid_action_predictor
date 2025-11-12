import subprocess
from pathlib import Path
import numpy as np

def test_visualize_figures(project_root, pyexe, fast_cache, tmp_path):
    # On a besoin que Y_pred.npy/Y_test.npy existent -> entraîne vite fait, just in case
    out_dir = tmp_path / "processed"
    train_cmd = [
        pyexe, str(project_root / "scripts" / "train_model.py"),
        "--cache-dir", str(fast_cache),
        "--out-dir", str(out_dir),
        "--test-size", "0.2",
        "--random-state", "0",
        "--model", "dtree"
    ]
    subprocess.run(train_cmd, cwd=project_root, check=True)

    # Lance la visu
    viz_cmd = [
        pyexe, str(project_root / "scripts" / "visualize.py"),
        "--cache-dir", str(fast_cache),
        "--out-dir", str(out_dir),
    ]
    subprocess.run(viz_cmd, cwd=project_root, check=True)

    # Figures attendues
    assert (out_dir / "targets_hist.png").exists(), "targets_hist.png manquant"
    # pred_vs_true.png n'est généré que si Y_pred/Y_test existent
    assert (out_dir / "pred_vs_true.png").exists(), "pred_vs_true.png manquant"
