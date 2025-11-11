
from __future__ import annotations
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

def plot_targets_hist(Y: pl.DataFrame):
    arr = Y.to_numpy().ravel()
    arr = arr[np.isfinite(arr)]
    plt.figure(figsize=(8,4))
    plt.hist(arr, bins=40)
    plt.xlabel("rho")
    plt.ylabel("count")
    plt.title("Distribution of max rho across actions (finite only)")
    plt.tight_layout()
    return plt.gcf()

def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, n_points: int = 4000):
    y_true = y_true.ravel(); y_pred = y_pred.ravel()
    n = min(n_points, y_true.size)
    idx = np.random.default_rng(0).choice(y_true.size, size=n, replace=False)
    plt.figure(figsize=(5,5))
    plt.scatter(y_true[idx], y_pred[idx], s=5, alpha=0.4)
    plt.xlabel("true rho")
    plt.ylabel("predicted rho")
    plt.title("Predicted vs True (sampled)")
    plt.tight_layout()
    return plt.gcf()

def save_figures(out_dir: str, cache_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    import numpy as np, glob
    yp = out / "Y_pred.npy"; yt = out / "Y_test.npy"
    if yp.exists() and yt.exists():
        y_pred = np.load(yp); y_true = np.load(yt)
        fig = plot_pred_vs_true(y_true, y_pred); fig.savefig(out / "pred_vs_true.png", dpi=150)
        print("Saved", out / "pred_vs_true.png")
    tpaths = sorted(glob.glob(str(Path(cache_dir) / "targets_*.parquet")))
    if tpaths:
        Y = pl.read_parquet(tpaths[-1])
        fig = plot_targets_hist(Y); fig.savefig(out / "targets_hist.png", dpi=150)
        print("Saved", out / "targets_hist.png")
