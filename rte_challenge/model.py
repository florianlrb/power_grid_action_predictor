
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
import polars as pl
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor

def _clean_inf(y: np.ndarray) -> np.ndarray:
    y = y.copy()
    finite = np.isfinite(y)
    if finite.any():
        y[~finite] = np.max(y[finite]) + 1.0
    else:
        y[:] = 1e6
    return y

def train_test_split(X: pl.DataFrame, Y: pl.DataFrame, test_size: float = 0.2, random_state: int = 0):
    rng = np.random.default_rng(random_state)
    idx = np.arange(X.height); rng.shuffle(idx)
    n_test = max(1, int(test_size * len(idx)))
    test_idx = idx[:n_test]; train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]

def fit_model(X_train: pl.DataFrame, Y_train: pl.DataFrame):
    model = MultiOutputRegressor(DecisionTreeRegressor(random_state=0))
    model.fit(X_train.to_numpy(), _clean_inf(Y_train.to_numpy()))
    return model

def predict(model, X: pl.DataFrame) -> np.ndarray:
    return model.predict(X.to_numpy())

def evaluate(model, X_test: pl.DataFrame, Y_test: pl.DataFrame) -> dict:
    y_pred = predict(model, X_test)
    y_true = _clean_inf(Y_test.to_numpy())
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    true_rank = np.argsort(y_true, axis=1)
    pred_best = np.argmin(y_pred, axis=1)
    hits = [(pred_best[i] == true_rank[i,0]) for i in range(y_pred.shape[0])]
    return {"rmse_mean": float(rmse.mean() if rmse.size else 0.0),
            "top1_acc": float(np.mean(hits) if hits else 0.0)}

def save_artifacts(out_dir: str, model, X_test: pl.DataFrame, Y_test: pl.DataFrame):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    import pickle, numpy as np
    with open(out / "model.pkl", "wb") as f: pickle.dump(model, f)
    np.save(out / "X_test.npy", X_test.to_numpy())
    np.save(out / "Y_test.npy", Y_test.to_numpy())
    np.save(out / "Y_pred.npy", predict(model, X_test))
