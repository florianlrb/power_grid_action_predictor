from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import polars as pl

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold

# ---------- utils ----------

def _clean_inf(y: np.ndarray) -> np.ndarray:
    y = y.copy()
    finite = np.isfinite(y)
    if finite.any():
        y[~finite] = np.max(y[finite]) + 1.0
    else:
        y[:] = 1e6
    return y

def _rmse_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0)).mean())

def _top1_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_rank = np.argsort(y_true, axis=1)
    pred_best = np.argmin(y_pred, axis=1)
    hits = [(pred_best[i] == true_rank[i, 0]) for i in range(y_pred.shape[0])]
    return float(np.mean(hits) if hits else 0.0)

# ---------- modèles ----------

def _wrap_multioutput(base):
    # Par sécurité, on wrappe tout : ça fonctionne pour Ridge/GBR, et ça reste ok pour DT/RF.
    return MultiOutputRegressor(base)

def get_estimator(name: str, random_state: int = 0, **kw):
    name = name.lower()
    if name == "dtree":
        return _wrap_multioutput(DecisionTreeRegressor(random_state=random_state, **kw))
    if name == "rf":
        return _wrap_multioutput(RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1, **kw))
    if name == "ridge":
        return _wrap_multioutput(Ridge(**kw))
    if name == "gbr":
        return _wrap_multioutput(GradientBoostingRegressor(random_state=random_state, **kw))
    raise ValueError(f"Unknown model name: {name} (use: dtree, rf, ridge, gbr)")

# ---------- API entraînement / évaluation ----------

def train_test_split(
    X: pl.DataFrame, Y: pl.DataFrame, test_size: float = 0.2, random_state: int = 0
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(random_state)
    idx = np.arange(X.height); rng.shuffle(idx)
    n_test = max(1, int(test_size * len(idx)))
    test_idx = idx[:n_test]; train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]

def fit_model(X_train: pl.DataFrame, Y_train: pl.DataFrame, model_name: str = "dtree", random_state: int = 0, **kw):
    est = get_estimator(model_name, random_state=random_state, **kw)
    Xnp = X_train.to_numpy()
    Ynp = _clean_inf(Y_train.to_numpy())
    est.fit(Xnp, Ynp)
    return est

def predict(model, X: pl.DataFrame) -> np.ndarray:
    return model.predict(X.to_numpy())

def evaluate(model, X_test: pl.DataFrame, Y_test: pl.DataFrame) -> dict:
    y_pred = predict(model, X_test)
    y_true = _clean_inf(Y_test.to_numpy())
    return {
        "rmse_mean": _rmse_mean(y_true, y_pred),
        "top1_acc": _top1_acc(y_true, y_pred),
    }

# ---------- cross-validation ----------

def cross_validate(
    X: pl.DataFrame, Y: pl.DataFrame, model_name: str, cv: int = 5, random_state: int = 0, **kw
) -> dict:
    Xnp = X.to_numpy()
    Ynp = _clean_inf(Y.to_numpy())
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    rmses, accs = [], []
    for tr, va in kf.split(Xnp):
        est = get_estimator(model_name, random_state=random_state, **kw)
        est.fit(Xnp[tr], Ynp[tr])
        y_pred = est.predict(Xnp[va])
        rmses.append(_rmse_mean(Ynp[va], y_pred))
        accs.append(_top1_acc(Ynp[va], y_pred))
    return {
        "model": model_name,
        "cv": cv,
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "top1_mean": float(np.mean(accs)),
        "top1_std": float(np.std(accs)),
        "folds": [{"rmse": float(r), "top1": float(a)} for r, a in zip(rmses, accs)],
    }

def compare_models(
    X: pl.DataFrame, Y: pl.DataFrame, models: list[str], cv: int = 5, random_state: int = 0
) -> list[dict]:
    return [cross_validate(X, Y, m, cv=cv, random_state=random_state) for m in models]

# ---------- artefacts ----------

def save_artifacts(out_dir: str, model, X_test: pl.DataFrame, Y_test: pl.DataFrame):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    import pickle, numpy as np
    with open(out / "model.pkl", "wb") as f: pickle.dump(model, f)
    np.save(out / "X_test.npy", X_test.to_numpy())
    np.save(out / "Y_test.npy", Y_test.to_numpy())
    np.save(out / "Y_pred.npy", predict(model, X_test))
