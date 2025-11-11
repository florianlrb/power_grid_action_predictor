import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from pathlib import Path
import polars as pl

from rte_challenge.model import (
    train_test_split, fit_model, evaluate, save_artifacts,
    cross_validate, compare_models
)

def pick_matching_pair(cache_dir: str):
    """Paire la plus récente features_<key>.parquet / targets_<key>.parquet (clé identique)."""
    cdir = Path(cache_dir)
    feats = sorted(cdir.glob("features_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not feats:
        raise SystemExit(f"Aucun features_*.parquet dans {cache_dir}. Lance d'abord la génération.")
    for f in feats:
        key = f.name.replace("features_", "").replace(".parquet", "")
        t = cdir / f"targets_{key}.parquet"
        if t.exists():
            return f, t
    raise SystemExit("Aucun targets_<clé>.parquet correspondant trouvé. Regénère avec --force.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--model", default="dtree",
                    help="Un nom ou une liste séparée par des virgules: dtree,rf,ridge,gbr")
    ap.add_argument("--cv", type=int, default=0,
                    help="Si >0: lance une cross-val (KFold) avec k=cv. Si plusieurs modèles sont fournis, compare.")
    args = ap.parse_args()

    fX, fY = pick_matching_pair(args.cache_dir)
    print("Using pair:\n ", fX, "\n ", fY)

    X = pl.read_parquet(fX)
    Y = pl.read_parquet(fY)

    # Auto-fix si Y a été transposé par un vieux cache
    if X.height != Y.height and Y.width == X.height:
        print(" Y semble transposé; correction automatique (transpose).")
        Y = Y.transpose()
        # renommer colonnes proprement
        Y.columns = [f"rho_{i}" for i in range(Y.width)]

    if X.height != Y.height:
        raise SystemExit(f"Mismatch de lignes: X={X.height}, Y={Y.height}. Regénère le cache avec --force.")

    model_names = [s.strip() for s in args.model.split(",") if s.strip()]

    # --- Cross-validation éventuelle ---
    if args.cv and args.cv > 0:
        if len(model_names) > 1:
            print(f"\n== Cross-validation (cv={args.cv}) — comparaison modèles ==")
            results = compare_models(X, Y, model_names, cv=args.cv, random_state=args.random_state)
            # tri par RMSE croissante
            results = sorted(results, key=lambda d: d["rmse_mean"])
            for r in results:
                print(f"- {r['model']:>5s}: RMSE={r['rmse_mean']:.4f}±{r['rmse_std']:.4f} | "
                      f"Top1={r['top1_mean']:.3f}±{r['top1_std']:.3f}")
            # on prend le meilleur pour l'entraînement final
            best_model = results[0]["model"]
            print(f"\n→ Meilleur modèle (RMSE): {best_model}")
            chosen = best_model
        else:
            # single model CV
            m = model_names[0]
            r = cross_validate(X, Y, m, cv=args.cv, random_state=args.random_state)
            print(f"\n== Cross-validation (cv={args.cv}) — {m} ==")
            print(f"RMSE={r['rmse_mean']:.4f}±{r['rmse_std']:.4f} | Top1={r['top1_mean']:.3f}±{r['top1_std']:.3f}")
            chosen = m
    else:
        chosen = model_names[0]

    # --- Split train/test + entraînement final ---
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=args.test_size, random_state=args.random_state)
    model = fit_model(Xtr, Ytr, model_name=chosen, random_state=args.random_state)
    metrics = evaluate(model, Xte, Yte)
    save_artifacts(args.out_dir, model, Xte, Yte)

    print(f"\n== Final model: {chosen} ==")
    print("Metrics:", metrics)
