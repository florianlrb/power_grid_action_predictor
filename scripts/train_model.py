import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from pathlib import Path
import polars as pl

from rte_challenge.model import train_test_split, fit_model, evaluate, save_artifacts

def pick_matching_pair(cache_dir: str):
    """Trouve la paire features_<key>.parquet / targets_<key>.parquet la plus récente,
    en s'assurant que la clé est identique des deux côtés."""
    cdir = Path(cache_dir)
    feats = sorted(cdir.glob("features_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not feats:
        raise SystemExit(f"Aucun fichier features_*.parquet dans {cache_dir}. Lance d'abord la génération.")
    for f in feats:
        key = f.name.replace("features_", "").replace(".parquet", "")
        t = cdir / f"targets_{key}.parquet"
        if t.exists():
            return f, t
    raise SystemExit("Aucun targets_<clé>.parquet correspondant trouvé. "
                     "Nettoie le cache ou régénère avec --force pour recréer une paire.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=0)
    args = ap.parse_args()

    fX, fY = pick_matching_pair(args.cache_dir)
    print("Using pair:\n ", fX, "\n ", fY)

    X = pl.read_parquet(fX)
    Y = pl.read_parquet(fY)
    if X.height != Y.height:
        raise SystemExit(f"Mismatch de lignes: X={X.height}, Y={Y.height}. "
                         "Supprime les vieux caches ou relance la génération avec --force.")

    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=args.test_size, random_state=args.random_state)
    model = fit_model(Xtr, Ytr)
    metrics = evaluate(model, Xte, Yte)
    save_artifacts(args.out_dir, model, Xte, Yte)

    print("Metrics:", metrics)
