# tests/test_generate_data.py
from pathlib import Path
import polars as pl

def _pick_matching_pair(cache_dir: Path):
    feats = sorted(cache_dir.glob("features_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    assert feats, "Aucun features_*.parquet généré"
    for f in feats:
        key = f.name.replace("features_", "").replace(".parquet", "")
        t = cache_dir / f"targets_{key}.parquet"
        if t.exists():
            return f, t
    raise AssertionError("Aucun targets_<clé>.parquet correspondant trouvé")

def test_generate_minimal_dataset(fast_cache):
    fX, fY = _pick_matching_pair(Path(fast_cache))
    X = pl.read_parquet(fX)
    Y = pl.read_parquet(fY)

    # Formes cohérentes
    assert X.height == Y.height, f"X.height={X.height} != Y.height={Y.height}"
    assert Y.width >= 2, "On attend au moins 2 colonnes d'actions (1 action + do-nothing)"
    assert X.width > 0, "Features vides"
