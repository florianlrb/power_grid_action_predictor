import os
import sys
import json
import subprocess
from pathlib import Path

import pytest

# Skip propre si Grid2Op n'est pas installé (ex: env partiel)
pytest.importorskip("grid2op")

@pytest.fixture(scope="session")
def project_root() -> Path:
    # racine = dossier qui contient 'scripts' et 'rte_challenge'
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "scripts").exists() and (p / "rte_challenge").exists():
            return p
    return Path.cwd()

@pytest.fixture(scope="session")
def pyexe() -> str:
    return sys.executable  # l'interpréteur courant

@pytest.fixture(scope="session")
def fast_cache(tmp_path_factory, project_root, pyexe):
    """
    Génère un petit cache rapide (1 épisode, 2 actions) dans un dossier temporaire,
    que tous les tests peuvent réutiliser.
    """
    cache_dir = tmp_path_factory.mktemp("cache")
    cmd = [
        pyexe, str(project_root / "scripts" / "generate_data.py"),
        "--episodes", "1",
        "--actions", "2",
        "--cache-dir", str(cache_dir),
        "--force",
    ]
    # Important: cwd=project_root pour que les imports relatifs fonctionnent
    subprocess.run(cmd, cwd=project_root, check=True)

    # Retourne le chemin du cache
    return cache_dir

def pick_matching_pair(cache_dir: Path) -> tuple[Path, Path]:
    feats = sorted(cache_dir.glob("features_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    assert feats, "Aucun features_*.parquet généré"
    for f in feats:
        key = f.name.replace("features_", "").replace(".parquet", "")
        t = cache_dir / f"targets_{key}.parquet"
        if t.exists():
            return f, t
    raise AssertionError("Aucun targets_<clé>.parquet correspondant trouvé")
