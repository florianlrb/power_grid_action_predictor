
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import polars as pl
from tqdm import tqdm

import grid2op
from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction
from grid2op.Environment import Environment

# Optional LightSim (subject's requirements); fallback to default backend if missing
try:
    from lightsim2grid import LightSimBackend
    _HAVE_LIGHTSIM = True
except Exception:
    LightSimBackend = None  # type: ignore
    _HAVE_LIGHTSIM = False

# ----- helpers (cache) --------------------------------------------------------

def _key_from_params(**kwargs) -> str:
    items = sorted(kwargs.items())
    blob = repr(items).encode("utf-8")
    import hashlib
    return hashlib.sha1(blob).hexdigest()[:10]

def _cached_paths(cache_dir: Path, key: str) -> tuple[Path, Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (
        cache_dir / f"features_{key}.parquet",
        cache_dir / f"targets_{key}.parquet",
        cache_dir / f"actions_{key}.json",
    )

# ----- main functions -----------------------------------------------

def extract_features(obs: BaseObservation) -> pl.DataFrame:
    """Flatten a Grid2Op observation into a single-row Polars DataFrame.
    Matches the intent of the provided main.py scaffold.
    """
    parts = []
    names = []
    def append(arr, prefix):
        if arr is None: return
        arr = np.asarray(arr).astype(float).ravel()
        parts.append(arr)
        names.extend([f"{prefix}_{i}" for i in range(arr.size)]) # on ajoute les noms des features
    append(getattr(obs, "gen_p", None), "gen_p")
    append(getattr(obs, "gen_q", None), "gen_q")
    append(getattr(obs, "load_p", None), "load_p")
    append(getattr(obs, "load_q", None), "load_q")
    append(getattr(obs, "topo_vect", None), "topo")
    append(getattr(obs, "rho", None), "rho")
    vec = np.concatenate(parts) if parts else np.zeros(1)
    return pl.DataFrame([vec], schema=names, orient="row")


def create_realistic_observation(
    episode_count: int,
    env: Environment,
) -> list[BaseObservation]:
    """Collect observations by traversing episodes with the 'do nothing' action.
    Keeps the main.py function signature and doc intent.
    """
    list_obs: list[BaseObservation] = []
    for _ in tqdm(range(episode_count), desc="episodes"):
        obs = env.reset()
        list_obs.append(obs)
        # API-robust bound across Grid2Op versions
        max_ts_attr = getattr(env.chronics_handler, "max_timestep", None)
        try:
            max_steps = int(max_ts_attr()) if callable(max_ts_attr) else int(max_ts_attr)
        except Exception:
            # conservative fallback
            max_steps = 2000
        t = 0
        done = False
        while not done and t < max_steps:
            obs, reward, done, info = env.step(env.action_space())  # do-nothing
            if done: break
            list_obs.append(obs)
            t += 1
    return list_obs


def create_training_data(
    list_obs: list[BaseObservation],
    all_actions: list[BaseAction],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (df_features, df_targets) as Polars DataFrames, in the spirit of main.py.
    df_targets[i, j] = max(rho) after applying action j to observation i (inf if not converged).
    """
    X_rows = []
    Y_rows = []
    for obs in tqdm(list_obs, total=len(list_obs), desc="simulate_actions"):
        sim = obs.get_simulator()
        scores = []
        for act in all_actions:
            pred = sim.predict(act=act)
            n_obs = pred.current_obs
            if getattr(pred, "converged", True):
                scores.append(float(np.max(n_obs.rho)))
            else:
                scores.append(float(np.inf))
        Y_rows.append(scores)
        X_rows.append(extract_features(obs))
    df_features = pl.concat(X_rows) if X_rows else pl.DataFrame()
    df_targets = pl.DataFrame(Y_rows, orient="row")
    df_targets.columns = [f"rho_{i}" for i in range(df_targets.width)]
    return df_features, df_targets

# ----- convenience: generate + cache s -

def generate_and_cache(
    cache_dir: str = "data/cache",
    episode_count: int = 1,
    n_actions: int = 20,
    force: bool = False,
    grid_case: str = "l2rpn_case14_sandbox",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """High-level entrypoint with caching. Optionally writes df_features.parquet / df_targets.parquet
    to mimic the original main.py behavior.
    """
    cache_dir_p = Path(cache_dir)
    key = _key_from_params(case=grid_case, episodes=episode_count, n_actions=n_actions)
    fX, fY, fA = _cached_paths(cache_dir_p, key)

    if (fX.exists() and fY.exists()) and not force:
        df_features = pl.read_parquet(fX)
        df_targets = pl.read_parquet(fY)
        print("Using cached data. You can use --force to force regeneration of data.")
    else:
        backend = LightSimBackend() if _HAVE_LIGHTSIM else None
        env = grid2op.make(grid_case, backend=backend) if backend else grid2op.make(grid_case)

        all_actions = [env.action_space.sample() for _ in range(n_actions)]
        all_actions.append(env.action_space())

        # debug/trace 
        print("Example action:", all_actions[0])

        list_obs = create_realistic_observation(episode_count, env)
        df_features, df_targets = create_training_data(list_obs, all_actions)

        df_features.write_parquet(fX)
        df_targets.write_parquet(fY)
        with fA.open("w", encoding="utf-8") as f:
            json.dump([{"index": i, "repr": str(a)} for i, a in enumerate(all_actions)],
                      f, indent=2, ensure_ascii=False)


    return df_features, df_targets
