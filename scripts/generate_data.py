import sys, os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from rte_challenge.data import generate_and_cache

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--episodes", type=int, default=1, dest="episode_count")
    ap.add_argument("--actions", type=int, default=20, dest="n_actions")
    ap.add_argument("--case", default="l2rpn_case14_sandbox")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--write-main-files", action="store_true", help="Also write df_features/df_targets parquet like main.py")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    dfX, dfY = generate_and_cache(
        cache_dir=args.cache_dir,
        episode_count=args.episode_count,
        n_actions=args.n_actions,
        force=args.force,
        grid_case=args.case,
        write_main_style_files=args.write_main_files,
        out_dir=args.out_dir,
    )
    print("X shape:", dfX.shape, "| Y shape:", dfY.shape)
