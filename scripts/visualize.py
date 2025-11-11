import sys, os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from rte_challenge.viz import save_figures

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()
    save_figures(args.out_dir, args.cache_dir)
