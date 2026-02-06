# -*- coding: utf-8 -*-
import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm

def compute_lik_from_err(df: pd.DataFrame, long_w: int, short_w: int) -> pd.Series:
    """
    LIK model (simple, effective):
      1) rolling mean/std of ERR with window=long_w (past-only)
      2) z = |(err - mu)/sigma|
      3) lik_raw = Phi(z)            # increases with deviation (0.5~normal, ->1 when extreme)
      4) lik = SMA(lik_raw, short_w) # optional smoothing
    """
    err = pd.to_numeric(df["err"], errors="coerce").fillna(0.0).astype(float)

    mu = err.rolling(window=long_w, min_periods=max(5, long_w//5)).mean()
    sd = err.rolling(window=long_w, min_periods=max(5, long_w//5)).std(ddof=0).replace(0, np.nan)
    z  = (err - mu).abs() / (sd + 1e-12)

    lik_raw = pd.Series(norm.cdf(z), index=df.index)  # higher => more “anomalous”
    if short_w and short_w > 1:
        lik = lik_raw.rolling(window=short_w, min_periods=1).mean()
    else:
        lik = lik_raw

    return lik.clip(0, 1)

def process_series_csv(csv_path: Path, long_w: int, short_w: int):
    df = pd.read_csv(csv_path)
    if "err" not in df.columns:
        return False

    lik = compute_lik_from_err(df, long_w, short_w)
    df["lik"] = lik

    # clear any stale predictions; they will be re-written by rethreshold
    for c in ("pred_lik",):
        if c in df.columns:
            df[c] = 0

    df.to_csv(csv_path, index=False)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--long_window", type=int, required=True)
    ap.add_argument("--short_window", type=int, required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    series_root = run_dir / "series"
    n_ok = 0
    for csv in series_root.rglob("scores.csv"):
        if process_series_csv(csv, args.long_window, args.short_window):
            n_ok += 1
    print(f"[recompute_lik] updated {n_ok} series csv files.")

if __name__ == "__main__":
    sys.exit(main())
