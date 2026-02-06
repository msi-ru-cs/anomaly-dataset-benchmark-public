# -*- coding: utf-8 -*-
"""
W&B sweep runner for NAB likelihood tuning per group.

- Uses EXACT same scoring logic as test_nab_evaluation_on_NAB_dataset_perGroup.ipynb.
- Assumes each sweep is run on a single subgroup directory, e.g.
    working_data/subgroups/realAWSCloudwatch

Sweep config example (sweep.yaml):

program: "lik_sweep_runner.py"
method: bayes
project: hyper_tune_aws
entity: mylab

metric:
  name: dataset_raw_sum
  goal: maximize

parameters:
  series_dir:
    value: "./working_data/subgroups/artificialNoAnomaly"
  profile:
    value: standard
  likelihood.long_window:
    distribution: int_uniform
    min: 300
    max: 500
  likelihood.short_window:
    distribution: int_uniform
    min: 10
    max: 60
  likelihood.threshold:
    distribution: uniform
    min: 0.99
    max: 0.9999
"""

import os
import glob
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from scipy.stats import norm
import wandb


# ---------------------------------------------------------------------------
# Helper: robust access to W&B config keys (supports dotted "likelihood.xxx")
# ---------------------------------------------------------------------------
def get_cfg_value(cfg, key, default=None):
    # direct attr
    if hasattr(cfg, key):
        return getattr(cfg, key)
    # direct dict-style
    if key in cfg:
        return cfg[key]
    # dotted notation
    if "." in key:
        top, rest = key.split(".", 1)
        sub = None
        if hasattr(cfg, top):
            sub = getattr(cfg, top)
        elif top in cfg:
            sub = cfg[top]
        if isinstance(sub, dict) and rest in sub:
            return sub[rest]
    return default


# ---------------------------------------------------------------------------
# 1) Load only the columns we want (same as notebook)
# ---------------------------------------------------------------------------
def load_scores_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    keep = ["timestamp", "split", "y_true", "err", "x_value"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].sort_values("timestamp").reset_index(drop=True)
    return df





# ---------------------------------------------------------------------------
# 1) apply_split_mode
# ---------------------------------------------------------------------------
def apply_split_mode(
    df: pd.DataFrame,
    split_mode: str,
    rel: Path,
    long_win: int,
    short_win: int,
    threshold: float,
    cfg=None,
) -> pd.DataFrame:

    """
    split_mode:
      - all: use all rows
      - train: use split=='train'
      - test: use split=='test'
      - validation: take last part of TRAIN with length equal to test size (if available),
                    otherwise estimate using ratios (default test/train = 0.30/0.70).
    """
    split_mode = (split_mode or "all").lower()

    if split_mode == "all":
        return df

    if "split" not in df.columns:
        print(f"[WARN] split_mode={split_mode} but no 'split' column in {rel}; using all rows.")
        return df

    if split_mode in ("train", "test"):
        before = len(df)
        out = df[df["split"] == split_mode].copy()
        if out.empty:
            print(f"[WARN] No rows with split='{split_mode}' in {rel}; skipping.")
            return pd.DataFrame()
        print(f"[INFO] {rel}: kept {len(out)}/{before} rows with split='{split_mode}'.")
        return out

    if split_mode == "validation":
        train_df = df[df["split"] == "train"].copy()
        test_df = df[df["split"] == "test"].copy()

        if train_df.empty:
            print(f"[WARN] split_mode=validation but no train rows in {rel}; skipping.")
            return pd.DataFrame()

        # If test exists, match validation length to test length
        if not test_df.empty:
            val_len = len(test_df)
            source = "test_size"
        else:
            # Estimate from ratios: val_len ≈ train_size * (test_ratio/train_ratio)
            test_ratio = float(get_cfg_value(cfg, "validation.test_ratio", 0.30)) if cfg is not None else 0.30
            train_ratio = float(get_cfg_value(cfg, "validation.train_ratio", 0.70)) if cfg is not None else 0.70
            val_len = int(round(len(train_df) * (test_ratio / max(train_ratio, 1e-9))))
            source = f"ratio({test_ratio}/{train_ratio})"

        val_len = max(1, min(val_len, len(train_df)))
        val_df = train_df.tail(val_len).copy()

        # Compute likelihood on full train history, but score only validation tail
        train_with_pred = add_lik_and_pred(train_df, long_win, short_win, threshold)
        val_with_pred = train_with_pred.tail(val_len).copy()

        print(
            f"[INFO] {rel}: validation from TRAIN tail, val_len={val_len} "
            f"(source={source}), train_len={len(train_df)}."
        )
        return val_with_pred

    print(f"[WARN] Unknown split_mode={split_mode} in {rel}; using all rows.")
    return df




# ---------------------------------------------------------------------------
# 2) Real NAB-style likelihood from error (same as notebook)
# ---------------------------------------------------------------------------
def compute_anomaly_likelihood(
    historical_recon_error,
    anomaly_threshold,
    anomaly_long_window_length,
    anomaly_short_window_length,
):
    """
    Compute anomaly likelihood using NAB formula.
    Returns (likelihood, is_anomaly).
    """
    if len(historical_recon_error) <= anomaly_long_window_length:
        return 0.5, False

    wide_data_window = historical_recon_error[-anomaly_long_window_length:]
    narrow_data_window = historical_recon_error[-anomaly_short_window_length:]

    mean_of_wide_data_window = mean(wide_data_window)
    stdev_of_wide_data_window = float(np.std(wide_data_window, ddof=0))

    if (not np.isfinite(stdev_of_wide_data_window)) or stdev_of_wide_data_window <= 1e-12:
        return 0.5, False

    z = (mean(narrow_data_window) - mean_of_wide_data_window) / stdev_of_wide_data_window
    likelihood = 0.5 + 0.5 * (1 - norm.sf(z))
    is_anomaly = likelihood >= anomaly_threshold
    return float(likelihood), bool(is_anomaly)


def compute_nab_anom_lik(
    err_series,
    long_win: int,
    short_win: int,
    warmup_val: float = 0.5,
    lo: float = 0.5,
    hi: float = 0.9993,
    threshold: float = 0.9993,
):
    """
    Wrapper version: computes a likelihood vector
    by repeatedly calling compute_anomaly_likelihood() on the running history.
    """
    errs = np.asarray(err_series, dtype=float)
    n = len(errs)

    lik = np.zeros(n, dtype=float)
    hist = []

    for i, e in enumerate(errs):
        hist.append(e)
        likelihood, _ = compute_anomaly_likelihood(
            hist,
            anomaly_threshold=threshold,
            anomaly_long_window_length=long_win,
            anomaly_short_window_length=short_win,
        )

        if i < long_win:
            lik_val = warmup_val
        else:
            lik_val = np.clip(likelihood, lo, hi)

        lik[i] = lik_val

    return lik


# ---------------------------------------------------------------------------
# 3) Process one series: add lik + pred_anom (no disk write needed for scoring)
# ---------------------------------------------------------------------------
def add_lik_and_pred(df: pd.DataFrame, long_win: int, short_win: int, thr: float) -> pd.DataFrame:
    df = df.copy()
    df["lik"] = compute_nab_anom_lik(df["err"], long_win, short_win)
    df["pred_anom"] = (df["lik"] > thr).astype(int)
    return df


# ---------------------------------------------------------------------------
# 4) NAB scoring helpers (exactly as notebook)
# ---------------------------------------------------------------------------
PROFILES = {
    "standard": {"A_tp": 1.0, "A_fp": -0.11, "A_fn": -1.0},
    "low_fp": {"A_tp": 1.0, "A_fp": -0.22, "A_fn": -1.0},
    "low_fn": {"A_tp": 1.0, "A_fp": -0.11, "A_fn": -2.0},
}


def labels_to_windows(ts: pd.Series, y: pd.Series):
    windows, in_win, start = [], False, None
    for t, v in zip(ts, y):
        if v == 1 and not in_win:
            in_win, start = True, t
        elif v == 0 and in_win:
            in_win = False
            windows.append((start, t))
    if in_win:
        windows.append((start, ts.iloc[-1]))
    return windows


def _seconds(x):
    if hasattr(x, "total_seconds"):
        return float(x.total_seconds())
    return float(x)


def _abs_nearest_boundary_seconds(t, windows):
    best = None
    for ws, we in windows:
        d = min(abs(_seconds(t - ws)), abs(_seconds(t - we)))
        best = d if (best is None or d < best) else best
    return 0.0 if best is None else float(best)


def tp_weight(y_sec, L_sec, A_tp):
    L = max(float(L_sec), 1e-9)
    k = 6.0 / L
    sig = 1.0 / (1.0 + math.exp(+k * y_sec))
    val = A_tp * min(1.0, 2.0 * sig)
    return max(0.0, min(A_tp, val))


def fp_weight(d_sec, L_sec, A_fp):
    L = max(float(L_sec), 1e-9)
    k = 6.0 / L
    sig = 1.0 / (1.0 + math.exp(-k * d_sec))
    val = A_fp * (2.0 * sig - 1.0)
    return min(0.0, max(A_fp, val))


def score_one_file(df: pd.DataFrame, profile: str = "standard"):
    p = PROFILES[profile]
    A_tp, A_fp, A_fn = p["A_tp"], p["A_fp"], p["A_fn"]

    ts = df["timestamp"]
    gts = df["y_true"].fillna(0).astype(int)
    prs = df["pred_anom"].fillna(0).astype(int)

    windows = labels_to_windows(ts, gts)
    if len(windows) == 0:
        tp = int(((gts == 1) & (prs == 1)).sum())
        fp = int(((gts == 0) & (prs == 1)).sum())
        fn = int(((gts == 1) & (prs == 0)).sum())
        tn = int(((gts == 0) & (prs == 0)).sum())
        return {
            "score_raw": 0.0,
            "score_norm": 0.0,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "n_win": 0,
            "tp_win": 0,
            "fn_win": 0,
            "tn_win": 0,
            "best_possible": 0.0,
            "null_score": 0.0,
        }

    w_lens_sec = [max(_seconds(we - ws), 1e-9) for (ws, we) in windows]
    avg_wlen_sec = float(np.mean(w_lens_sec))
    det_times = ts[prs == 1].tolist()

    total, used_tp = 0.0, set()

    # TP per window (earliest only)
    for (ws, we), L in zip(windows, w_lens_sec):
        inside = [t for t in det_times if ws <= t <= we]
        if inside:
            t_hit = min(inside)
            used_tp.add(t_hit)
            y_sec = _seconds(t_hit - ws)
            total += tp_weight(y_sec, L, A_tp)

    # FN once per missed window
    missed = len(windows) - len(used_tp)
    total += missed * A_fn

    # FP: every detection outside all windows
    for t in det_times:
        if t in used_tp or any(ws <= t <= we for (ws, we) in windows):
            continue
        d = _abs_nearest_boundary_seconds(t, windows)
        total += fp_weight(d, avg_wlen_sec, A_fp)

    best, null = len(windows) * A_tp, len(windows) * A_fn
    nab_norm = 0.0 if best == null else (total - null) / (best - null)

    tp_pts = int(((gts == 1) & (prs == 1)).sum())
    fp_pts = int(((gts == 0) & (prs == 1)).sum())
    fn_pts = int(((gts == 1) & (prs == 0)).sum())
    prec = tp_pts / (tp_pts + fp_pts + 1e-9)
    rec = tp_pts / (tp_pts + fn_pts + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    return {
        "score_raw": total,
        "score_norm": nab_norm,
        "tp": tp_pts,
        "fp": fp_pts,
        "tn": 0,
        "fn": fn_pts,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "n_win": len(windows),
        "tp_win": len(used_tp),
        "fn_win": missed,
        "tn_win": 0,
        "best_possible": best,
        "null_score": null,
    }


# ---------------------------------------------------------------------------
# 5) Main sweep run
# ---------------------------------------------------------------------------
def main():
    wandb.init()
    cfg = wandb.config

    series_dir = Path(get_cfg_value(cfg, "series_dir")).resolve()
    dataset_name = extract_dataset_name_from_series(series_dir)
    print(f"[CFG] dataset_name = {dataset_name}")

    profile = str(get_cfg_value(cfg, "profile", "standard"))

    long_win = int(get_cfg_value(cfg, "likelihood.long_window", 400))
    short_win = int(get_cfg_value(cfg, "likelihood.short_window", 10))
    threshold = float(get_cfg_value(cfg, "likelihood.threshold", 0.9993))
    # NEW: all / train / test
    split_mode = str(get_cfg_value(cfg, "split_mode", "all")).lower()

    print(f"[CFG] series_dir = {series_dir}")
    print(f"[CFG] profile    = {profile}")
    print(f"[CFG] long_win   = {long_win}")
    print(f"[CFG] short_win  = {short_win}")
    print(f"[CFG] threshold  = {threshold}")
    print(f"[CFG] split_mode = {split_mode}")

    if not series_dir.exists():
        raise SystemExit(f"[FATAL] series_dir does not exist: {series_dir}")

    # 1) Discover all CSV files (same pattern as notebook)
    subdirs = sorted(
        [d for d in glob.glob(os.path.join(str(series_dir), "*")) if os.path.isdir(d)]
    )
    all_csvs = []
    for d in subdirs:
        csvs = glob.glob(os.path.join(d, "*.csv"))
        all_csvs.extend(csvs)

    print(f"[INFO] Found {len(subdirs)} series folders.")
    print(f"[INFO] Total CSV files found (recursive): {len(all_csvs)}")

    # 2) Process and score all series
    records = []
    sum_score = 0.0
    sum_best = 0.0
    sum_null = 0.0

    for src in all_csvs:
        rel = Path(src).relative_to(series_dir)
        df = load_scores_csv(src)
        if "err" not in df.columns or "y_true" not in df.columns:
            print(f"[WARN] Missing 'err' or 'y_true' in {rel}; skipping.")
            continue
        # =================NEW==================
        # NEW: filter by split column if requested
        # =================NEW==================
        # NEW: apply split_mode (all/train/test/validation)
        df2 = apply_split_mode(
            df,
            split_mode,
            rel,
            long_win=long_win,
            short_win=short_win,
            threshold=threshold,
            cfg=cfg,
        )

        if df2.empty:
            continue
        df = df2


        if split_mode == "validation":
            df_with_pred = df
        else:
            df_with_pred = add_lik_and_pred(df, long_win, short_win, threshold)
        # =================NEW END==================

        # df_with_pred = add_lik_and_pred(df, long_win, short_win, threshold)  # Updaed in new section above
        res = score_one_file(df_with_pred, profile=profile)

        n_win = res["n_win"]
        p = PROFILES[profile]
        best = n_win * p["A_tp"]
        null = n_win * p["A_fn"]

        sum_score += res["score_raw"]
        sum_best += best
        sum_null += null

        print(
            f"{rel}: "
            f"raw={res['score_raw']:.4f}  "
            f"best={best:.4f}  "
            f"null={null:.4f}  "
            f"nab={res['score_norm']:.4f}  "
            f"nab_x100={res['score_norm']*100:.2f}  "
            f"win_tp={res['tp_win']}  "
            f"win_fn={res['fn_win']}  "
            f"win_tn={res['tn_win']}"
        )

        records.append(
            {
                "series": str(rel),
                **res,
            }
        )

    if sum_best == sum_null:
        dataset_nab = 0.0
    else:
        dataset_nab = (sum_score - sum_null) / (sum_best - sum_null)

    print("\n=== DATASET-LEVEL NAB ===")
    print("raw_sum_score :", sum_score)
    print("raw_sum_best  :", sum_best)
    print("raw_sum_null  :", sum_null)
    print("NAB (0-1)     :", dataset_nab)
    print("NAB (0-100)   :", dataset_nab * 100.0)

    # Log to W&B (metric for sweep)
    wandb.log({"dataset_raw_sum": sum_score, "dataset_nab": dataset_nab})

    wandb.summary["dataset_raw_sum"] = sum_score
    wandb.summary["dataset_nab"] = dataset_nab
    wandb.summary["sum_best"] = sum_best
    wandb.summary["sum_null"] = sum_null
    wandb.summary["long_window"] = long_win
    wandb.summary["short_window"] = short_win
    wandb.summary["threshold"] = threshold

    # -----------------------------------------------------------------------
    # 6) Save/update BEST result CSV under working_data/best_results/
    #    One file per group (group_name = last part of series_dir).
    #    Only overwritten if this run is better (higher dataset_raw_sum).
    # -----------------------------------------------------------------------
    # Log to W&B (metric for sweep)
    wandb.log({"dataset_raw_sum": sum_score, "dataset_nab": dataset_nab})

    wandb.summary["dataset_raw_sum"] = sum_score
    wandb.summary["dataset_nab"] = dataset_nab
    wandb.summary["sum_best"] = sum_best
    wandb.summary["sum_null"] = sum_null
    wandb.summary["long_window"] = long_win
    wandb.summary["short_window"] = short_win
    wandb.summary["threshold"] = threshold

    # Turn per-series records into a DataFrame
    per_series_df = pd.DataFrame.from_records(records) if records else pd.DataFrame()

    # Add a DATASET summary row similar to your notebook output
    if not per_series_df.empty:
        dataset_row = {
            "series": "__DATASET__",
            "score_raw": sum_score,
            "score_norm": dataset_nab,
            "tp": per_series_df["tp"].sum(),
            "fp": per_series_df["fp"].sum(),
            "tn": per_series_df["tn"].sum(),
            "fn": per_series_df["fn"].sum(),
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "n_win": per_series_df["n_win"].sum(),
            "tp_win": per_series_df["tp_win"].sum(),
            "fn_win": per_series_df["fn_win"].sum(),
            "tn_win": per_series_df["tn_win"].sum(),
            "best_possible": sum_best,
            "null_score": sum_null,
        }
        per_series_df = pd.concat(
            [per_series_df, pd.DataFrame([dataset_row])],
            ignore_index=True,
        )

    # -----------------------------------------------------------------------
    # 6) Save/update BEST result CSV under working_data/best_results/
    #    One pair of files per group:
    #      best_<group>.csv                  -> best params + dataset stats
    #      best_<group>_series_scores.csv    -> per-series + dataset row
    # -----------------------------------------------------------------------
    group_name = series_dir.name  # e.g. "realTweets" or "realAWSCloudwatch"

    # Optional: allow an override name from sweep config, else use group_name
    result_name = get_cfg_value(cfg, "result_name", None) or group_name

    root = Path.cwd()
    # results_root = root / "working_data" / "best_results" / "likelihood"
    results_root = root / "working_data" / "best_results" / dataset_name / "likelihood"


    results_root.mkdir(parents=True, exist_ok=True)

    params_csv = results_root / f"best_{result_name}.csv"
    series_csv = results_root / f"best_{result_name}_series_scores.csv"

    previous_best = None
    if params_csv.exists():
        try:
            prev_df = pd.read_csv(params_csv)
            if not prev_df.empty and "dataset_raw_sum" in prev_df.columns:
                previous_best = float(prev_df["dataset_raw_sum"].iloc[0])
        except Exception:
            previous_best = None

    if (previous_best is None) or (sum_score > previous_best):
        print(
            f"[RESULT] Updating best result for group '{result_name}': "
            f"{previous_best} -> {sum_score}"
        )

        # 6a) best params + dataset-level numbers (one row)
        out_row = pd.DataFrame(
            [
                {
                    "group": result_name,
                    "series_dir": str(series_dir),
                    "profile": profile,
                    "likelihood_long_window": long_win,
                    "likelihood_short_window": short_win,
                    "likelihood_threshold": threshold,
                    "dataset_raw_sum": sum_score,
                    "dataset_nab": dataset_nab,
                    "sum_best": sum_best,
                    "sum_null": sum_null,
                    "n_series": len(records),
                    "wandb_run_id": wandb.run.id if wandb.run else "",
                    "wandb_run_name": wandb.run.name if wandb.run else "",
                }
            ]
        )
        out_row.to_csv(params_csv, index=False)

        # 6b) per-series + DATASET row, exactly like notebook table
        if not per_series_df.empty:
            per_series_df.to_csv(series_csv, index=False)
            print(f"[RESULT] Wrote per-series scores to: {series_csv}")
    else:
        print(
            f"[RESULT] Existing best for group '{result_name}' "
            f"is better or equal ({previous_best} >= {sum_score}); not updating."
        )


def extract_dataset_name_from_series(series_dir: Path) -> str:
    parts = series_dir.parts
    dataset = None
    for i, p in enumerate(parts):
        if p.lower() == "subgroups" and i + 1 < len(parts):
            dataset = parts[i + 1]
            break
    return dataset




if __name__ == "__main__":
    main()
