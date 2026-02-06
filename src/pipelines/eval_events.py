# src/pipelines/eval_events.py
import os, sys, json, argparse
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import yaml

from src.metrics.nab_eval import NabProfile, evaluate_series_events


# ----------------------------
# Config helpers
# ----------------------------
def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_cfg(cfg: dict, path: str, default=None):
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ----------------------------
# FS helpers
# ----------------------------
def _find_series_scores(run_dir: Path) -> List[Path]:
    series_dir = run_dir / "series"
    if not series_dir.exists():
        return []
    return sorted(series_dir.rglob("scores.csv"))

def _mean_safe(x: pd.Series) -> float:
    return float(x.mean()) if len(x) else 0.0


# ----------------------------
# Profiles
# ----------------------------
def _profile_dict_from_cfg(cfg: dict) -> Dict[str, Tuple[float,float,float]]:
    """
    Return dict: name -> (w_tp, w_fp, w_fn)
    NOTE: w_fp is expected as positive magnitude; NabProfile should internally
    apply it as a penalty for FPs (i.e., subtract). Keep signs consistent with
    your src.metrics.nab_eval implementation.
    """
    out: Dict[str, Tuple[float,float,float]] = {}
    rp = get_cfg(cfg, "report.profiles", {}) or {}
    if rp:
        for name, d in rp.items():
            out[str(name)] = (float(d.get("w_tp", 1.0)),
                              float(d.get("w_fp", 0.11)),
                              float(d.get("w_fn", 1.0)))
    else:
        # Defaults if not specified
        out = {
            "standard": (1.0, 0.11, 1.0),
            "low_fn":   (1.0, 0.11, 2.0),
            "low_fp":   (1.0, 0.22, 1.0),
        }
    return out


# ----------------------------
# One profile eval
# ----------------------------
def _eval_one_profile(
    run_dir: Path,
    detector: str,
    test_only: bool,
    prof_name: str,
    w: Tuple[float,float,float],
    anomalies_only: bool,
    append: bool
) -> pd.DataFrame:
    """
    Evaluate all series for one detector + one profile.
    Returns a DataFrame with per-series results and writes
    overall_events_<detector>_<profile>.csv (append or overwrite).
    """
    rows = []
    series_scores = _find_series_scores(run_dir)
    if not series_scores:
        raise SystemExit(f"[STOP] no series/*/scores.csv under {run_dir}")

    profile = NabProfile(*w)
    for p in series_scores:
        # series name relative to series root
        series_name = str(p.parent).split(str(run_dir / "series") + os.sep, 1)[-1]
        df = pd.read_csv(p)
        pred_col = f"pred_{detector}"
        if pred_col not in df.columns:
            # skip series if missing the predictor column
            continue

        res = evaluate_series_events(df, detector=detector, test_only=test_only, profile=profile)
        # 'res' is expected to include keys like:
        # n_gt_events, n_pred_events, evt_tp, evt_fp, evt_fn, evt_precision,
        # evt_recall, evt_f1, nab_raw, nab_null, nab_opt, nab_norm
        rows.append({
            "series": series_name,
            "detector": detector,
            "profile": prof_name,
            **res
        })

    S = pd.DataFrame(rows)

    # Optional filter: keep only series with any anomaly signal
    if anomalies_only and not S.empty:
        keep = (S.get("n_gt_events", 0) > 0) | (S.get("n_pred_events", 0) > 0)
        S = S[keep].reset_index(drop=True)

    # Write per-profile file with detector suffix to avoid overwrites
    out = run_dir / f"overall_events_{detector}_{prof_name}.csv"
    mode = "a" if (append and out.exists()) else "w"
    header = not (append and out.exists())
    S.to_csv(out, index=False, mode=mode, header=header)

    return S


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Event-level & NAB evaluation over a run_dir (supports multi-profile NAB table)."
    )
    ap.add_argument("--run_dir", required=True,
                    help="Path to completed run (contains series/*/scores.csv)")
    ap.add_argument("--config", default="config/config.yaml",
                    help="Training cfg (to read report.detector & metadata)")
    ap.add_argument("--detector", choices=["err","lik","md"],
                    help="Override report.detector (err|lik|md)")
    ap.add_argument("--profiles", default=None,
                    help="Comma list of profile names; defaults from config.report.profiles")
    ap.add_argument("--test_only", action="store_true",
                    help="If set, use only test windows; otherwise use config.report.test_only (default True)")
    ap.add_argument("--append", action="store_true",
                    help="Append to existing per-profile CSVs instead of overwriting")
    ap.add_argument("--anomalies_only", type=int, choices=[0,1],
                    help="1: keep only series with any anomaly/pred event; 0: keep all (default from config, else 0)")
    ap.add_argument("--write_table", action="store_true",
                    help="Append a one-row NAB-style table into nab_table.csv")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"[FATAL] run_dir not found: {run_dir}")
        sys.exit(2)
    cfg = load_cfg(args.config)

    # Detector to score
    detector = (args.detector or str(get_cfg(cfg, "report.detector", "err"))).lower()
    if detector not in {"err", "lik", "md"}:
        print(f"[FATAL] detector must be err|lik|md, got {detector}")
        sys.exit(2)

    # test_only default: True unless config overrides; --test_only forces True
    test_only = bool(get_cfg(cfg, "report.test_only", True))
    if args.test_only:
        test_only = True

    # anomalies_only default: from config, else False
    anomalies_only = bool(get_cfg(cfg, "report.anomalies_only", False))
    if args.anomalies_only is not None:
        anomalies_only = bool(args.anomalies_only)

    # Profiles
    prof_map = _profile_dict_from_cfg(cfg)
    if args.profiles:
        names = [n.strip() for n in args.profiles.split(",") if n.strip()]
        prof_map = {n: prof_map[n] for n in names if n in prof_map}
        if not prof_map:
            print(f"[WARN] none of the requested profiles {names} found in config; using defaults.")

    # Evaluate each profile and collect macro NAB scores
    macro_scores: Dict[str, float] = {}   # name -> macro NAB (%) over the corpus
    per_profile_frames: Dict[str, pd.DataFrame] = {}
    for name, weights in prof_map.items():
        S = _eval_one_profile(
            run_dir,
            detector=detector,
            test_only=test_only,
            prof_name=name,
            w=weights,
            anomalies_only=anomalies_only,
            append=args.append
        )
        per_profile_frames[name] = S

        # ===== CORPUS-LEVEL NAB NORMALIZATION (correct) =====
        # Use summed raw/null/opt across all series, then normalize once.
        need_cols = {"nab_raw", "nab_null", "nab_opt"}
        if not S.empty and need_cols.issubset(S.columns):
            sum_raw  = float(S["nab_raw"].sum())
            sum_null = float(S["nab_null"].sum())
            sum_opt  = float(S["nab_opt"].sum())
            denom    = (sum_opt - sum_null)
            macro_norm = (sum_raw - sum_null) / (denom if abs(denom) > 1e-12 else 1e-12)
            macro_scores[name] = 100.0 * macro_norm
        else:
            # Fallback if needed columns are absent
            macro_scores[name] = 0.0
        # ====================================================

    # If caller wants the single row NAB table, write/append it now
    if args.write_table:
        # Pull hyperparams from config
        window     = int(get_cfg(cfg, "split.seq_len", 64))
        layers     = int(get_cfg(cfg, "model.layers", 1))
        cells      = int(get_cfg(cfg, "model.hidden", 64))
        epochs     = int(get_cfg(cfg, "model.epochs", 10))
        batch_size = int(get_cfg(cfg, "model.batch_size", 128))

        row = {
            "Detector": detector.upper(),
            "Window": window,
            "Layers": layers,
            "Cells": cells,
            "Epocs": epochs,
            "BatchSize": batch_size,
        }
        # Canonical labels (plus any extras)
        labeled = {
            "standard": "Standard profile",
            "low_fn":   "Reward_low_FN",
            "low_fp":   "Reward_low_FP",
        }
        for k, label in labeled.items():
            if k in macro_scores:
                row[label] = round(macro_scores[k], 2)
        for k, v in macro_scores.items():
            if k not in labeled:
                row[k] = round(v, 2)

        out_path = run_dir / "nab_table.csv"
        if out_path.exists():
            pd.DataFrame([row]).to_csv(out_path, mode="a", index=False, header=False)
        else:
            pd.DataFrame([row]).to_csv(out_path, index=False)
        print(f"[TABLE] wrote {out_path}")

    # Aggregate JSON per detector (avoid clobbering across detectors)
    agg_json = run_dir / f"events_aggregate_profiles_{detector}.json"
    with open(agg_json, "w", encoding="utf-8") as f:
        json.dump({k: round(v, 4) for k, v in macro_scores.items()}, f, indent=2)

    print(f"[DONE] detector={detector} | profiles={list(prof_map.keys())} | "
          f"test_only={test_only} | anomalies_only={anomalies_only} | run_dir={run_dir}")

if __name__ == "__main__":
    main()
