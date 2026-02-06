# -*- coding: utf-8 -*-
"""
Multi-detector sweep runner (LIK / ERR / MD) for NAB, no retraining.

Per W&B run:
  1) Copy base run's series/ -> sweep_runs/<base>/<wandb_run_id>_<ts>/
  2) Write an overlay YAML for the chosen detector and its params
  3) Recompute (for LIK) and threshold to generate predictions
  4) Run eval_events and parse NAB score (corpus-level normalization)
  5) Log results to W&B + sweep_results.csv
"""

import os, sys, json, time, shutil, subprocess as sp, platform, re
from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml
import wandb


# ---------- helpers ----------
def _copy_series_only(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "series").mkdir(parents=True, exist_ok=True)
    ssrc = src / "series"
    if not ssrc.exists():
        raise FileNotFoundError(f"No 'series' folder in base run: {ssrc}")
    for p in ssrc.iterdir():
        tgt = (dst / "series" / p.name)
        if p.is_dir():
            shutil.copytree(p, tgt, dirs_exist_ok=True)
        else:
            shutil.copy2(p, tgt)


def _link_series(src_series: Path, dst_series: Path):
    """Create a directory link/junction to src_series at dst_series."""
    dst_series.parent.mkdir(parents=True, exist_ok=True)
    if dst_series.exists():
        return
    if platform.system() == "Windows":
        try:
            sp.check_call([
                "powershell", "-NoProfile", "-Command",
                f'New-Item -ItemType Junction -Path "{dst_series}" -Target "{src_series}"'
            ])
        except Exception:
            sp.check_call(["cmd", "/c", "mklink", "/J", str(dst_series), str(src_series)])
    else:
        os.symlink(src_series, dst_series, target_is_directory=True)


def _run(cmd: list, cwd: Path = None) -> int:
    print("[CMD]", " ".join(str(x) for x in cmd))
    return sp.call(cmd, cwd=str(cwd) if cwd else None)


def _load_cfg(cfg_path: Path) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_cfg(d: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, sort_keys=False)


def _apply_detector_overlay(base: dict, cfg) -> dict:
    """Return a copy of base config with detector-specific overrides."""
    det = str(cfg.get("detector", "lik")).lower()
    out = dict(base)

    out.setdefault("report", {})
    out["report"]["detector"] = det

    if det == "lik":
        out.setdefault("likelihood", {})
        out["likelihood"]["long_window"]  = int(cfg.get("likelihood.long_window"))
        out["likelihood"]["short_window"] = int(cfg.get("likelihood.short_window"))
        out["likelihood"]["threshold"]    = float(cfg.get("likelihood.threshold"))

    elif det == "err":
        out.setdefault("scoring", {})
        out["scoring"]["use_percentile"] = False
        out["scoring"]["mode"] = out["scoring"].get("mode", "stddev")
        out["scoring"]["std_k"]     = float(cfg.get("scoring.std_k"))
        out["scoring"]["two_sided"] = bool(cfg.get("scoring.two_sided"))
        out["scoring"]["threshold"] = float(cfg.get("scoring.threshold"))

    elif det == "md":
        out.setdefault("mahalanobis", {})
        out["mahalanobis"]["threshold_percentile"] = float(cfg.get("mahalanobis.threshold_percentile"))

    else:
        raise ValueError(f"Unknown detector: {det}")

    return out


def _parse_score(run_dir: Path, det: str, retries: int = 6, delay: float = 0.5):
    """
    Return (NAB_percent, source_hint).
    Robust order (now corpus-level correct):
      1) overall_events_<det>_standard.csv -> corpus-level NAB from nab_raw/null/opt
      2) events_aggregate_profiles_<det>.json -> try flat map OR nested {"profiles": {...}}
      3) nab_table.csv / nab_score_table.csv -> last numeric in last row
    """
    def _read_overall_events(detector: str):
        p = run_dir / f"overall_events_{detector}_standard.csv"
        if not p.exists():
            return None
        try:
            df = pd.read_csv(p)
            need = {"nab_raw", "nab_null", "nab_opt"}
            if need.issubset(set(df.columns)):
                sum_raw  = float(df["nab_raw"].sum())
                sum_null = float(df["nab_null"].sum())
                sum_opt  = float(df["nab_opt"].sum())
                denom    = sum_opt - sum_null
                if abs(denom) < 1e-12:
                    return None
                nab_pct = 100.0 * (sum_raw - sum_null) / denom
                return nab_pct, f"{p.name}:corpus_norm(nab_raw/null/opt)"
        except Exception:
            return None
        return None

    def _read_events_json(detector: str):
        p = run_dir / f"events_aggregate_profiles_{detector}.json"
        if not p.exists():
            return None
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            # Accept both flat {"standard": 57.8, ...} and nested {"profiles": {"standard": {...}}}
            if isinstance(obj, dict):
                # Flat number (percent) for "standard"
                if "standard" in obj and isinstance(obj["standard"], (int, float)):
                    return float(obj["standard"]), f"{p.name}:flat_standard"
                # Nested case
                profs = obj.get("profiles")
                if isinstance(profs, dict):
                    std = profs.get("standard")
                    if isinstance(std, dict):
                        # pick number-like fields
                        for k in ("total", "reward", "score", "normalized", "nab", "percent"):
                            if k in std and isinstance(std[k], (int, float)):
                                return float(std[k]), f"{p.name}:profiles.standard.{k}"
        except Exception:
            return None
        return None

    def _read_table():
        for name in ("nab_table.csv", "nab_score_table.csv"):
            p = run_dir / name
            if p.exists():
                try:
                    lines = p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
                    if lines:
                        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", lines[-1])
                        if nums:
                            return float(nums[-1]), f"{name}:last_numeric"
                except Exception:
                    pass
        return None

    for _ in range(max(1, retries)):
        got = _read_overall_events(det)
        if got: return got
        got = _read_events_json(det)
        if got: return got
        got = _read_table()
        if got: return got
        time.sleep(delay)
    return (None, "not_found")


# ---------- main ----------
def main():
    wandb.init()
    cfg = wandb.config

    base_run = Path(cfg.get("base_run_dir"))
    base_cfg = Path(cfg.get("config_path", "config/config.yaml"))
    copy_mode = str(cfg.get("copy_mode", "series")).lower()
    det = str(cfg.get("detector", "lik")).lower()

    if not base_run.exists():
        raise FileNotFoundError(f"Base run_dir not found: {base_run}")
    if not base_cfg.exists():
        raise FileNotFoundError(f"Config file not found: {base_cfg}")

    sweep_root = Path("sweep_runs") / base_run.name
    sweep_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_dir = sweep_root / f"{wandb.run.id}_{ts}"
    print(f"[INFO] Trial dir: {trial_dir}")

    # Copy / link series
    if copy_mode == "full":
        shutil.copytree(base_run, trial_dir, dirs_exist_ok=True)
    elif copy_mode == "series":
        _copy_series_only(base_run, trial_dir)
    elif copy_mode == "link":
        trial_dir.mkdir(parents=True, exist_ok=True)
        _link_series(base_run / "series", trial_dir / "series")
    else:
        raise ValueError(f"Unknown copy_mode: {copy_mode}")

    # Build overlay config
    base_dict = _load_cfg(base_cfg)
    overlay = _apply_detector_overlay(base_dict, cfg)
    tmp_cfg = trial_dir / "tmp_cfg.yaml"
    _write_cfg(overlay, tmp_cfg)

    py = sys.executable

    # === LIK recompute + threshold + eval ===
    if det == "lik":
        rc = _run([
            py, "-m", "src.pipelines.recompute_lik",
            "--run_dir", str(trial_dir),
            "--long_window", str(int(cfg.get("likelihood.long_window"))),
            "--short_window", str(int(cfg.get("likelihood.short_window")))
        ])
        if rc != 0:
            print("[WARN] recompute_lik returned", rc)

        rc = _run([
            py, "-m", "src.pipelines.rethreshold",
            "--run_dir", str(trial_dir),
            "--config", str(tmp_cfg),
            "--inplace",
            "--lik_threshold", str(float(cfg.get("likelihood.threshold")))
        ])
        if rc != 0:
            print("[WARN] rethreshold returned", rc)

    # === ERR or MD detectors just threshold directly ===
    elif det in ("err", "md"):
        rc = _run([
            py, "-m", "src.pipelines.rethreshold",
            "--run_dir", str(trial_dir),
            "--config", str(tmp_cfg),
            "--inplace"
        ])
        if rc != 0:
            print("[WARN] rethreshold returned", rc)

    # 3) Evaluate (write table for visibility)
    rc = _run([
        py, "-m", "src.pipelines.eval_events",
        "--run_dir", str(trial_dir),
        "--config", str(tmp_cfg),
        "--write_table"  # so we always have nab_table.csv as a fallback
    ])
    if rc != 0:
        print("[WARN] eval returned code", rc)

    # Parse and log score
    score, hint = _parse_score(trial_dir, det=det, retries=6, delay=0.5)
    if score is None:
        score = float("nan")
        print("[ERROR] Could not parse NAB score.")

    wandb.log({
        "detector": det,
        "nab_standard_score": score,  # percent
        "score_source": hint,
        "trial_dir": str(trial_dir),
    })

    out_csv = sweep_root / "sweep_results.csv"
    header = not out_csv.exists()
    pd.DataFrame([{
        "ts": ts,
        "wandb_run_id": wandb.run.id,
        "detector": det,
        "trial_dir": str(trial_dir),
        "nab_standard_score": score,
        "score_source": hint,
        "lik_long": cfg.get("likelihood.long_window"),
        "lik_short": cfg.get("likelihood.short_window"),
        "lik_thr": cfg.get("likelihood.threshold"),
        "err_std_k": cfg.get("scoring.std_k"),
        "err_two_sided": cfg.get("scoring.two_sided"),
        "err_thr": cfg.get("scoring.threshold"),
        "md_thr_pct": cfg.get("mahalanobis.threshold_percentile"),
    }]).to_csv(out_csv, mode="a", header=header, index=False)

    print("[DONE] Score (Standard %):", score, "| From:", hint)
    print("[PATH]", out_csv)


if __name__ == "__main__":
    main()
