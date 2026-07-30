# src/pipelines/rethreshold.py
import os, sys, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_cfg(cfg: dict, path: str, default=None):
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def main():
    ap = argparse.ArgumentParser(description="Re-threshold series/*/scores.csv in a run_dir")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--inplace", action="store_true", help="Overwrite scores.csv (else write scores_re.csv)")
    # Optional CLI overrides
    ap.add_argument("--std_k", type=float, default=None)
    ap.add_argument("--two_sided", type=str, default=None)   # "true"/"false"
    ap.add_argument("--use_percentile", type=str, default=None)
    ap.add_argument("--percentile", type=float, default=None)
    ap.add_argument("--lik_threshold", type=float, default=None)
    ap.add_argument("--md_percentile", type=float, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_cfg(args.config)

    # Pull settings (allow CLI override)
    use_pct = (str(args.use_percentile).lower()=="true") if args.use_percentile is not None else bool(get_cfg(cfg,"scoring.use_percentile", False))
    percentile = args.percentile if args.percentile is not None else float(get_cfg(cfg,"scoring.percentile", 0.99))
    mode = str(get_cfg(cfg,"scoring.mode","fixed")).lower()
    std_k = args.std_k if args.std_k is not None else float(get_cfg(cfg,"scoring.std_k",3.0))
    two_sided = (str(args.two_sided).lower()=="true") if args.two_sided is not None else bool(get_cfg(cfg,"scoring.two_sided", False))
    fixed_thr = float(get_cfg(cfg,"scoring.threshold",0.5))

    lik_thr = args.lik_threshold if args.lik_threshold is not None else float(get_cfg(cfg,"likelihood.threshold",0.9))
    md_pct  = args.md_percentile if args.md_percentile is not None else float(get_cfg(cfg,"mahalanobis.threshold_percentile",0.99))

    series_scores = sorted((run_dir/"series").rglob("scores.csv"))
    if not series_scores:
        print(f"[STOP] no scores.csv under {run_dir}/series")
        sys.exit(0)

    for p in series_scores:
        df = pd.read_csv(p)
        if "split" not in df.columns or "y_true" not in df.columns:
            print(f"[WARN] skipping (missing split/y_true): {p}")
            continue
        # Use train rows as train reference; approximate "clean" by y_true==0
        tr = df[df["split"]=="train"].copy()
        tr_clean_err = tr.loc[tr["y_true"]==0, "err"].to_numpy(dtype=float)
        tr_md = tr["md"].to_numpy(dtype=float)

        errs = df["err"].to_numpy(dtype=float)
        md   = df["md"].to_numpy(dtype=float)
        lik  = df["lik"].to_numpy(dtype=float)

        # ---- pred_err
        if use_pct:
            thr = float(np.percentile(tr_clean_err, 100*percentile)) if tr_clean_err.size else float(np.percentile(errs, 100*percentile))
            pred_err = (errs >= thr).astype(int)
            err_src = f"percentile({100*percentile:.1f})"
        else:
            if mode == "stddev":
                mu = float(np.mean(tr_clean_err)) if tr_clean_err.size else float(np.mean(errs))
                sigma = float(np.std(tr_clean_err, ddof=0)) if tr_clean_err.size else float(np.std(errs, ddof=0))
                thr_hi = mu + std_k*sigma
                thr_lo = max(0.0, mu - std_k*sigma)
                pred_err = ((errs >= thr_hi) | ((two_sided) & (errs <= thr_lo))).astype(int)
                err_src = f"stddev(mu={mu:.3g},sigma={sigma:.3g},k={std_k},two_sided={two_sided})"
            else:
                thr = fixed_thr
                pred_err = (errs >= thr).astype(int)
                err_src = "fixed"
        # ---- pred_lik
        pred_lik = (lik > lik_thr).astype(int) # aligned with Tuning code
        # ---- pred_md
        md_thr = float(np.percentile(tr_md, 100*md_pct)) if tr_md.size else float(np.percentile(md, 100*md_pct))
        pred_md = (md >= md_thr).astype(int)

        out = df.copy()
        out["pred_err"] = pred_err
        out["pred_lik"] = pred_lik
        out["pred_md"]  = pred_md

        if args.inplace:
            out.to_csv(p, index=False)
            with open(p.parent/"rethreshold_meta.json","w",encoding="utf-8") as f:
                json.dump({"err_source": err_src, "lik_threshold": lik_thr, "md_percentile": md_pct}, f, indent=2)
            print(f"[WRITE] {p}")
        else:
            out.to_csv(p.parent/"scores_re.csv", index=False)
            print(f"[WRITE] {p.parent/'scores_re.csv'}")

    print("[DONE] rescoring complete.")
if __name__ == "__main__":
    main()
