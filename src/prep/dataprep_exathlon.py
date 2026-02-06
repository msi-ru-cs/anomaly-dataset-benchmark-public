# src/prep/dataprep_exathlon.py
# Exathlon dataprep → produces labeled CSVs similar to NAB
# Handles csv/tsv/parquet, their gz variants, and csv/tsv inside .zip

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import DictConfig


# =========================
# Small debug helpers
# =========================
def _dbg_head(df: pd.DataFrame, n: int = 3) -> str:
    with pd.option_context("display.max_columns", None, "display.width", 160):
        return df.head(n).to_string(index=False)


def _print_series_summary(rel: str, df: pd.DataFrame):
    n = len(df)
    n_pos = int(df["label"].sum()) if "label" in df.columns else 0
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    print(f"[SUMMARY] {rel} | rows={n} | positives={n_pos} | range=[{ts_min} -> {ts_max}]")


# =========================
# Span utils (for plotting)
# =========================
def _find_anomaly_spans(df: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    if "label" not in df.columns or df.empty:
        return spans
    lab = df["label"].to_numpy()
    ts = df["timestamp"].to_numpy()
    in_span = False
    start_idx = 0
    for i, v in enumerate(lab):
        if v == 1 and not in_span:
            in_span = True
            start_idx = i
        if v == 0 and in_span:
            spans.append((ts[start_idx], ts[i - 1]))
            in_span = False
    if in_span:
        spans.append((ts[start_idx], ts[len(lab) - 1]))
    return spans


def _plot_series(df: pd.DataFrame, out_png: Path, title: str):
    # choose y
    if "value" in df.columns:
        y_name = "value"  # univariate path
    else:
        metric_cols = [c for c in df.columns if c not in ("timestamp", "label", "series_id")]
        # pick the first numeric metric column (fallback to first if none typed numeric yet)
        y_name = None
        for c in metric_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                y_name = c
                break
        if y_name is None and metric_cols:
            y_name = metric_cols[0]
        if not y_name:
            # nothing to plot
            print(f"[PLOT] no numeric metric columns for {title}; skip")
            return

    # optional light downsample (keep to ~4k points for speed)
    step = max(1, len(df) // 4000)
    d = df.iloc[::step]

    fig = plt.figure(figsize=(11, 3.2))
    ax = plt.gca()
    ax.plot(d["timestamp"], d[y_name], linewidth=1.0)
    for s, e in _find_anomaly_spans(d):
        ax.axvspan(s, e, alpha=0.25, ymin=0.0, ymax=1.0)
    ax.set_title(f"{title}  ·  y={y_name}")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("value")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)



# =========================
# Label helpers
# =========================
def _normalize_windows(win_obj):
    pairs = []
    if isinstance(win_obj, dict):
        for key in ("windows", "anomalies", "intervals", "root_cause", "extended_effect"):
            if key in win_obj and isinstance(win_obj[key], list):
                win_obj = win_obj[key]
                break
    if isinstance(win_obj, list):
        if not win_obj:
            return pairs
        first = win_obj[0]
        if isinstance(first, dict):
            for w in win_obj:
                s = w.get("start") or w.get("Start") or w.get("begin") or w.get("ts_start")
                e = w.get("end") or w.get("End") or w.get("finish") or w.get("ts_end")
                if s is not None and e is not None:
                    pairs.append((s, e))
        elif isinstance(first, (list, tuple)) and len(first) >= 2:
            for w in win_obj:
                pairs.append((w[0], w[1]))
    return pairs


def _load_label_index(label_dir_or_file: Path) -> Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """
    Accepts either a directory (scan *.json/*.ndjson) or a CSV/JSON file with windows.
    Build mapping: loose key (series/file hint) -> list of (start,end) timestamps.
    """
    idx: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    p = label_dir_or_file

    # Single CSV with windows? (Exathlon ships a ground_truth.csv in the repo)
    if p.is_file() and p.suffix.lower() == ".csv":
        print(f"[LABEL] reading CSV labels at: {p}")
        df = pd.read_csv(p)
        # Expect columns like: key,start,end OR app,metric,start,end ...
        cols = {c.lower(): c for c in df.columns}
        # Try common shapes
        key_col = (cols.get("key") or cols.get("series") or cols.get("file") or
           cols.get("trace") or cols.get("trace_name") or  # added
           cols.get("app") or cols.get("application") or None)
        start_col = cols.get("start") or cols.get("ts_start") or cols.get("begin")
        end_col = cols.get("end") or cols.get("ts_end") or cols.get("finish")
        if not (key_col and start_col and end_col):
            # If app + metric present, concat them to a key
            if cols.get("app") and cols.get("metric") and start_col and end_col:
                key_col = "__am__"
                df[key_col] = df[cols["app"]].astype(str) + "::" + df[cols["metric"]].astype(str)
            else:
                print("[LABEL] CSV columns not recognized; skipping label index build.")
                return {}

        for k, grp in df.groupby(key_col):
            spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
            for _, r in grp.iterrows():
                s = pd.to_datetime(r[start_col], errors="coerce", utc=True)
                e = pd.to_datetime(r[end_col], errors="coerce", utc=True)
                if pd.isna(s) or pd.isna(e):
                    continue
                spans.append((s.tz_convert(None), e.tz_convert(None)))
            if spans:
                idx[str(k)] = spans
        print(f"[LABEL] built index for {len(idx)} keys (CSV)")
        return idx

    # Otherwise, scan directory for JSONs
    label_dir = p
    print(f"[LABEL] scanning {label_dir} for *.json / *.ndjson ...")
    files = list(label_dir.rglob("*.json")) + list(label_dir.rglob("*.ndjson"))
    for f in files:
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            print(f"[WARN] could not parse labels at {f}")
            continue

        if isinstance(obj, dict):
            for k, v in obj.items():
                wins = _normalize_windows(v)
                spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
                for s, e in wins:
                    s = pd.to_datetime(s, errors="coerce", utc=True)
                    e = pd.to_datetime(e, errors="coerce", utc=True)
                    if pd.isna(s) or pd.isna(e):
                        continue
                    spans.append((s.tz_convert(None), e.tz_convert(None)))
                if spans:
                    idx[k] = spans
        elif isinstance(obj, list):
            wins = _normalize_windows(obj)
            spans = []
            for s, e in wins:
                s = pd.to_datetime(s, errors="coerce", utc=True)
                e = pd.to_datetime(e, errors="coerce", utc=True)
                if pd.isna(s) or pd.isna(e):
                    continue
                spans.append((s.tz_convert(None), e.tz_convert(None)))
            if spans:
                idx[f.stem] = spans

    print(f"[LABEL] built index for {len(idx)} keys")
    return idx


def _match_spans_for(rel: str, label_idx: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]) \
        -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    # exact
    if rel in label_idx:
        return label_idx[rel]
    stem = Path(rel).stem
    if stem in label_idx:
        return label_idx[stem]
    # heuristic: look for key substring
    for k, spans in label_idx.items():
        if k in rel or Path(k).stem in rel:
            return spans
    return []


# =========================
# File discovery (incl .zip)
# =========================
def _iter_data_files(root: Path):
    # regular files
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        suf = "".join(p.suffixes).lower()
        if suf.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz", ".parquet")):
            yield ("file", p, None)
    # csv/tsv inside zip
    for z in root.rglob("*.zip"):
        try:
            with zipfile.ZipFile(z) as zf:
                for name in zf.namelist():
                    low = name.lower()
                    if low.endswith(".csv") or low.endswith(".tsv"):
                        yield ("zip", z, name)
        except Exception as e:
            print(f"[WARN] cannot open zip {z}: {e}")


# =========================
# Loading raw Exathlon
# =========================
def _as_datetime(col: pd.Series) -> pd.Series:
    # numeric → infer epoch unit by magnitude
    if pd.api.types.is_numeric_dtype(col):
        s = pd.to_numeric(col, errors="coerce").astype("float64")
        mx = float(np.nanmax(s)) if len(s) else 0.0
        if mx > 1e13:   # ns
            return pd.to_datetime(s, unit="ns", errors="coerce")
        if mx > 1e11:   # us
            return pd.to_datetime(s, unit="us", errors="coerce")
        if mx > 1e9:    # ms
            return pd.to_datetime(s, unit="ms", errors="coerce")
        return pd.to_datetime(s, unit="s", errors="coerce")  # seconds
    # strings
    return pd.to_datetime(col, errors="coerce")


def _load_exathlon_files(
    data_dir: Path,
    timestamp_cols: Optional[List[str]],
    mode: str = "univariate"
) -> Dict[str, pd.DataFrame]:
    print(f"[LOAD] scanning {data_dir} for data files ...")
    out: Dict[str, pd.DataFrame] = {}
    count_seen = 0

    for kind, p, member in _iter_data_files(data_dir):
        count_seen += 1
        rel_base = p.relative_to(data_dir).as_posix()
        rel_src = f"{rel_base}::{member}" if member else rel_base

        # read
        try:
            if kind == "file":
                suf = "".join(p.suffixes).lower()
                if suf.endswith(".parquet"):
                    df = pd.read_parquet(p)
                elif suf.endswith(".tsv") or suf.endswith(".tsv.gz"):
                    df = pd.read_csv(p, sep="\t", compression="infer")
                else:
                    df = pd.read_csv(p, compression="infer")
            else:
                with zipfile.ZipFile(p) as zf:
                    data = zf.read(member)
                sep = "\t" if str(member).lower().endswith(".tsv") else ","
                df = pd.read_csv(BytesIO(data), sep=sep)
        except Exception as e:
            print(f"[WARN] skip {rel_src}: {e}")
            continue

        # timestamp handling
        ts_candidates = (timestamp_cols or
                         ["timestamp", "time", "ts", "datetime", "DateTime", "event_time"])
        ts_col = next((c for c in ts_candidates if c in df.columns), None)

        if ts_col is not None:
            df["timestamp"] = _as_datetime(df[ts_col])
        else:
            df = df.reset_index().rename(columns={"index": "offset"})
            df["timestamp"] = pd.to_datetime(df["offset"].astype("int64"), unit="s", errors="coerce")

        before = len(df)
        df = (
            df.dropna(subset=["timestamp"])
              .sort_values("timestamp")
              .drop_duplicates("timestamp", keep="last")
              .reset_index(drop=True)
        )

        # If timestamps are degenerate (all identical / almost identical), synthesize a monotonic clock
        if len(df) > 1 and (df["timestamp"].nunique() <= max(1, int(len(df) * 0.001))):
            df["timestamp"] = pd.to_datetime(np.arange(len(df)), unit="s")

        dropped = before - len(df)

        non_ts_cols = [c for c in df.columns if c not in (ts_col, "timestamp")]

        if mode == "multivariate":
            df = df[["timestamp"] + non_ts_cols].copy()
            df.columns = ["timestamp"] + [str(c).strip().replace(" ", "_") for c in non_ts_cols]
            df["series_id"] = rel_src
            out[rel_src] = df
            print(f"[LOAD] {rel_src} rows={len(df)} (dropped {dropped} bad timestamps) [MULTI]")
        else:
            # One univariate series per metric column
            drop_like = {"label", "series_id"}
            value_cols = [c for c in non_ts_cols if c not in drop_like]
            for vc in value_cols:
                ser = df[["timestamp", vc]].rename(columns={vc: "value"}).copy()
                ser["series_id"] = f"{rel_src}::{vc}"
                out_key = f"{rel_src}__{vc}"
                out[out_key] = ser
            if value_cols:
                print(f"[LOAD] {rel_src} -> {len(value_cols)} metrics, rows={len(df)} (dropped {dropped})")

    print(f"[LOAD] found {count_seen} files (including zip members); produced {len(out)} series")
    if not out:
        raise RuntimeError(
            f"No usable data files under {data_dir}. "
            f"Expected csv/tsv/parquet or zip with csv/tsv."
        )
    return out


# =========================
# Apply labels
# =========================
def _apply_labels_exathlon(
    datasets: Dict[str, pd.DataFrame],
    label_idx: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]],
) -> Dict[str, pd.DataFrame]:
    labeled: Dict[str, pd.DataFrame] = {}
    for rel, df in datasets.items():
        df = df.copy()
        df["label"] = 0
        spans = _match_spans_for(rel, label_idx)
        pos = 0
        for s, e in spans:
            mask = (df["timestamp"] >= s) & (df["timestamp"] <= e)
            df.loc[mask, "label"] = 1
            pos += int(mask.sum())
        labeled[rel] = df
        print(f"[LABEL] {rel} -> windows={len(spans)} | pos={pos}")
        # print(_dbg_head(df))
    return labeled


# =========================
# Write
# =========================
def _write_labeled(root: Path, rel: str, df: pd.DataFrame, mode: str):
    out_path = root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[WRITE] {rel} -> {out_path}  (#rows={len(df)}, #pos={int(df['label'].sum())}, mode={mode})")


# =========================
# Orchestrate
# =========================
def run_preprocess(cfg: DictConfig):
    """
    cfg fields expected:
      cfg.dataset.name: "exathlon"
      cfg.exathlon.data_dir: path to raw traces root
      cfg.exathlon.label_dir: path to ground-truth file/dir (CSV or JSONs)
      cfg.exathlon.mode: "univariate" | "multivariate"
      cfg.exathlon.timestamp_cols: optional list of timestamp column names to try
      cfg.prepared_dir: output root for prepared CSVs
    """
    data_dir = Path(cfg.exathlon.data_dir)
    label_path = Path(cfg.exathlon.label_dir)
    labeled_root = Path(cfg.prepared_dir)
    plots_root = Path("output") / "plots" / str(cfg.dataset.name)

    mode = str(getattr(cfg.exathlon, "mode", "univariate")).lower()
    ts_cols = list(getattr(cfg.exathlon, "timestamp_cols", [])) or None

    print("[PREP] invoked: src.prep.dataprep_exathlon")
    print(f"[PREP] dataset={cfg.dataset.name} | mode={mode}")
    print(f"[PREP] data_dir={data_dir}")
    print(f"[PREP] label_dir/file={label_path}")
    print(f"[PREP] prepared_dir={labeled_root}")
    print(f"[PREP] plots_dir={plots_root}")

    if not data_dir.exists():
        raise FileNotFoundError(f"Exathlon data_dir not found: {data_dir}")
    if not label_path.exists():
        raise FileNotFoundError(f"Exathlon label file/dir not found: {label_path}")

    # 1) load
    datasets = _load_exathlon_files(data_dir, timestamp_cols=ts_cols, mode=mode)

    # 2) label index
    label_idx = _load_label_index(label_path)

    # 3) apply labels
    labeled = _apply_labels_exathlon(datasets, label_idx)

    # 4) save labeled + plot
    count = 0
    print(f"[WRITE] target root: {labeled_root}")
    for rel, df in labeled.items():
        out_rel = rel if rel.endswith(".csv") else (rel + ".csv")
        _write_labeled(labeled_root, out_rel, df, mode=mode)
        _print_series_summary(rel, df)

        # plots (non-fatal)
        out_png = plots_root / (Path(out_rel).as_posix() + ".png")
        # print(f"[PLOT] skipped: {out_png}")
        # (intentionally do not call plot_series if you don't want it)
        # """
        try:
            _plot_series(df, out_png, title=rel)
            print(f"[PLOT] {out_png}")
        except Exception as e:
            print(f"[PLOT] skip {rel}: {e}")
        count += 1
        # """

    # 5) manifest
    cols = []
    if labeled:
        cols = list(next(iter(labeled.values())).columns)
    schema = ["series_id", "timestamp", "label"]
    if "value" in cols:
        schema = ["series_id", "timestamp", "value", "label"]

    manifest = {
        "dataset": cfg.dataset.name,
        "prepared_dir": str(labeled_root),
        "plots_dir": str(plots_root),
        "schema": schema,
        "mode": mode,
        "count_series": len(labeled),
        "total_rows": int(sum(len(df) for df in labeled.values())) if labeled else 0,
        "total_pos": int(sum(int(df["label"].sum()) for df in labeled.values())) if labeled else 0,
    }
    labeled_root.mkdir(parents=True, exist_ok=True)
    (labeled_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[WRITE] wrote {count} series")
    print(f"[DONE] wrote manifest: {labeled_root / 'manifest.json'}")

    return labeled_root
