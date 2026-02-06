import json
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from omegaconf import DictConfig
import matplotlib.pyplot as plt

# ---------------------------
# Debug helpers
# ---------------------------
def _dbg_head(df: pd.DataFrame, n: int = 3) -> str:
    with pd.option_context("display.max_columns", None, "display.width", 120):
        return df.head(n).to_string(index=False)

def _print_series_summary(rel: str, df: pd.DataFrame):
    n = len(df)
    n_pos = int(df["label"].sum()) if "label" in df.columns else 0
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    print(f"[SUMMARY] {rel} | rows={n} | positives={n_pos} | "
          f"range=[{ts_min} -> {ts_max}]")

# ---------------------------
# Load + label
# ---------------------------
def _load_nab_csvs(data_dir: Path) -> Dict[str, pd.DataFrame]:
    print(f"[LOAD] scanning {data_dir} for *.csv ...")
    out: Dict[str, pd.DataFrame] = {}
    files = list(data_dir.rglob("*.csv"))
    print(f"[LOAD] found {len(files)} files")
    for p in files:
        rel = p.relative_to(data_dir).as_posix()
        df = pd.read_csv(p)
        ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
        val_col = "value" if "value" in df.columns else df.columns[1]
        df = df[[ts_col, val_col]].rename(columns={ts_col: "timestamp", val_col: "value"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        dropped = before - len(df)
        df["series_id"] = rel
        print(f"[LOAD] {rel} rows={len(df)} (dropped {dropped} bad timestamps)")
        out[rel] = df
    return out

def _apply_labels(datasets: Dict[str, pd.DataFrame], label_json: Path) -> Dict[str, pd.DataFrame]:
    print(f"[LABEL] reading windows from: {label_json}")
    spec = json.loads(Path(label_json).read_text(encoding="utf-8"))
    labeled: Dict[str, pd.DataFrame] = {}

    def _normalize_windows(win_obj):
        """Return list of (start, end) strings from various formats."""
        pairs = []
        # common nesting
        if isinstance(win_obj, dict):
            for key in ("windows", "anomalies", "intervals"):
                if key in win_obj and isinstance(win_obj[key], list):
                    win_obj = win_obj[key]
                    break
        # now expect a list
        if isinstance(win_obj, list):
            if len(win_obj) == 0:
                return pairs
            first = win_obj[0]
            # format: [{"start": "...", "end": "..."}, ...]
            if isinstance(first, dict):
                for w in win_obj:
                    s = w.get("start") or w.get("Start") or w.get("begin")
                    e = w.get("end")   or w.get("End")   or w.get("finish")
                    pairs.append((s, e))
            # format: [["start", "end"], ...] or tuples
            elif isinstance(first, (list, tuple)):
                for w in win_obj:
                    if len(w) >= 2:
                        pairs.append((w[0], w[1]))
        return pairs

    for rel, df in datasets.items():
        df = df.copy()
        df["label"] = 0
        win_obj = spec.get(rel, [])
        pairs = _normalize_windows(win_obj)

        pos = 0
        for s, e in pairs:
            s = pd.to_datetime(s, errors="coerce")
            e = pd.to_datetime(e, errors="coerce")
            if pd.isna(s) or pd.isna(e):
                continue
            mask = (df["timestamp"] >= s) & (df["timestamp"] <= e)
            df.loc[mask, "label"] = 1
            pos += int(mask.sum())

        labeled[rel] = df
        print(f"[LABEL] {rel} -> windows={len(pairs)} | pos={pos}")
        print(_dbg_head(df))

    return labeled


# ---------------------------
# Save labeled CSVs
# ---------------------------
def _write_labeled(root: Path, rel: str, df: pd.DataFrame):
    out_path = root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[WRITE] {rel} -> {out_path}  (#rows={len(df)}, #pos={int(df['label'].sum())})")

# ---------------------------
# Plotting
# ---------------------------
def _find_anomaly_spans(df: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return contiguous [start, end] spans where label==1."""
    spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    lab = df["label"].to_numpy()
    ts = df["timestamp"].to_numpy()
    if len(lab) == 0:
        return spans
    in_span = False
    start_idx = 0
    for i, v in enumerate(lab):
        if v == 1 and not in_span:
            in_span = True
            start_idx = i
        if (v == 0 and in_span):
            spans.append((ts[start_idx], ts[i-1]))
            in_span = False
    if in_span:
        spans.append((ts[start_idx], ts[len(lab)-1]))
    return spans

def _plot_series(df: pd.DataFrame, out_png: Path, title: str):
    fig = plt.figure(figsize=(11, 3.2))
    ax = plt.gca()
    ax.plot(df["timestamp"], df["value"], linewidth=1.0)
    spans = _find_anomaly_spans(df)
    for s, e in spans:
        ax.axvspan(s, e, alpha=0.15, color='red', ymin=0.0, ymax=1.0)
    ax.set_title(title)
    ax.set_xlabel("timestamp")
    ax.set_ylabel("value")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

# ---------------------------
# Orchestrate: read -> label -> save -> plot
# ---------------------------
def run_preprocess(cfg: DictConfig):
    data_dir  = Path(cfg.nab.data_dir)
    label_file = Path(cfg.nab.label_file)
    labeled_root = Path(cfg.prepared_dir)        # we use prepared_dir as the labeled output root
    plots_root = Path("output") / "plots" / cfg.dataset.name

    print(f"[PREP] dataset={cfg.dataset.name}")
    print(f"[PREP] data_dir={data_dir}")
    print(f"[PREP] label_file={label_file}")
    print(f"[PREP] prepared_dir={labeled_root}")
    print(f"[PREP] plots_dir={plots_root}")

    if not data_dir.exists():   raise FileNotFoundError(f"NAB data_dir not found: {data_dir}")
    if not label_file.exists(): raise FileNotFoundError(f"NAB label_file not found: {label_file}")

    # 1) load
    datasets = _load_nab_csvs(data_dir)

    # 2) label
    labeled = _apply_labels(datasets, label_file)

    # 3) save labeled + plot
    for rel, df in labeled.items():
        _write_labeled(labeled_root, rel, df)
        _print_series_summary(rel, df)

        # plot to output/plots/<dataset>/<rel>.png
        out_png = plots_root / (rel + ".png")
        _plot_series(df, out_png, title=rel)
        print(f"[PLOT] {out_png}")

    # 4) write a small manifest
    manifest = {
        "dataset": cfg.dataset.name,
        "prepared_dir": str(labeled_root),
        "plots_dir": str(plots_root),
        "schema": ["series_id", "timestamp", "value", "label"],
        "count_series": len(labeled),
        "total_rows": int(sum(len(df) for df in labeled.values())),
        "total_pos": int(sum(int(df["label"].sum()) for df in labeled.values()))
    }
    (labeled_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[DONE] wrote manifest: {labeled_root / 'manifest.json'}")
    return labeled_root
