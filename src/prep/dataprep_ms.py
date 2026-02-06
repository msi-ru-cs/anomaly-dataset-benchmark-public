# src/prep/dataprep_ms.py
"""
Microsoft Cloud Monitoring dataset preprocessor.

Goal:
- Keep the SAME normalized output schema we use for NAB:
  columns = [timestamp, value, series_id, label]
  where series_id = relative path under cfg.ms.data_dir (e.g., "application-crash-rate-1/app1-01.csv")

Key differences vs NAB:
- MS CSVs ALREADY contain labels (TimeStamp, Value, Label), so no external windows.json is needed.
"""

from pathlib import Path
from typing import Dict, Any
import json
import pandas as pd
import matplotlib.pyplot as plt

try:
    # If Hydra/OmegaConf is available, cfg may be an OmegaConf object.
    from omegaconf import DictConfig
except Exception:  # pragma: no cover
    DictConfig = Any  # type: ignore


def _cfg_get(cfg, path: str, default=None):
    """
    Safe getter for both OmegaConf objects and plain dicts.
    """
    cur = cfg
    for k in path.split("."):
        try:
            # OmegaConf or simple namespace
            if hasattr(cur, k):
                cur = getattr(cur, k)
            elif isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        except Exception:
            return default
    return cur


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize MS columns:
      TimeStamp, Value, Label  ->  timestamp, value, label
    Tolerates minor variants (e.g., "time", "anomaly", "is_anomaly").
    """
    # Build a case-insensitive map
    lower_map = {c.lower(): c for c in df.columns}
    ts_col = lower_map.get("timestamp") or lower_map.get("time") or list(df.columns)[0]
    val_col = lower_map.get("value") or list(df.columns)[1]
    lab_col = lower_map.get("label") or lower_map.get("anomaly") or lower_map.get("is_anomaly")

    out = df[[ts_col, val_col, lab_col]].rename(
        columns={ts_col: "timestamp", val_col: "value", lab_col: "label"}
    )

    # Parse & clean
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Force binary int labels (0/1)
    out["label"] = (out["label"].astype(float) > 0).astype(int)
    return out


def _plot_series(df: pd.DataFrame, out_png: Path, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 3.0))
    ax = plt.gca()
    ax.plot(df["timestamp"], df["value"], linewidth=1.0)

    # shade anomaly spans
    lab = df["label"].to_numpy()
    ts = df["timestamp"].to_numpy()
    in_span = False
    s_idx = 0
    for i, v in enumerate(lab):
        if v == 1 and not in_span:
            in_span = True
            s_idx = i
        if v == 0 and in_span:
            ax.axvspan(ts[s_idx], ts[i - 1], alpha=0.25)
            in_span = False
    if in_span:
        ax.axvspan(ts[s_idx], ts[len(lab) - 1], alpha=0.25)

    ax.set_title(title)
    ax.set_xlabel("timestamp")
    ax.set_ylabel("value")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def run_preprocess(cfg: DictConfig):
    """
    Entry point used by src/main.py when dataset.name == "ms".
    Reads all CSVs recursively under cfg.ms.data_dir and writes normalized CSVs to cfg.dataset.prepared_dir.
    Also writes a simple manifest and quick plots.
    """
    data_dir = Path(_cfg_get(cfg, "ms.data_dir", ""))
    prepared_root = Path(_cfg_get(cfg, "dataset.prepared_dir", "output/prepared/ms"))
    plots_root = Path("output") / "plots" / str(_cfg_get(cfg, "dataset.name", "ms"))

    if not data_dir.exists():
        raise FileNotFoundError(f"[MS] data_dir not found: {data_dir}")

    files = list(data_dir.rglob("*.csv"))
    print(f"[MS] scanning {data_dir} -> {len(files)} CSVs")

    total_rows = 0
    total_pos = 0
    count_series = 0

    for p in files:
        rel = p.relative_to(data_dir).as_posix()
        df = pd.read_csv(p)
        df = _normalize_df(df)
        df["series_id"] = rel  # match NAB style: directory/filename repeats on each row

        out_csv = prepared_root / rel
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df[["timestamp", "value", "series_id", "label"]].to_csv(out_csv, index=False)

        count_series += 1
        total_rows += len(df)
        total_pos += int(df["label"].sum())

        # optional plot
        _plot_series(df, plots_root / (rel + ".png"), title=rel)
        print(f"[MS] wrote {out_csv} | rows={len(df)} | pos={int(df['label'].sum())}")

    manifest = {
        "dataset": str(_cfg_get(cfg, "dataset.name", "ms")),
        "prepared_dir": str(prepared_root),
        "plots_dir": str(plots_root),
        "schema": ["timestamp", "value", "series_id", "label"],
        "count_series": int(count_series),
        "total_rows": int(total_rows),
        "total_pos": int(total_pos),
    }
    (prepared_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[MS] manifest -> {prepared_root / 'manifest.json'}")

    return prepared_root
