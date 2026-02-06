# tools/normalize_exathlon_gt.py
import sys, pandas as pd, numpy as np
from pathlib import Path

def to_dt_numcol(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    m = float(np.nanmax(s)) if len(s) else 0.0
    unit = "s"
    if m > 1e13: unit = "ns"
    elif m > 1e11: unit = "us"
    elif m > 1e9:  unit = "ms"
    return pd.to_datetime(s, unit=unit, errors="coerce")

inp = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else inp.with_name("ground_truth_normalized.csv")

df = pd.read_csv(inp)

# Start always from root_cause_start
df["start"] = to_dt_numcol(df["root_cause_start"])
# End = extended_effect_end if available else root_cause_end
end_source = "extended_effect_end" if "extended_effect_end" in df.columns else "root_cause_end"
df["end"] = to_dt_numcol(df[end_source])

# Final slim file: only what prep needs
df_out = df[["trace_name", "start", "end"]].dropna()
df_out.to_csv(out, index=False)

print(f"[OK] wrote normalized GT: {out}  (end from '{end_source}')")
