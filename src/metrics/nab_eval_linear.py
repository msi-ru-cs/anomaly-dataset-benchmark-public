"""
Original linear NAB implementation retained for comparison.

The benchmark currently uses the sigmoid implementation in nab_eval.py,
which matches the likelihood tuning scorer.
"""

# src/metrics/nab_eval.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

@dataclass
class NabProfile:
    w_tp: float = 1.0   # reward per detected GT window (scaled by earliness)
    w_fp: float = 0.11  # penalty per FP window
    w_fn: float = 1.0   # penalty per FN window

def _group_events(idx: np.ndarray, flag: np.ndarray) -> List[Tuple[int, int]]:
    """
    Collapse contiguous 1s in 'flag' (following 'idx' order) into [start_idx, end_idx] pairs.
    'idx' must be a strictly increasing integer-like array (e.g., t_idx).
    """
    if len(idx) == 0:
        return []
    events = []
    in_evt = False
    s = None
    for i in range(len(idx)):
        if flag[i] and not in_evt:
            in_evt = True
            s = idx[i]
        elif (not flag[i]) and in_evt:
            e = idx[i-1]
            events.append((s, e))
            in_evt = False
    if in_evt:
        events.append((s, idx[-1]))
    return events

def _overlaps(a: Tuple[int,int], b: Tuple[int,int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])

def _earliest_detection_in_window(gt: Tuple[int,int], pred_pos_idx: np.ndarray) -> Optional[int]:
    """Return earliest predicted t_idx inside gt window [s,e], else None."""
    s, e = gt
    # pred_pos_idx sorted
    left = np.searchsorted(pred_pos_idx, s, side="left")
    if left < len(pred_pos_idx):
        t = pred_pos_idx[left]
        if t <= e:
            return int(t)
    return None

def _linear_reward(s: int, e: int, t: int) -> float:
    """Reward in [0,1] where 1 at s, 0 at e (linear decay)."""
    if e <= s:
        return 1.0 if t <= s else 0.0
    if t <= s:
        return 1.0
    if t >= e:
        return 0.0
    return 1.0 - (t - s) / float(e - s)

def _event_counts(gt_windows: List[Tuple[int,int]], pred_windows: List[Tuple[int,int]]) -> Dict[str,int]:
    tp = 0
    matched_gt = set()
    matched_pred = set()
    for gi, g in enumerate(gt_windows):
        hit = False
        for pi, p in enumerate(pred_windows):
            if _overlaps(g, p):
                hit = True
                matched_gt.add(gi)
                matched_pred.add(pi)
        if hit:
            tp += 1
    fn = len(gt_windows) - tp
    fp = len(pred_windows) - len(matched_pred)
    return {"tp": tp, "fp": fp, "fn": fn}

def event_f1_from_counts(c: Dict[str,int]) -> Dict[str,float]:
    tp, fp, fn = c.get("tp",0), c.get("fp",0), c.get("fn",0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}

def nab_score(
    gt_windows: List[Tuple[int,int]],
    pred_pos_idx: np.ndarray,
    profile: NabProfile = NabProfile()
) -> Dict[str, float]:
    """
    Compute normalized NAB-like score:
      - Reward earliest detection inside each GT window with linear earliness.
      - FP penalty for any predicted index not inside any GT window (counted once per predicted *event*).
      - FN penalty for any GT window without detection.
    Normalization: (S_algo - S_null) / (S_opt - S_null)
    """
    # Precompute predicted windows from indices
    if pred_pos_idx.size == 0:
        pred_windows = []
    else:
        ones = np.ones_like(pred_pos_idx, dtype=int)
        pred_windows = _group_events(pred_pos_idx, ones)

    # Raw algorithm score
    S_algo = 0.0
    # Rewards for GT windows
    for g in gt_windows:
        t = _earliest_detection_in_window(g, pred_pos_idx)
        if t is None:
            S_algo -= profile.w_fn
        else:
            S_algo += profile.w_tp * _linear_reward(g[0], g[1], t)
    # FP penalties (one per predicted window that doesn't overlap any GT)
    for p in pred_windows:
        if not any(_overlaps(p, g) for g in gt_windows):
            S_algo -= profile.w_fp

    # Null score: never detect (all FNs)
    S_null = - profile.w_fn * float(len(gt_windows))

    # Optimal score: detect at start for each GT, no FPs
    S_opt = profile.w_tp * float(len(gt_windows))

    # Normalize
    denom = (S_opt - S_null)
    if abs(denom) < 1e-12:
        score_norm = 0.0
    else:
        score_norm = (S_algo - S_null) / denom

    return {
        "nab_raw": S_algo,
        "nab_null": S_null,
        "nab_opt": S_opt,
        "nab_norm": score_norm,
    }

def evaluate_series_events(
    df_scores: pd.DataFrame,
    detector: str = "err",
    test_only: bool = True,
    profile: NabProfile = NabProfile(),
) -> Dict[str, object]:
    """
    Evaluate one series (scores.csv) for a given detector in {'err','lik','md'}.
    Returns counts, event F1, and NAB scores.
    """
    if detector not in {"err", "lik", "md"}:
        raise ValueError("detector must be one of: 'err','lik','md'")

    # Filter split
    df = df_scores.copy()
    if test_only and "split" in df.columns:
        df = df[df["split"] == "test"]
    df = df.sort_values("t_idx")

    # Ground-truth & predicted flags
    y = df["y_true"].astype(int).to_numpy()
    pred = df[f"pred_{detector}"].astype(int).to_numpy()
    t_idx = df["t_idx"].astype(int).to_numpy()

    # Events
    gt_windows   = _group_events(t_idx, y)
    pred_windows = _group_events(t_idx, pred)

    # Event counts
    counts = _event_counts(gt_windows, pred_windows)
    f1 = event_f1_from_counts(counts)

    # NAB-like score (using predicted positive indices)
    pred_pos_idx = t_idx[pred == 1]
    nab = nab_score(gt_windows, pred_pos_idx, profile=profile)

    out = {
        "n_gt_events": len(gt_windows),
        "n_pred_events": len(pred_windows),
        **{f"evt_{k}": v for k, v in counts.items()},      # evt_tp, evt_fp, evt_fn
        **{f"evt_{k}": v for k, v in f1.items()},          # evt_precision, evt_recall, evt_f1
        **nab,                                            # nab_raw, nab_null, nab_opt, nab_norm
    }
    return out
