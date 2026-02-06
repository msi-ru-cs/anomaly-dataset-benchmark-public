import os
import glob
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

def list_csvs(root: str) -> List[str]:
    """Recursively list all CSV files under root."""
    return sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))

def train_test_split_sequence(df: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Sequence-preserving split: first part = train, last part = test.
    Always returns non-empty train and test when len(df) >= 2.
    """
    n = len(df)
    if n <= 1:
        return df.copy(), df.iloc[0:0].copy(), n  # edge case
    split = max(1, min(n - 1, int(round(n * train_ratio))))
    return df.iloc[:split].copy(), df.iloc[split:].copy(), split

def make_windows(arr: np.ndarray, seq_len: int, step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn [T, D] (or [T]) into sliding windows [N, seq_len, D] and return last indices per window.
    """
    if arr.ndim == 1:
        arr = arr[:, None]
    T, D = arr.shape
    if T < seq_len:
        return np.empty((0, seq_len, D)), np.array([], dtype=int)

    xs, idxs = [], []
    i = seq_len
    while i <= T:
        xs.append(arr[i - seq_len:i])
        idxs.append(i - 1)
        i += max(1, int(step))
    return np.stack(xs), np.array(idxs, dtype=int)

def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Precision/Recall/F1/Accuracy with safe zero-division handling."""
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    m = min(len(y_true), len(y_pred))
    yt, yp = y_true[:m], y_pred[:m]
    if (yp.sum() == 0) and (yt.sum() == 0):
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0}
    return {
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
        "accuracy": float(accuracy_score(yt, yp)),
    }

def choose_device(kind: str = "auto") -> str:
    """
    TF-only device selector that returns 'gpu' or 'cpu'.
    kind: 'auto' | 'cpu' | 'gpu' | 'cuda'  (cuda treated as gpu)
    """
    try:
        import tensorflow as tf  # noqa
        gpus = tf.config.list_physical_devices("GPU")
        has_gpu = bool(gpus)
    except Exception:
        has_gpu = False

    k = (kind or "auto").lower()
    if k == "cpu":
        return "cpu"
    if k in ("gpu", "cuda"):
        return "gpu" if has_gpu else "cpu"
    # auto
    return "gpu" if has_gpu else "cpu"
