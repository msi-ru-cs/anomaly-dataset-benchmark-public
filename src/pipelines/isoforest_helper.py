# src/pipelines/isoforest_helper.py
import numpy as np

def fit_isoforest_and_score(model, X_train_w, X_all_w):
    """
    model: IsolationForestWrapper
    X_train_w: (N_train, T, D) clean train windows
    X_all_w:   (N_all,   T, D) all windows

    Returns:
      errs: (N_all,) anomaly score per window (higher = more anomalous)
    """
    X_train_flat = X_train_w.reshape(X_train_w.shape[0], -1)
    X_all_flat   = X_all_w.reshape(X_all_w.shape[0], -1)

    model.fit(X_train_flat)
    errs = model.score_samples(X_all_flat)  # wrapper already inverts sklearn score
    errs = np.asarray(errs, dtype=np.float32)

    return errs
