# src/pipelines/train_tf.py
import os, sys, json, argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml
import tensorflow as tf

from src.metrics.scoring import reconstruction_errors, anomaly_likelihood, mahalanobis_scores

# ==========================
# Model registry (TF/Keras)
# ==========================
MODEL_REGISTRY = {}

def register_model(name):
    def _wrap(fn):
        MODEL_REGISTRY[name] = fn
        return fn
    return _wrap

@register_model("gru_ae")
def _build_gru_ae(input_dim, seq_len, cfg):
    from src.models_tf.gru_ae_tf import build_gru_autoencoder
    return build_gru_autoencoder(
        input_dim=input_dim,
        seq_len=seq_len,
        hidden=int(get_cfg(cfg, "model.hidden", 64)),
        layers_n=int(get_cfg(cfg, "model.layers", 1)),
        dropout=float(get_cfg(cfg, "model.dropout", 0.0)),
        lr=float(get_cfg(cfg, "model.lr", 1e-3)),
        loss=str(get_cfg(cfg, "model.loss", "mse")),
    )

@register_model("tcn")
def _build_tcn(input_dim, seq_len, cfg):
    from src.models_tf.tcn_ae_tf import build_tcn_autoencoder
    return build_tcn_autoencoder(
        input_dim=input_dim,
        seq_len=seq_len,
        hidden=int(get_cfg(cfg, "model.hidden", 64)),
        layers_n=int(get_cfg(cfg, "model.layers", 6)),
        kernel_size=int(get_cfg(cfg, "tcn.kernel_size", 3)),
        dilation_base=int(get_cfg(cfg, "tcn.dilation_base", 2)),
        dropout=float(get_cfg(cfg, "model.dropout", 0.1)),
        lr=float(get_cfg(cfg, "model.lr", 1e-3)),
        loss=str(get_cfg(cfg, "model.loss", "mse")),
    )

@register_model("transformer")
def _build_transformer(input_dim, seq_len, cfg):
    from src.models_tf.transformer_ae_tf import build_transformer_autoencoder
    return build_transformer_autoencoder(
        input_dim=input_dim,
        seq_len=seq_len,
        d_model=int(get_cfg(cfg, "model.hidden", 64)),
        layers_n=int(get_cfg(cfg, "model.layers", 2)),
        heads=int(get_cfg(cfg, "transformer.heads", 4)),
        ff_mult=int(get_cfg(cfg, "transformer.ff_mult", 4)),
        dropout=float(get_cfg(cfg, "model.dropout", 0.1)),
        lr=float(get_cfg(cfg, "model.lr", 1e-3)),
        loss=str(get_cfg(cfg, "model.loss", "mse")),
    )

@register_model("tsmixer")
def _build_tsmixer(input_dim, seq_len, cfg):
    from src.models_tf.tsmixer_ae_tf import build_tsmixer_autoencoder
    return build_tsmixer_autoencoder(
        input_dim=input_dim,
        seq_len=seq_len,
        hidden=int(get_cfg(cfg, "model.hidden", 64)),
        layers_n=int(get_cfg(cfg, "model.layers", 4)),
        time_mlp=int(get_cfg(cfg, "tsmixer.time_mlp", 128)),
        channel_mlp=int(get_cfg(cfg, "tsmixer.channel_mlp", 128)),
        dropout=float(get_cfg(cfg, "model.dropout", 0.1)),
        lr=float(get_cfg(cfg, "model.lr", 1e-3)),
        loss=str(get_cfg(cfg, "model.loss", "mse")),
    )


@register_model("isoforest")
def _build_isoforest(input_dim, seq_len, cfg):
    from src.models_tf.isoforest import IsolationForestWrapper
    return IsolationForestWrapper(
        n_estimators=int(get_cfg(cfg, "isoforest.n_estimators", 300)),
        max_samples=get_cfg(cfg, "isoforest.max_samples", "auto"),
        contamination=get_cfg(cfg, "isoforest.contamination", "auto"),
        max_features=float(get_cfg(cfg, "isoforest.max_features", 1.0)),
        random_state=int(get_cfg(cfg, "isoforest.random_state", 42)),
        n_jobs=int(get_cfg(cfg, "isoforest.n_jobs", -1)),
    )

# ==========================
# Config / logging
# ==========================
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

class Logger:
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"[LOG START] {datetime.now().isoformat()}\n")
    def __call__(self, msg: str):
        print(msg)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(str(msg).rstrip() + "\n")

# ==========================
# Data helpers
# ==========================
def list_csvs(root: str) -> List[str]:
    return [str(p) for p in Path(root).rglob("*.csv")]

def train_test_split_sequence(df: pd.DataFrame, train_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    n = len(df)
    split_idx = max(1, min(n - 1, int(round(n * train_ratio))))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy(), split_idx

def numeric_feature_columns(df: pd.DataFrame, drop: List[str]) -> List[str]:  # type: ignore[name-defined]
    cols = df.select_dtypes(include=["number"]).columns.tolist()
    return [c for c in cols if c not in drop]

def load_labeled_csv(path: str):
    """
    Loads one prepared CSV and returns:
        df, X(float32, NaN-safe), y(np.int8 or None), feat_cols(list[str])

    - Excludes non-feature columns: timestamp, series_id, label
    - Coerces object columns to numeric when possible
    - Fills NaN/Inf and casts to float32 to reduce RAM and avoid NaN losses
    """
    import pandas as pd
    import numpy as np

    df = pd.read_csv(path)

    non_feat = {"timestamp", "series_id", "label"}

    # Prefer project helper if present
    try:
        feat_cols = numeric_feature_columns(df, drop=list(non_feat))  # may not exist
    except Exception:
        feat_cols = None

    if not feat_cols:
        cand = [c for c in df.columns if c not in non_feat]
        for c in cand:
            if df[c].dtype == object:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        feat_cols = [c for c in cand if pd.api.types.is_numeric_dtype(df[c])] or cand

    X = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    y = df["label"].to_numpy(dtype=np.int8) if "label" in df.columns else None

    return df, X, y, feat_cols


# ==========================
# Simple scaler
# ==========================
class TSScaler:
    def __init__(self, kind="minmax", clip_value=None):
        assert kind in ("minmax", "standard")
        self.kind = kind
        self.clip_value = clip_value
        self.fitted = False

    def fit(self, X):
        import numpy as np
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        if self.kind == "minmax":
            self.min_ = np.nanmin(X, axis=0)
            self.max_ = np.nanmax(X, axis=0)
            rng = self.max_ - self.min_
            self.range_ = np.where(np.isfinite(rng) & (rng != 0), rng, 1.0).astype(np.float32)
        else:
            self.mean_ = np.nanmean(X, axis=0).astype(np.float32)
            std = np.nanstd(X, axis=0)
            self.std = np.where(np.isfinite(std) & (std != 0), std, 1.0).astype(np.float32)

        self.fitted = True

    def transform(self, X):
        import numpy as np
        assert self.fitted, "Scaler not fitted"
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        if self.kind == "minmax":
            Z = (X - self.min_) / self.range_
        else:
            Z = (X - self.mean_) / self.std

        if self.clip_value is not None:
            Z = np.clip(Z, -self.clip_value, self.clip_value)

        return Z.astype(np.float32, copy=False)


# ==========================
# Windowing (with labels)
# ==========================
def make_windows_with_labels(X: np.ndarray, y: np.ndarray, seq_len: int, step: int):
    T = int(seq_len)
    if X.shape[0] < T:
        return np.empty((0, T, X.shape[1])), np.empty((0,), dtype=int), np.empty((0,), dtype=int)
    windows, idx_last, has_anom = [], [], []
    for last in range(T - 1, len(X)):
        if ((last - (T - 1)) % step) != 0:
            continue
        start = last - (T - 1)
        win_y = y[start:last + 1]
        windows.append(X[start:last + 1])
        idx_last.append(last)
        has_anom.append(int(win_y.max()))
    return np.asarray(windows), np.asarray(idx_last), np.asarray(has_anom, dtype=int)

# ==========================
# Metrics helpers
# ==========================
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

def point_metrics_from_counts(c):
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.0
    acc       = (tp + tn) / max(1, (tp + tn + fp + fn))
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc}

# ==========================
# Main
# ==========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.config):
        print(f"[FATAL] Config not found: {args.config}")
        sys.exit(2)
    cfg = load_cfg(args.config)

    prepared = get_cfg(cfg, "dataset.prepared_dir") or get_cfg(cfg, "prepared_dir")
    if prepared is None or not os.path.exists(prepared):
        print(f"[FATAL] prepared_dir not found: {prepared}")
        sys.exit(2)

    dataset_name = str(get_cfg(cfg, "dataset.name", "dataset"))
    output_root  = str(get_cfg(cfg, "output.dir", "runs"))
    seq_len      = int(get_cfg(cfg, "split.seq_len", 64))
    step         = int(get_cfg(cfg, "split.step", 1))
    batch_size   = int(get_cfg(cfg, "model.batch_size", 128))
    epochs       = int(get_cfg(cfg, "model.epochs", 10))
    loss_name    = str(get_cfg(cfg, "model.loss", "mse"))
    model_name   = str(get_cfg(cfg, "model.name", "gru_ae")).lower()

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{ts}__TF_{model_name.upper()}__seq{seq_len}_bs{batch_size}_{get_cfg(cfg,'scaler.kind','minmax')}"
    tag = args.tag or get_cfg(cfg, "output.tag", "")
    if tag:
        run_id += f"__{tag}"
    run_dir = Path(output_root) / dataset_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(run_dir / "_debug.log")

    # filenames (configurable via config.output.*)
    summary_name   = str(get_cfg(cfg, "output.overall_summary_filename", "overall_Summary.csv"))
    aggregate_name = str(get_cfg(cfg, "output.aggregate_filename", "aggregate.json"))

    gpus = tf.config.list_physical_devices("GPU")
    logger(f"[ENV] TensorFlow {tf.__version__} | GPUs={len(gpus)}")

    csvs = list_csvs(prepared)
    logger(f"[DISCOVER] {len(csvs)} CSVs under {prepared}")
    if not csvs:
        logger("[STOP] No CSVs found.")
        return

    rows = []
    # totals (micro)
    micro = {
        "err_tp": 0, "err_fp": 0, "err_fn": 0, "err_tn": 0,
        "lik_tp": 0, "lik_fp": 0, "lik_fn": 0, "lik_tn": 0,
        "md_tp":  0, "md_fp":  0, "md_fn":  0, "md_tn":  0,
        "n_test_windows": 0, "n_test_anom": 0
    }

    # -------- per-series loop --------
    for i, path in enumerate(csvs, start=1):
        series_name = os.path.relpath(path, prepared).replace(os.sep, "__").replace(".csv", "")
        logger(f"\n=== [{i}/{len(csvs)}] Series: {series_name} ===")

        try:
            df, X_raw, y, feat_cols = load_labeled_csv(path)
            if y is None:
                logger("[SKIP] no 'label' column")
                continue

            train_df, test_df, split_idx = train_test_split_sequence(df, float(get_cfg(cfg, "split.train_ratio", 0.7)))
            logger(f"[SPLIT] train={len(train_df)} test={len(test_df)} split_idx={split_idx}")

            # Use sanitized X_raw from loader (float32, NaN/Inf handled)
            X_all = X_raw.astype(np.float32, copy=False)
            y_all = y

            # Scaler on CLEAN train points
            scaler = TSScaler(kind=get_cfg(cfg, "scaler.kind", "minmax"),
                              clip_value=get_cfg(cfg, "scaler.clip_value", None))
            mask_clean_train_pts = (np.arange(len(df)) < split_idx) & (y_all == 0)
            n_clean = int(mask_clean_train_pts.sum())
            if n_clean == 0:
                raise RuntimeError("No normal points in train split. Increase train_ratio or check labels.")
            scaler.fit(X_all[mask_clean_train_pts])

            X_all_z = np.empty_like(X_all, dtype=np.float32)
            X_all_z[:split_idx] = scaler.transform(X_all[:split_idx])
            X_all_z[split_idx:] = scaler.transform(X_all[split_idx:])
            logger(f"[SCALE] fitted on clean train points: {n_clean}/{split_idx}")

            X_all_w, idx_all, win_has_anom = make_windows_with_labels(X_all_z, y_all, seq_len, step)
            if X_all_w.shape[0] == 0:
                logger(f"[SKIP] not enough data for seq_len={seq_len}")
                continue

            # Training windows: anomaly-free in train region
            train_window_mask = (idx_all < split_idx) & (win_has_anom == 0)
            X_train_w = X_all_w[train_window_mask]
            n_train_w = int(X_train_w.shape[0])
            if n_train_w == 0:
                raise RuntimeError("No anomaly-free train windows. Reduce seq_len or check labels.")
            logger(f"[WINDOWS] seq_len={seq_len} step={step} | train_windows(clean)={n_train_w} | all_windows={X_all_w.shape[0]}")

            # =======================================
            # Build + train with validation/callbacks
            # =======================================
            if model_name not in MODEL_REGISTRY:
                raise KeyError(f"Unknown model.name={model_name}. Choices: {list(MODEL_REGISTRY.keys())}")
            model = MODEL_REGISTRY[model_name](input_dim=X_train_w.shape[2], seq_len=X_train_w.shape[1], cfg=cfg)
            logger(f"[MODEL] {model_name} built")

            # Validation split from clean train windows (sequence-preserving)
            val_ratio = float(get_cfg(cfg, "model.val_ratio", 0.1))
            if val_ratio > 0.0 and X_train_w.shape[0] >= 10:
                n_val = max(1, int(round(X_train_w.shape[0] * val_ratio)))
                X_val_w = X_train_w[-n_val:]       # last chunk as validation
                X_tr_w  = X_train_w[:-n_val]
                logger(f"[VAL] Using last {n_val}/{X_train_w.shape[0]} clean train windows as validation")
            else:
                X_tr_w  = X_train_w
                X_val_w = None
                logger("[VAL] No validation split (val_ratio=0 or too few windows)")

            ds_tr  = tf.data.Dataset.from_tensor_slices((X_tr_w, X_tr_w)).batch(batch_size, drop_remainder=False)
            ds_val = tf.data.Dataset.from_tensor_slices((X_val_w, X_val_w)).batch(batch_size, drop_remainder=False) if X_val_w is not None else None

            # Callbacks: EarlyStopping, Checkpoint, CSVLogger
            callbacks = []
            es_enabled = bool(get_cfg(cfg, "model.early_stopping.enabled", True))
            if es_enabled:
                patience  = int(get_cfg(cfg, "model.early_stopping.patience", 10))
                min_delta = float(get_cfg(cfg, "model.early_stopping.min_delta", 0.0))
                monitor_cfg = str(get_cfg(cfg, "model.early_stopping.monitor", "auto")).lower()
                monitor = "val_loss" if (monitor_cfg == "auto" and ds_val is not None) else ("loss" if monitor_cfg == "auto" else monitor_cfg)
                mode = str(get_cfg(cfg, "model.early_stopping.mode", "min"))
                callbacks.append(tf.keras.callbacks.EarlyStopping(
                    monitor=monitor, patience=patience, min_delta=min_delta,
                    mode=mode, restore_best_weights=True
                ))
                logger(f"[ES] EarlyStopping enabled: monitor={monitor} patience={patience} min_delta={min_delta} mode={mode}")

            if bool(get_cfg(cfg, "model.checkpoint", True)):
                ckpt_dir = run_dir / "checkpoints"
                ckpt_path = ckpt_dir / "best.keras"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_monitor = monitor if es_enabled else ("val_loss" if ds_val is not None else "loss")
                callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(ckpt_path), monitor=ckpt_monitor, mode="min",
                    save_best_only=True, save_weights_only=False
                ))
                logger(f"[CKPT] Will save best model to {ckpt_path} (monitor={ckpt_monitor})")

            callbacks.append(tf.keras.callbacks.CSVLogger(str(run_dir / "training_log.csv"), append=False))


            # --------------------------
            # Train + score (5 models)
            # --------------------------
            X_hat_all = None
            last_loss, last_val = None, None

            if model_name == "isoforest":
                # IsolationForest: fit on flattened clean train windows, score all windows
                from src.pipelines.isoforest_helper import fit_isoforest_and_score

                errs = fit_isoforest_and_score(model, X_train_w, X_all_w)
                logger("[TRAIN] isoforest fitted + scored")

                # No reconstructions / residuals for isoforest
                residuals_all = None
                residuals_train = None

            else:
                # TF/Keras models: unchanged behavior
                hist = model.fit(
                    ds_tr,
                    epochs=epochs,
                    verbose=1,
                    shuffle=False,
                    validation_data=ds_val,
                    callbacks=callbacks,
                )
                last_loss = hist.history.get("loss", [None])[-1]
                last_val  = hist.history.get("val_loss", [None])[-1] if ds_val is not None else None
                logger(f"[TRAIN] done. Final loss={last_loss} val_loss={last_val}")

                # Reconstruct all windows
                X_hat_all = model.predict(X_all_w, verbose=0)

                # Scores per window (reconstruction error)
                errs = reconstruction_errors(X_all_w, X_hat_all, loss=loss_name)

                residuals_all  = (X_all_w - X_hat_all)[:, -1, :]
                residuals_train = residuals_all[train_window_mask]
                if residuals_train.shape[0] < 5:
                    residuals_train = residuals_all[:max(5, residuals_all.shape[0] // 5)]


            # -------- Thresholds for reconstruction error (pred_err) --------
            thr_hi, thr_lo = None, None
            if bool(get_cfg(cfg, "scoring.use_percentile", False)):
                pctl = 100 * float(get_cfg(cfg, "scoring.percentile", 0.99))
                base = errs[train_window_mask]
                if base.size == 0:
                    base = errs
                thr = float(np.percentile(base, pctl))
                thr_hi = thr
                thr_src = f"percentile({pctl:.1f})"
                pred_err = (errs >= thr).astype(int)
            else:
                mode = str(get_cfg(cfg, "scoring.mode", "fixed")).lower()
                if mode == "stddev":
                    k = float(get_cfg(cfg, "scoring.std_k", 3.0))
                    two_sided = bool(get_cfg(cfg, "scoring.two_sided", False))
                    tr_errs = errs[train_window_mask]
                    if tr_errs.size == 0:
                        tr_errs = errs[:max(5, len(errs)//5)]
                    mu = float(np.mean(tr_errs))
                    sigma = float(np.std(tr_errs, ddof=0))
                    thr_hi = mu + k * sigma
                    thr_lo = max(0.0, mu - k * sigma)
                    thr = thr_hi
                    thr_src = f"stddev(mu={mu:.6g}, sigma={sigma:.6g}, k={k}, two_sided={two_sided})"
                    if two_sided:
                        pred_err = ((errs >= thr_hi) | (errs <= thr_lo)).astype(int)
                    else:
                        pred_err = (errs >= thr_hi).astype(int)
                else:
                    thr = float(get_cfg(cfg, "scoring.threshold", 0.5))
                    thr_hi = thr
                    thr_src = "fixed"
                    pred_err = (errs >= thr).astype(int)

            # Likelihood detector
            lik = anomaly_likelihood(
                errs,
                int(get_cfg(cfg, "likelihood.long_window", 400)),
                int(get_cfg(cfg, "likelihood.short_window", 10)),
            )
            pred_lik = (lik >= float(get_cfg(cfg, "likelihood.threshold", 0.9))).astype(int)

            # Mahalanobis on residuals (last step)
            if model_name == "isoforest":
                # No residual vectors for isoforest; keep schema stable
                md = np.zeros_like(errs, dtype=np.float32)
                md_thr = 0.0
                pred_md = np.zeros_like(pred_err, dtype=int)
            else:
                md = mahalanobis_scores(residuals_train, residuals_all)
                md_thr = float(np.percentile(
                    md[train_window_mask], 100 * float(get_cfg(cfg, "mahalanobis.threshold_percentile", 0.99))
                ))
                pred_md = (md >= md_thr).astype(int)

            logger(f"[THRESH] err={thr_src}:{thr:.6f} | lik>={get_cfg(cfg,'likelihood.threshold',0.9)} | md pctl={get_cfg(cfg,'mahalanobis.threshold_percentile',0.99)}")

            # Labels aligned to windows (label at last index of each window)
            y_idx = y_all[idx_all]

            # Evaluate on TEST windows only
            test_mask = idx_all >= split_idx
            n_test_windows = int(test_mask.sum())
            n_test_anom    = int((y_idx[test_mask] == 1).sum())

            # Confusion counts + point metrics
            counts_err = confusion_counts(y_idx[test_mask], pred_err[test_mask])
            counts_lik = confusion_counts(y_idx[test_mask], pred_lik[test_mask])
            counts_md  = confusion_counts(y_idx[test_mask], pred_md[test_mask])

            m_err = point_metrics_from_counts(counts_err) if n_test_windows else {}
            m_lik = point_metrics_from_counts(counts_lik) if n_test_windows else {}
            m_md  = point_metrics_from_counts(counts_md)  if n_test_windows else {}
            logger(f"[METRICS] err={m_err} | lik={m_lik} | md={m_md}")

            # ---------- Save per-series ----------
            series_dir = run_dir / "series" / series_name
            series_dir.mkdir(parents=True, exist_ok=True)

            # options from config
            save_last_features   = bool(get_cfg(cfg, "output.save_last_features", True))
            save_scaled_features = bool(get_cfg(cfg, "output.save_scaled_features", False))
            save_window_npz      = bool(get_cfg(cfg, "output.save_window_npz", False))

            # try to carry a timestamp if present in the raw df
            ts_col = None
            for cand in ("timestamp", "ts", "time", "datetime"):
                if cand in df.columns:
                    ts_col = cand
                    break
            timestamps = df.iloc[idx_all][ts_col].values if ts_col else None

            # Base columns for scores.csv
            df_out = pd.DataFrame({
                "t_idx": idx_all,
                "split": np.where(idx_all >= split_idx, "test", "train"),
                "y_true": y_idx,
                "err": errs,
                "lik": lik,
                "md": md,
                "pred_err": pred_err,
                "pred_lik": pred_lik,
                "pred_md": pred_md,
            })

            if timestamps is not None:
                df_out.insert(1, "timestamp", timestamps)

            # --------- FIX: add many feature columns in ONE go (no fragmentation) ---------
            if save_last_features and feat_cols:
                # last-step raw features for each window
                feat_last_raw = X_all[idx_all]     # (N_windows, n_features)
                add_cols = {f"x_{name}": feat_last_raw[:, j]
                            for j, name in enumerate(feat_cols)}

                if save_scaled_features:
                    feat_last_scaled = X_all_z[idx_all]
                    add_cols.update({f"z_{name}": feat_last_scaled[:, j]
                                     for j, name in enumerate(feat_cols)})

                add_df = pd.DataFrame(add_cols, index=df_out.index)
                # single concat to keep DataFrame consolidated
                df_out = pd.concat([df_out, add_df], axis=1, copy=False)
            # ------------------------------------------------------------------------------

            df_out.to_csv(series_dir / "scores.csv", index=False)

            # Optional: store full windows & reconstructions (scaled) for deep-dive
            if save_window_npz:
                np.savez_compressed(
                    series_dir / "windows_and_preds.npz",
                    X_all_w=X_all_w, X_hat_all=X_hat_all,
                    idx_all=idx_all, y_idx=y_idx,
                    train_window_mask=train_window_mask,
                    feat_cols=np.array(feat_cols, dtype=object),
                    split_idx=np.array([split_idx]),
                )

            # Per-series summary.json
            with open(series_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump({
                    "series": series_name,
                    "split_idx": int(split_idx),
                    "n_windows": int(len(idx_all)),
                    "n_test_windows": n_test_windows,
                    "n_test_anom": n_test_anom,
                    "thresholds": {
                        "err": {"source": thr_src,
                                "hi": float(thr_hi) if thr_hi is not None else None,
                                "lo": float(thr_lo) if thr_lo is not None else None},
                        "lik": {"source": "fixed", "value": float(get_cfg(cfg, "likelihood.threshold", 0.9))},
                        "md":  {"source": "percentile",
                                "value": float(get_cfg(cfg, "mahalanobis.threshold_percentile", 0.99))}
                    },
                    "counts_err": counts_err,
                    "counts_lik": counts_lik,
                    "counts_md": counts_md,
                    "metrics_err": m_err,
                    "metrics_lik": m_lik,
                    "metrics_md": m_md,
                    "train_windows_clean": int(n_train_w),
                    "features": feat_cols,
                    "model": {"name": model_name, "framework": "tf"},
                }, f, indent=2)

            # row for run-level overall summary
            row = {
                "series": series_name,
                "n_test_windows": n_test_windows,
                "n_test_anom": n_test_anom,
            }
            for prefix, mm, cc in (
                ("err", m_err, counts_err),
                ("lik", m_lik, counts_lik),
                ("md",  m_md,  counts_md),
            ):
                row[f"{prefix}_precision"] = mm.get("precision", 0.0)
                row[f"{prefix}_recall"]    = mm.get("recall",    0.0)
                row[f"{prefix}_f1"]        = mm.get("f1",        0.0)
                row[f"{prefix}_accuracy"]  = mm.get("accuracy",  0.0)
                row[f"{prefix}_tp"] = cc["tp"]
                row[f"{prefix}_fp"] = cc["fp"]
                row[f"{prefix}_fn"] = cc["fn"]
                row[f"{prefix}_tn"] = cc["tn"]

                # accumulate micro totals
                micro[f"{prefix}_tp"] += cc["tp"]
                micro[f"{prefix}_fp"] += cc["fp"]
                micro[f"{prefix}_fn"] += cc["fn"]
                micro[f"{prefix}_tn"] += cc["tn"]

            micro["n_test_windows"] += n_test_windows
            micro["n_test_anom"]    += n_test_anom

            rows.append(row)
            logger(f"[SAVE] {series_dir/'scores.csv'} | {series_dir/'summary.json'}")

        except Exception as e:
            logger(f"[ERROR] {series_name}: {e}")
            continue

    # ---------- Aggregate ----------
    if rows:
        S = pd.DataFrame(rows)
        S.to_csv(run_dir / summary_name, index=False)

        def mean_no_none(vals):
            vals = [x for x in vals if x is not None]
            return float(np.mean(vals)) if vals else None

        agg = {
            # macro (means across series)
            "macro_err_precision": mean_no_none(S.get("err_precision", []) .tolist() if "err_precision" in S else []),
            "macro_err_recall":    mean_no_none(S.get("err_recall", [])    .tolist() if "err_recall" in S else []),
            "macro_err_f1":        mean_no_none(S.get("err_f1", [])        .tolist() if "err_f1" in S else []),
            "macro_err_accuracy":  mean_no_none(S.get("err_accuracy", [])  .tolist() if "err_accuracy" in S else []),
            "macro_lik_precision": mean_no_none(S.get("lik_precision", []) .tolist() if "lik_precision" in S else []),
            "macro_lik_recall":    mean_no_none(S.get("lik_recall", [])    .tolist() if "lik_recall" in S else []),
            "macro_lik_f1":        mean_no_none(S.get("lik_f1", [])        .tolist() if "lik_f1" in S else []),
            "macro_lik_accuracy":  mean_no_none(S.get("lik_accuracy", [])  .tolist() if "lik_accuracy" in S else []),
            "macro_md_precision":  mean_no_none(S.get("md_precision", [])  .tolist() if "md_precision" in S else []),
            "macro_md_recall":     mean_no_none(S.get("md_recall", [])     .tolist() if "md_recall" in S else []),
            "macro_md_f1":         mean_no_none(S.get("md_f1", [])         .tolist() if "md_f1" in S else []),
            "macro_md_accuracy":   mean_no_none(S.get("md_accuracy", [])   .tolist() if "md_accuracy" in S else []),
            # micro totals (sums across series)
            "micro_err_tp": micro["err_tp"], "micro_err_fp": micro["err_fp"],
            "micro_err_fn": micro["err_fn"], "micro_err_tn": micro["err_tn"],
            "micro_lik_tp": micro["lik_tp"], "micro_lik_fp": micro["lik_fp"],
            "micro_lik_fn": micro["lik_fn"], "micro_lik_tn": micro["lik_tn"],
            "micro_md_tp": micro["md_tp"],  "micro_md_fp": micro["md_fp"],
            "micro_md_fn": micro["md_fn"],  "micro_md_tn": micro["md_tn"],
            "total_test_windows": micro["n_test_windows"],
            "total_test_anom": micro["n_test_anom"],
        }

        # also compute micro metrics
        def micro_metrics(tp, fp, fn, tn):
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.0
            acc       = (tp + tn) / max(1, (tp + tn + fp + fn))
            return {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc}

        me = micro_metrics(micro["err_tp"], micro["err_fp"], micro["err_fn"], micro["err_tn"])
        ml = micro_metrics(micro["lik_tp"], micro["lik_fp"], micro["lik_fn"], micro["lik_tn"])
        mm = micro_metrics(micro["md_tp"],  micro["md_fp"],  micro["md_fn"],  micro["md_tn"])

        agg.update({
            "micro_err_precision": me["precision"], "micro_err_recall": me["recall"],
            "micro_err_f1": me["f1"], "micro_err_accuracy": me["accuracy"],
            "micro_lik_precision": ml["precision"], "micro_lik_recall": ml["recall"],
            "micro_lik_f1": ml["f1"], "micro_lik_accuracy": ml["accuracy"],
            "micro_md_precision":  mm["precision"], "micro_md_recall":  mm["recall"],
            "micro_md_f1":  mm["f1"], "micro_md_accuracy":  mm["accuracy"],
        })

        with open(run_dir / aggregate_name, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)

        logger(f"\n[AGG] wrote {run_dir/summary_name} and {run_dir/aggregate_name}")
        logger(f"[DONE] run saved at: {run_dir}")
    else:
        logger("\n[AGG] No series summarized; nothing saved.")

if __name__ == "__main__":
    main()

