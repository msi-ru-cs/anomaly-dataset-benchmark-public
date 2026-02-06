import numpy as np
from scipy.stats import norm
from numpy.linalg import inv

def reconstruction_errors(x, xhat, loss: str = "mse"):
    """
    x, xhat: [N, T, D]
    Return per-window scalar error using LAST timestep and mean over features.
    """
    if loss == "mae":
        e = np.abs(x - xhat)
    else:
        e = (x - xhat) ** 2
    last = e[:, -1, :]                 # [N, D]
    return last.mean(axis=1)           # [N]

def anomaly_likelihood(errors, long_window: int = 400, short_window: int = 10):
    """
    ICSE-style likelihood in [0.5, 1.0] using short vs long error windows.
    """
    L = []
    eps = 1e-10
    for i in range(len(errors)):
        if i < long_window:
            L.append(0.5)
            continue
        wide = errors[i - long_window:i]
        narrow = errors[max(0, i - short_window):i]  # handle beginning
        mu_w = wide.mean()
        sigma_w = wide.std() + eps
        mu_n = narrow.mean() if len(narrow) else errors[i]
        z = (mu_n - mu_w) / sigma_w
        L.append(0.5 + 0.5 * norm.cdf(z))
    return np.array(L)

def mahalanobis_scores(residuals_train: np.ndarray, residuals_all: np.ndarray):
    """
    MD over LAST-step residual vectors. Regularizes covariance.
    """
    mu = residuals_train.mean(axis=0, keepdims=True)       # [1, D]
    cov = np.cov(residuals_train.T)
    if residuals_train.shape[1] == 1:
        invcov = 1.0 / (cov + 1e-8)
    else:
        cov = cov + 1e-6 * np.eye(cov.shape[0])
        invcov = inv(cov)
    dif = residuals_all - mu
    if residuals_all.shape[1] == 1:
        md2 = (dif[:, 0] ** 2) * invcov
    else:
        md2 = np.einsum("ni,ij,nj->n", dif, invcov, dif)
    return np.sqrt(md2)
