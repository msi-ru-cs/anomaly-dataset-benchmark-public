from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

class TSScaler:
    """
    Simple wrapper for MinMax/Standard scaler.
    Fit on TRAIN only; apply to TEST and ALL.
    """
    def __init__(self, kind: str = "minmax", clip_value=None):
        self.kind = kind
        self.clip_value = clip_value
        self.scaler = MinMaxScaler() if kind == "minmax" else StandardScaler()

    def fit(self, x):
        self.scaler.fit(x)

    def transform(self, x):
        z = self.scaler.transform(x)
        if self.clip_value is not None:
            v = float(self.clip_value)
            z = np.clip(z, -v, v)
        return z

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, z):
        return self.scaler.inverse_transform(z)
