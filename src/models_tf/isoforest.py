# src/models_tf/isoforest.py
import numpy as np
from sklearn.ensemble import IsolationForest

class IsolationForestWrapper:
    """
    Isolation Forest operating on flattened windows.
    Produces anomaly scores compatible with existing pipeline.
    """

    def __init__(
        self,
        n_estimators=300,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X_train_flat):
        self.model.fit(X_train_flat)

    def score_samples(self, X_flat):
        # sklearn: higher = more normal → invert
        return -self.model.score_samples(X_flat)
