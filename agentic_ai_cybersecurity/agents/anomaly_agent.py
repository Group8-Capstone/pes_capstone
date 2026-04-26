import numpy as np
import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import NotFittedError

class AnomalyAgent:

    def __init__(self):
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.02,
            random_state=42,
            n_jobs=-1
        )
        self.is_fitted = False

    # ================================
    # TRAIN
    # ================================
    def train(self, X):
        print("Training Isolation Forest (safe mode)...")

        # convert to numpy if DataFrame
        if hasattr(X, "values"):
            X = X.values

        subset_size = min(200000, len(X))
        idx = np.random.choice(len(X), subset_size, replace=False)
        X_subset = X[idx]

        self.model.fit(X_subset)
        self.is_fitted = True

        print("Isolation Forest training completed")

    # ================================
    # SAVE
    # ================================
    def save(self, path):
        joblib.dump(self.model, path)
        print(f"Anomaly model saved at {path}")

    # ================================
    # LOAD
    # ================================
    def load(self, path):
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.is_fitted = True
            print("Anomaly model loaded")
            return True
        return False

    # ================================
    # DETECT
    # ================================
    def detect(self, X):

        if not self.is_fitted:
            raise NotFittedError("Anomaly model is not trained or loaded.")

        # convert DataFrame → numpy
        if hasattr(X, "values"):
            X = X.values

        preds = self.model.predict(X)   # -1 = anomaly, 1 = normal

        return np.where(preds == -1, 1, 0)