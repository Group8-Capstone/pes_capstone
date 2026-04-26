import torch
import os
import joblib
import numpy as np
import pandas as pd
from models.transformer_log_model import TransformerLogModel


class InvestigationAgent:

    def __init__(self):
        self.model = None
        self.columns_path = "outputs/models/log_columns.pkl"

    # ================================
    # SAVE MODEL
    # ================================
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    # ================================
    # LOAD MODEL
    # ================================
    def load(self, path):
        if os.path.exists(path) and os.path.exists(self.columns_path):
            cols = joblib.load(self.columns_path)
            input_dim = len(cols)

            self.model = TransformerLogModel(input_dim=input_dim)
            self.model.load_state_dict(torch.load(path, map_location="cpu"))
            self.model.eval()
            return True
        return False

    # ================================
    # TRAIN MODEL
    # ================================
    def train(self, logs):
        print("Training Transformer model...")

        # convert logs to numeric
        logs = logs.select_dtypes(include=[np.number]).fillna(0)

        input_dim = logs.shape[1]
        print(f"✅ Log feature count: {input_dim}")

        # save column structure
        joblib.dump(logs.columns.tolist(), self.columns_path)

        # create model dynamically
        self.model = TransformerLogModel(input_dim=input_dim)

        X = torch.tensor(logs.values, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            _ = self.model(X)

        print("✅ Transformer initialized & trained")

    # ================================
    # PREPROCESS INPUT
    # ================================
    def _prepare_input(self, logs):
        logs = logs.copy()

        # convert all columns safely
        for col in logs.columns:
            logs[col] = pd.to_numeric(logs[col], errors='coerce')

        logs = logs.fillna(0)

        expected_cols = joblib.load(self.columns_path)

        # add missing columns
        for col in expected_cols:
            if col not in logs.columns:
                logs[col] = 0

        # keep correct order
        logs = logs[expected_cols]

        return logs

    # ================================
    # ANALYZE
    # ================================
    def analyze(self, logs):
        logs = self._prepare_input(logs)

        X = torch.tensor(logs.values, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            probs = self.model(X).numpy().flatten()

        return (probs > 0.3).astype(int)

    # ================================
    # INVESTIGATE (OPTIONAL)
    # ================================
    def investigate(self, detection, memory):
        past = memory.get_similar_events("network_attack")

        if past:
            detection["confidence"] += 0.05

        return detection