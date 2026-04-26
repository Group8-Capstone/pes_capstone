import torch
import os
from models.transformer_log_model import TransformerLogModel

class InvestigationAgent:

    def __init__(self):
        self.model = TransformerLogModel(input_dim=50)

    # SAVE
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    # LOAD
    def load(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            return True
        return False

    def train(self, logs):
        # your training logic here
        print("Training Transformer model...")

    def analyze(self, logs):
        import torch
        X = torch.tensor(logs.values, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            probs = self.model(X).numpy().flatten()

        return (probs > 0.3).astype(int)