from sklearn.ensemble import RandomForestClassifier

class DetectionModel:
    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, threshold=0.5):
        probs = self.model.predict_proba(X)[:, 1]
        return (probs > threshold).astype(int)