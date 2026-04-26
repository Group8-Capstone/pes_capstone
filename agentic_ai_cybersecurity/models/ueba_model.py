import os
import joblib

from sklearn.ensemble import RandomForestClassifier

class UEBAModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    #SAVE
    def save(self, path="outputs/models/ueba_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    #LOAD
    def load(self, path="outputs/models/ueba_model.pkl"):
        if os.path.exists(path):
            self.model = joblib.load(path)
            return True
        return False