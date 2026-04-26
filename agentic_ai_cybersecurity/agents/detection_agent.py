# from models.cnn_detection_model import CNNDetectionModel

# class DetectionAgent:
#     def __init__(self, input_shape):
#         self.model = CNNDetectionModel(input_shape)

#     def train(self, X, y):
#         if hasattr(X, "values"):
#             X = X.values
#         if hasattr(y, "values"):
#             y = y.values
#         self.model.train(X, y)

#     def detect(self, X):
#         if hasattr(X, "values"):
#             X = X.values
#         return self.model.predict(X)

from xgboost import XGBClassifier
import os

class DetectionAgent:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            use_label_encoder=False
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def detect(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path="outputs/models/xgb_model.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)

    def load(self, path="outputs/models/xgb_model.json"):
        if os.path.exists(path):
            self.model.load_model(path)
            return True
        return False