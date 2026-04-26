from models.detection_model import DetectionModel

class DetectionAgent:
    def __init__(self):
        self.model = DetectionModel()

    def train(self, X, y):
        self.model.train(X, y)

    def detect(self, X):
        return self.model.predict(X)