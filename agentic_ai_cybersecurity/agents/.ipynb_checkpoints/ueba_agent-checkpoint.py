from models.ueba_model import UEBAModel

class UEBAAgent:
    def __init__(self):
        self.model = UEBAModel()

    def train(self, X, y):
        self.model.train(X, y)

    def detect(self, X):
        return self.model.predict(X)