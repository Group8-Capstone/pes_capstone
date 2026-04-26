
import os
import joblib
import numpy as np

from tensorflow.keras.models import Sequential, load_model   
from tensorflow.keras.layers import LSTM, Dense

class LSTMLogModel:
    def __init__(self):
        self.model = Sequential([
            LSTM(64, input_shape=(10, 1)),
            Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X, y):
        self.model.fit(X, y, epochs=3, batch_size=32)

    def predict(self, X):
        preds = self.model.predict(X)
        return (preds > 0.5).astype(int).flatten()

    # SAVE
    def save(self, path="outputs/models/lstm_model.keras"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    # LOAD
    def load(self, path="outputs/models/lstm_model.keras"):
        if os.path.exists(path):
            self.model = load_model(path)
            return True
        return False

