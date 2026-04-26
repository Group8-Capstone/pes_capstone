import numpy as np
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class CNNDetectionModel:
    def __init__(self, input_shape):
        self.model = Sequential([
            Conv1D(32, 2, activation='relu', input_shape=input_shape),
            MaxPooling1D(2),
            Conv1D(64, 2, activation='relu'),
            Flatten(),
            Dense(50, activation='relu'),
            Dropout(0.3),  #reduce overfitting
            Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        self.threshold = 0.5  # default

    def train(self, X, y):
        X = np.expand_dims(X, axis=2)

        early_stop = EarlyStopping(patience=2, restore_best_weights=True)

        self.history = self.model.fit(
            X, y,
            epochs=10,
            batch_size=64,
            validation_split=0.2,
            class_weight={0:1, 1:1.5},  # balanced
            callbacks=[early_stop]
        )

    #probability output
    def predict_proba(self, X):
        X = np.expand_dims(X, axis=2)
        return self.model.predict(X).flatten()

    #use dynamic threshold
    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs > self.threshold).astype(int)

    #SAVE
    def save(self, path="outputs/models/cnn_model.keras"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    #LOAD
    def load(self, path="outputs/models/cnn_model.keras"):
        if os.path.exists(path):
            self.model = load_model(path)
            return True
        return False