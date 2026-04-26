import time
import numpy as np

def stream_data(X):
    for i in range(len(X)):
        yield X[i]
        time.sleep(0.1)  # simulate real-time