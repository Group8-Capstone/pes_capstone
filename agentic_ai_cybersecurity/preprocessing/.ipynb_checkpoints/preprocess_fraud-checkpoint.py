import pandas as pd

def load_fraud_data(path="data/credit_card/creditcard.csv"):
    df = pd.read_csv(path)

    X = df.drop('Class', axis=1)
    y = df['Class']

    return X, y