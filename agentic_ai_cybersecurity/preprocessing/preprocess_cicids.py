import pandas as pd
import glob

# ======================================
# LOAD DATA ONLY (NO TRANSFORM HERE)
# ======================================

def load_cicids_data(path="data/cicids/*.csv"):
    files = glob.glob(path)
    df = pd.concat([pd.read_csv(f, encoding='latin1') for f in files])

    df.columns = df.columns.str.strip()

    drop_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    df.replace([float('inf'), -float('inf')], 0, inplace=True)
    df.dropna(inplace=True)

    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    X = df.drop('Label', axis=1)
    y = df['Label']

    return X.values, y.values


# ======================================
# NO BALANCING (let XGBoost handle it)
# ======================================
def balance_data(X, y):
    print("Before Balancing:")
    print(pd.Series(y).value_counts())
    return X, y