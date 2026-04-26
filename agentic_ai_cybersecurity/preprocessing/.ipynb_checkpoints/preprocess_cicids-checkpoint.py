import pandas as pd
import glob

def load_cicids_data(path="data/cicids/*.csv"):
    files = glob.glob(path)
    df = pd.concat([pd.read_csv(f, encoding='latin1') for f in files])

    df.columns = df.columns.str.strip()
    df.replace([float('inf'), -float('inf')], 0, inplace=True)
    df.dropna(inplace=True)

    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    X = df.drop('Label', axis=1)
    y = df['Label']

    return X, y