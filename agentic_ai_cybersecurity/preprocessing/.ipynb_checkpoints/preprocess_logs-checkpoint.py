import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_logs(path="data/hdfs/HDFS_2k.log_structured.csv"):
    logs = pd.read_csv(path)

    le = LabelEncoder()
    logs['EventId'] = le.fit_transform(logs['EventId'])

    return logs