import pandas as pd

class StreamGenerator:

    def __init__(self, X_net, logs):

        # Ensure DataFrame
        if not isinstance(X_net, pd.DataFrame):
            X_net = pd.DataFrame(X_net)

        if not isinstance(logs, pd.DataFrame):
            logs = pd.DataFrame(logs)

        self.X_net = X_net.reset_index(drop=True)
        self.logs = logs.reset_index(drop=True)

        self.idx_net = 0
        self.idx_log = 0

    def get_network_sample(self):
        row = self.X_net.iloc[self.idx_net]
        self.idx_net = (self.idx_net + 1) % len(self.X_net)
        return row.values

    def get_log_sample(self):
        row = self.logs.iloc[self.idx_log]
        self.idx_log = (self.idx_log + 1) % len(self.logs)
        return row.values

    def stream(self):
        while True:
            yield {
                "network": self.get_network_sample(),
                "log": self.get_log_sample()
            }