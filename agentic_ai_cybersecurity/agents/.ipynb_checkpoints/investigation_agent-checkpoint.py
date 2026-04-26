class InvestigationAgent:
    def analyze(self, logs):
        results = []
        for e in logs['EventId']:
            if e > 100:
                results.append("Suspicious")
            else:
                results.append("Normal")
        return results