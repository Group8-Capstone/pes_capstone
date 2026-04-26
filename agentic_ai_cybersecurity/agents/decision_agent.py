class DecisionAgent:
    def decide(self, investigation_result):
        score = investigation_result["threat_score"]

        if score > 0.9:
            return "block"
        elif score > 0.7:
            return "alert"
        elif score > 0.5:
            return "monitor"
        else:
            return "ignore"