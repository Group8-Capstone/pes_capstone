class FeedbackAgent:
    def __init__(self, memory):
        self.memory = memory

    def learn(self, inputs, decisions):
        for data, decision in zip(inputs, decisions):
            score = decision["score"]

            # evaluate correctness (simple logic)
            if decision["decision"] == "BLOCK" and score < 0.7:
                feedback = "OVERREACTED"
            elif decision["decision"] == "ALLOW" and score > 0.7:
                feedback = "MISSED_THREAT"
            else:
                feedback = "CORRECT"

            # STORE IN MEMORY
            self.memory.store_event({
                "decision": decision["decision"],
                "score": score,
                "feedback": feedback,
                "signals": data
            })