# class CoordinatorAgent:
#     def decide(self, net, fraud, logs):
#         decisions = []

#         min_len = min(len(net), len(fraud), len(logs))

#         for i in range(min_len):
#             score = 0.5*net[i] + 0.3*fraud[i] + 0.2*logs[i]

#             if score >= 0.6:
#                 decisions.append("BLOCK")
#             else:
#                 decisions.append("ALLOW")

#         return decisions


# class CoordinatorAgent:
#     def decide(self, net, fraud, logs):
#         decisions = []

#         for i in range(len(net)):
#             # stricter rule → reduces FP
#             if net[i] == 1 and (fraud[i] == 1 or logs[i] == 1):
#                 decisions.append("BLOCK")
#             else:
#                 decisions.append("ALLOW")

#         return decisions

from utils.memory import AgentMemory

class CoordinatorAgent:
    def __init__(self, memory):
        self.memory = memory  # injected dependency

    def decide(self, inputs):
        decisions = []

        for data in inputs:
            score = (
                0.4 * data["net_conf"] +
                0.2 * data["fraud"] +
                0.2 * data["logs"] +
                0.2 * data["anomaly"]
            )

            # MEMORY READ ONLY
            recent_events = self.memory.get_recent_events(5)

            if any(e.get("decision") == "BLOCK" for e in recent_events):
                score += 0.05

            # decision logic
            if score > 0.85:
                decision = "BLOCK"
            elif score > 0.65:
                decision = "ALERT"
            elif score > 0.5:
                decision = "MONITOR"
            else:
                decision = "ALLOW"

            decisions.append({
                "decision": decision,
                "score": round(score, 3),
                "reason": f"net={data['net']}, fraud={data['fraud']}, logs={data['logs']}"
            })

        return decisions