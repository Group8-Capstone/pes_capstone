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


class CoordinatorAgent:
    def decide(self, net, fraud, logs):
        decisions = []

        for i in range(len(net)):
            # stricter rule → reduces FP
            if net[i] == 1 and (fraud[i] == 1 or logs[i] == 1):
                decisions.append("BLOCK")
            else:
                decisions.append("ALLOW")

        return decisions