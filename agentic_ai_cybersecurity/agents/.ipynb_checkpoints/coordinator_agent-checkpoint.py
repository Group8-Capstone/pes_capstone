class CoordinatorAgent:
    def decide(self, net, fraud, logs):
        decisions = []

        for i in range(len(net)):
            if net[i] == 1 or fraud[i] == 1:
                decisions.append("BLOCK")
            else:
                decisions.append("ALLOW")

        return decisions