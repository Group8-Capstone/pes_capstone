class ResponseAgent:
    def execute(self, decisions):
        for d in decisions:
            if d == "BLOCK":
                print("Threat blocked")
            else:
                print("Safe")