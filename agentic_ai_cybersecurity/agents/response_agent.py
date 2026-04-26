class ResponseAgent:
    def execute(self, decisions):
        for i, d in enumerate(decisions):
            if d == "BLOCK":
                print(f"Sample {i+1}: Threat blocked")
            else:
                print(f"Sample {i+1}: Safe")