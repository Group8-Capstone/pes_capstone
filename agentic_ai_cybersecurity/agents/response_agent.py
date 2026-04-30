class ResponseAgent:
    def execute(self, decisions):
        for d in decisions:
            if d["decision"] == "BLOCK":
                print(f"BLOCK (score={d['score']})")
            elif d["decision"] == "ALERT":
                print(f"ALERT (score={d['score']})")
            elif d["decision"] == "MONITOR":
                print(f"MONITOR (score={d['score']})")
            else:
                print(f"ALLOW (score={d['score']})")