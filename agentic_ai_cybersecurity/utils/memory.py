class AgentMemory:
    def __init__(self):
        self.short_term = []
        self.long_term = []

    def store_event(self, event):
        self.short_term.append(event)

        # move important events to long-term
        if event.get("threat_score", 0) > 0.8:
            self.long_term.append(event)

    def get_recent_events(self, n=10):
        return self.short_term[-n:]

    def get_similar_events(self, threat_type):
        return [e for e in self.long_term if e.get("type") == threat_type]