from __future__ import annotations

from collections import defaultdict, deque


class SessionStore:
    def __init__(self):
        self.sessions = defaultdict(lambda: deque(maxlen=6))

    def history(self, session_id): return list(self.sessions[session_id])
    def add(self, session_id, turn): self.sessions[session_id].append(turn)
