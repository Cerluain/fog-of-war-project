"""A simple memory of what an agent has seen, built only from its own observations.

Agents are free to track history however they like; this is the version the reference
bots and the baseline use. It keeps:
- last_seen: square -> piece letter for enemy pieces seen earlier and not seen elsewhere since
- captured: how many enemy pieces of each type you have captured
"""
from __future__ import annotations

START_COUNT = {"P": 8, "N": 2, "B": 2, "R": 2, "Q": 1, "K": 1}


class Memory:
    def __init__(self) -> None:
        self.last_seen: dict[str, str] = {}
        self.captured = {t: 0 for t in START_COUNT}
        self._prev: dict | None = None

    def update(self, obs: dict) -> None:
        prev = self._prev
        mv = obs.get("last_move")
        if prev is not None and mv is not None and mv in prev["legal_moves"]:
            to = mv[2:4]
            if to in prev["enemy"]:
                self.captured[prev["enemy"][to]] += 1
            elif prev["own"].get(mv[0:2]) == "P" and mv[0] != mv[2]:
                self.captured["P"] += 1          # en passant
        for sq in obs["visible"]:
            self.last_seen.pop(sq, None)
        self.last_seen.update(obs["enemy"])
        # keep no more sightings of a type than that type can still have on the board
        remaining = self.remaining()
        for t in START_COUNT:
            visible_t = sum(1 for u in obs["enemy"].values() if u == t)
            hidden = [s for s, u in self.last_seen.items() if u == t and s not in obs["enemy"]]
            for s in hidden[max(0, remaining[t] - visible_t):]:
                del self.last_seen[s]
        self._prev = obs

    def remaining(self) -> dict[str, int]:
        """Upper bound on enemy pieces of each type still on the board (ignores promotions)."""
        return {t: START_COUNT[t] - self.captured[t] for t in START_COUNT}
