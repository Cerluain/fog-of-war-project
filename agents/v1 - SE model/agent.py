"""Minimal agent: shows the interface every submission implements.

The referee loads this file in its own process and calls:
    Agent(agent_dir, device)        once, when your agent starts
    reset(color, seed)              at the start of every game ("white" or "black")
    act(obs) -> str                 every time it is your move; return one of obs["legal_moves"]

See docs/COMPETITION.md for the observation format and the limits.
"""
import random


class Agent:
    def __init__(self, agent_dir: str, device: str):
        self.agent_dir = agent_dir
        self.device = device

    def reset(self, color: str, seed: int) -> None:
        self.color = color
        self.rng = random.Random(seed)   # use only this seed for randomness, so games can be replayed

    def act(self, obs: dict) -> str:
        return self.rng.choice(obs["legal_moves"])
