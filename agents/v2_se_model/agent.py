"""SE-ResNet agent: loads config.json and weights/model.safetensors from its own directory,
tracks enemy piece memory, encodes the observation, and plays the highest-scoring legal move.
"""
import importlib.util
import json
import os

import torch
from safetensors.torch import load_file
from fow.memory import Memory


def _load_local_module(agent_dir: str, name: str):
    """Dynamically load features.py or model.py from this specific agent_dir."""
    path = os.path.join(agent_dir, f"{name}.py")
    folder_name = os.path.basename(os.path.normpath(agent_dir))
    spec = importlib.util.spec_from_file_location(f"agent_{folder_name}_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Agent:
    def __init__(self, agent_dir: str, device: str):
        with open(os.path.join(agent_dir, "config.json")) as f:
            cfg = json.load(f)

        self.device = torch.device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Load this agent's own features.py and model.py
        self.features = _load_local_module(agent_dir, "features")
        model_mod = _load_local_module(agent_dir, "model")

        self.net = model_mod.PolicyNet(cfg["in_planes"], cfg["channels"], cfg["blocks"])
        weights_path = os.path.join(agent_dir, "weights", "model.safetensors")
        self.net.load_state_dict(load_file(weights_path))
        self.net.to(self.device).eval()

    def reset(self, color: str, seed: int) -> None:
        mem_cls = getattr(self.features, "AgentMemory", Memory)
        self.memory = mem_cls()

    @torch.no_grad()
    def act(self, obs: dict) -> str:
        
        for uci in obs["legal_moves"]: # I assume this should be legal? Double check perhaps
            if obs["enemy"].get(uci[2:4]) == "K":
                return uci
            
        self.memory.update(obs)
        encoded = self.features.encode(obs, self.memory)
        x = torch.from_numpy(encoded).float().unsqueeze(0).to(self.device)
        out = self.net(x) 
        logits = (out[0] if isinstance(out, tuple) else out)[0].cpu()
        legal = self.features.legal_indices(obs)
        best = max(legal, key=lambda i: logits[i].item())
        return legal[best]

    