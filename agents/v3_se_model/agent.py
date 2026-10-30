"""SE-ResNet agent: loads config.json and weights/model.safetensors from its own directory,
tracks enemy piece memory, encodes the observation, and plays the best legal move.

Move choice: take the policy's top-K legal moves, step each one ply forward using the
network's belief head, let the network evaluate the opponent's resulting position
(value head), and combine:
    score = log pi(move) + value_weight * (-v_opp)
"""
import importlib.util
import json
import math
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

        self.features = _load_local_module(agent_dir, "features")
        model_mod = _load_local_module(agent_dir, "model")

        self.net = model_mod.PolicyNet(cfg["in_planes"], cfg["channels"], cfg["blocks"])
        weights_path = os.path.join(agent_dir, "weights", "model.safetensors")
        self.net.load_state_dict(load_file(weights_path))
        self.net.to(self.device).eval()

        # Lookahead hyperparameters (can also be overridden in config.json)
        self.top_k = cfg.get("top_k", 5)
        self.value_weight = cfg.get("value_weight", 1.0)
        self.use_continuous = cfg.get("use_continuous_belief", False)

        # Belief head was trained with pos_weight=10.0 -> raw logits are shifted by +ln(10)
        self.log_pos_weight = math.log(10.0)

    def reset(self, color: str, seed: int) -> None:
        mem_cls = getattr(self.features, "AgentMemory", Memory)
        self.memory = mem_cls()

    @torch.no_grad()
    def act(self, obs: dict) -> str:
        """Executes a 2-pass searchless neural evaluation."""
        self.memory.update(obs)

        # Game rule: capturing a visible enemy King immediately wins the game
        for uci in obs["legal_moves"]:
            if obs["enemy"].get(uci[2:4]) == "K":
                return uci

        legal = self.features.legal_indices(obs)
        legal_idx = list(legal)
        if len(legal_idx) == 1:
            return legal[legal_idx[0]]

        encoded = self.features.encode(obs, self.memory)
        x = torch.from_numpy(encoded).float().unsqueeze(0).to(self.device)

        # 1. PASS 1: Policy + Belief on the real observation
        out = self.net(x)
        if not isinstance(out, tuple) or len(out) < 3 or not hasattr(self.features, "step_1ply_tensors"):
            logits = (out[0] if isinstance(out, tuple) else out)[0].cpu()
            best = max(legal, key=lambda i: logits[i].item())
            return legal[best]

        logits, _, b_logits = out
        logits = logits[0].cpu()

        # 2. Restrict to legal moves, compute log-probabilities, and pick top-k candidates
        log_p_tensor = torch.log_softmax(logits[legal_idx], dim=0)
        log_p = dict(zip(legal_idx, log_p_tensor.tolist()))
        top = sorted(legal_idx, key=lambda i: logits[i].item(), reverse=True)[: self.top_k]
        cand_ucis = [legal[i] for i in top]
        cand_log_probs = [log_p[i] for i in top]
        k = len(cand_ucis)
        if k == 1:
            return cand_ucis[0]

        # 3. Calibrate belief probabilities (undoing pos_weight=10 prior shift) -> (6, 8, 8)
        b_probs = torch.sigmoid(b_logits[0] - self.log_pos_weight).cpu().numpy()

        # 4. Step state forward 1 ply for each of the k candidates
        next_x_np = self.features.step_1ply_tensors(
            obs, self.memory, b_probs, cand_ucis, use_continuous=self.use_continuous
        )

        # 5. PASS 2: Feed the k simulated 1-ply states back into PolicyNet
        next_x = torch.from_numpy(next_x_np).to(self.device)
        next_out = self.net(next_x)
        v_opp = next_out[1].squeeze(-1).cpu().numpy()
        our_1ply_value = -v_opp

        # 6. Combine Policy log-prob + 1-ply neural Value evaluation
        best_i, best_score = 0, -1e9
        for i in range(k):
            score = float(cand_log_probs[i]) + self.value_weight * float(our_1ply_value[i])
            if score > best_score:
                best_score, best_i = score, i

        return cand_ucis[best_i]