"""Generate imitation / DAgger replay buffer shards using reference bots and your current agent.

    python training/generate_buffer.py --iter 0 --games 2000 --workers 4 --agent-dir submission --out data/replay_buffer

Writes one .npz shard per worker named:
    iter_{iter:03d}_shard_{seed}_{wid}.npz
"""
import argparse
import importlib.util
import multiprocessing as mp
import os
import random
import sys

import chess
import chess.engine
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fow.bots import make_bot, stockfish_path  # noqa: E402
from fow.game import Game  # noqa: E402
from fow.memory import Memory  # noqa: E402


# Warmup matchups (used on Iteration 0 before your agent has trained weights)
WARMUP_MATCHUPS = [
    ("bot:guess-sf-8x500", "bot:guess-sf-8x500"),
    ("bot:guess-sf-8x500", "bot:greedy"),
    ("bot:naive-sf", "bot:guess-sf-8x500"),
    ("bot:greedy", "bot:greedy"),
    ("bot:guess-sf-4x500", "bot:random"),
]

# Iterative DAgger / Self-Play matchups (used once agent-dir has weights/model.safetensors)
ITERATIVE_MATCHUPS = [
    ("self", "self"),                        # Self-play (with opening exploration)
    ("self", "bot:guess-sf-8x500"),          # Agent vs strong fog-aware bot
    ("self", "bot:naive-sf"),                # Agent vs naive Stockfish
    ("self", "bot:greedy"),                  # Agent vs greedy capturer
    ("bot:guess-sf-8x500", "bot:guess-sf-8x500"),
    ("bot:greedy", "bot:greedy"),            # Keeps messy tactical positions in buffer
]


def load_local_module(agent_dir: str, name: str):
    path = os.path.join(agent_dir, f"{name}.py")
    folder = os.path.basename(os.path.normpath(agent_dir))
    spec = importlib.util.spec_from_file_location(f"gen_{folder}_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ExploratoryAgentWrapper:
    """Loads your trained agent on CPU for data generation and adds slight opening
    stochasticity (first 6 plies) so self-play games explore diverse trajectories.
    """
    def __init__(self, agent_dir: str, opening_plies: int = 6, temp: float = 0.6):
        agent_mod = load_local_module(agent_dir, "agent")
        self.agent = agent_mod.Agent(agent_dir, device="cpu")
        self.features = load_local_module(agent_dir, "features")
        self.opening_plies = opening_plies
        self.temp = temp
        self.ply = 0
        self.rng = random.Random(0)

    def reset(self, color: str, seed: int) -> None:
        self.agent.reset(color, seed)
        self.rng = random.Random(seed)
        self.ply = 0

    @torch.no_grad()
    def act(self, obs: dict) -> str:
        self.ply += 1
        # After opening plies (or if temp <= 0), use the agent's standard act()
        if self.ply > self.opening_plies or self.temp <= 0:
            return self.agent.act(obs)

        # Opening exploration: sample among top-3 moves within 0.75 logits of best
        self.agent.memory.update(obs)
        legal = self.features.legal_indices(obs)
        for uci in obs["legal_moves"]:
            if obs["enemy"].get(uci[2:4]) == "K":
                return uci

        encoded = self.features.encode(obs, self.agent.memory.last_seen)
        x = torch.from_numpy(encoded).float().unsqueeze(0)
        logits = self.agent.net(x)[0].cpu()

        legal_keys = list(legal.keys())
        legal_logits = torch.tensor([logits[i].item() for i in legal_keys])
        max_l = legal_logits.max()
        mask = legal_logits >= (max_l - 0.75)
        cand_idx = torch.where(mask)[0]
        if len(cand_idx) > 3:
            _, cand_idx = torch.topk(legal_logits, k=3)

        probs = F.softmax(legal_logits[cand_idx] / self.temp, dim=-1).tolist()
        chosen = self.rng.choices(cand_idx.tolist(), weights=probs, k=1)[0]
        return legal[legal_keys[chosen]]

    def close(self):
        pass


def teacher_move(engine, game, nodes):
    for mv in game.legal_uci():
        if game.board.piece_type_at(chess.parse_square(mv[2:4])) == chess.KING:
            return mv
    try:
        mv = engine.play(game.board.copy(stack=False), chess.engine.Limit(nodes=nodes)).move.uci()
    except Exception:
        return None
    return mv if mv in game.legal_uci() else None


def worker(args):
    wid, n_games, nodes, keep, out, seed, iteration, agent_dir, has_trained_agent = args
    rng = random.Random(seed + iteration * 1000 + wid)

    features = load_local_module(agent_dir, "features")
    N_MOVES = features.N_MOVES
    encode = features.encode
    legal_indices = features.legal_indices
    move_index = features.move_index

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path())
    engine.configure({"Threads": 1, "Hash": 16})

    matchups = ITERATIVE_MATCHUPS if has_trained_agent else WARMUP_MATCHUPS
    bots = {}
    xs, ys, legals = [], [], []

    for g in range(n_games):
        a, b = rng.choice(matchups)
        if rng.random() < 0.5:
            a, b = b, a

        # Instantiate bots / agent wrappers on demand
        for spec in (a, b):
            if spec not in bots:
                if spec == "self":
                    bots[spec] = ExploratoryAgentWrapper(agent_dir)
                else:
                    bots[spec] = make_bot(spec)

        # If self-play ("self" vs "self"), we need two separate instances so their memories don't collide!
        if a == "self" and b == "self":
            if "self_black" not in bots:
                bots["self_black"] = ExploratoryAgentWrapper(agent_dir)
            players = {"white": bots["self"], "black": bots["self_black"]}
        else:
            players = {"white": bots[a], "black": bots[b]}

        players["white"].reset("white", rng.randrange(1 << 30))
        players["black"].reset("black", rng.randrange(1 << 30))
        mem = {"white": Memory(), "black": Memory()}
        game = Game()

        while not game.over:
            obs = game.observation()
            c = obs["color"]
            mem[c].update(obs)
            if rng.random() < keep:
                label = teacher_move(engine, game, nodes)
                if label is not None:
                    if len(label) == 5:
                        label = label[:4] + "q"
                    mask = np.zeros(N_MOVES, np.uint8)
                    mask[list(legal_indices(obs))] = 1
                    xs.append(encode(obs, mem[c].last_seen))
                    ys.append(move_index(label, c))
                    legals.append(np.packbits(mask))
            game.play(players[c].act(obs))

        if (g + 1) % 50 == 0:
            print(f"[iter {iteration}] worker {wid}: {g + 1}/{n_games} games, {len(ys)} positions", flush=True)

    engine.quit()
    for bot in bots.values():
        if hasattr(bot, "close"):
            bot.close()

    path = os.path.join(out, f"iter_{iteration:03d}_shard_{seed}_{wid}.npz")
    np.savez_compressed(path, x=np.stack(xs), y=np.array(ys, np.int16), legal=np.stack(legals))
    return len(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, default=0, help="Current iteration number (names the shard)")
    ap.add_argument("--games", type=int, default=1500, help="Games to generate in this iteration")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nodes", type=int, default=5000, help="Stockfish nodes per teacher label")
    ap.add_argument("--keep", type=float, default=0.5, help="Fraction of positions labelled and kept")
    ap.add_argument("--agent-dir", default="submission", help="Agent directory for features and self-play")
    ap.add_argument("--out", default="data/replay_buffer", help="Replay buffer directory")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    weights_file = os.path.join(args.agent_dir, "weights", "model.safetensors")
    has_trained_agent = os.path.exists(weights_file) and args.iter > 0

    mode_str = f"DAgger/Self-Play (using {args.agent_dir})" if has_trained_agent else "Warmup Reference Bots"
    print(f"=== Generating Iteration {args.iter}: {args.games} games [{mode_str}] ===", flush=True)

    per = [args.games // args.workers + (1 if i < args.games % args.workers else 0) for i in range(args.workers)]
    jobs = [
        (i, per[i], args.nodes, args.keep, args.out, args.seed, args.iter, args.agent_dir, has_trained_agent)
        for i in range(args.workers)
        if per[i] > 0
    ]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        n = sum(pool.map(worker, jobs))
    print(f"=== Iteration {args.iter} complete: added {n:,} positions to {args.out} ===", flush=True)


if __name__ == "__main__":
    main()