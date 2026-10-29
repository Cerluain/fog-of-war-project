"""Generate outcome + decoupled teacher replay buffer shards.

    python training/generate_buffer.py --iter 0 --games 2000 --workers 4 --teacher-rate 0.5 --agent-dir submission --out data/replay_buffer

Writes one .npz shard per worker named:
    iter_{iter:03d}_shard_{seed}_{wid}.npz
Schema:
    x           : uint8 [N, C, 8, 8]  encoded observation
    belief      : uint8 [N, 48]       packed ground-truth enemy-piece locations
    y_played    : int16 [N]           move index actually played by behavior policy
    y_teacher   : int16 [N]           teacher move index (-1 if unlabeled)
    has_teacher : uint8 [N]           1 if teacher label present, 0 otherwise
    z           : int8  [N]           game outcome from mover's perspective (+1 win, 0 draw, -1 loss)
    ply         : int16 [N]           1-based ply index within the game
    game_id     : int32 [N]           unique game identifier
    legal       : uint8 [N, 512]      packed legal move mask
"""
import argparse
import importlib.util
import itertools
import multiprocessing as mp
import os
import random
import sys
import time

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


def all_vs_all(players):
    return list(itertools.combinations_with_replacement(players, 2))


WARMUP_MATCHUPS = all_vs_all([
    "bot:guess-sf-8x500",
    "bot:naive-sf",
    "bot:greedy",
    "bot:random",
])

ITERATIVE_MATCHUPS = (  # Trying to be more like alphago
    [("self", "self")] * 6 +
    [("league", "self")] * 3 +
    [("self", "bot:greedy")]
)


def load_local_module(agent_dir: str, name: str):
    path = os.path.join(agent_dir, f"{name}.py")
    folder = os.path.basename(os.path.normpath(agent_dir))
    spec = importlib.util.spec_from_file_location(f"gen_{folder}_{name}_{ random.getrandbits(32) }", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def build_bot(spec: str, agent_dir: str):
    if spec == "self":
        return ExploratoryAgentWrapper(agent_dir, scale=1.0)
    if spec == "self_hot":
        return ExploratoryAgentWrapper(agent_dir, scale=1.5)
    if spec.startswith("agent:"):
        return ExploratoryAgentWrapper(spec.split("agent:", 1)[1], scale=1.0)
    return make_bot(spec)

class ExploratoryAgentWrapper:
    """CPU self-play wrapper with smooth temperature/window decay."""
    def __init__(self, agent_dir: str, scale: float = 1.0):
        agent_mod = load_local_module(agent_dir, "agent")
        self.agent = agent_mod.Agent(agent_dir, device="cpu")
        self.scale = scale
        self.ply = 0
        self.jit = 1.0
        self.rng = random.Random(0)

    def reset(self, color: str, seed: int) -> None:
        self.agent.reset(color, seed)
        self.rng = random.Random(seed)
        self.ply = 0
        self.jit = self.rng.uniform(0.85, 1.15)

    @torch.no_grad()
    def act(self, obs: dict) -> str:
        self.ply += 1
        # Always update memory first so prev / pre_prev history stays strictly in sync
        self.agent.memory.update(obs)

        for uci in obs["legal_moves"]:
            if obs["enemy"].get(uci[2:4]) == "K":
                return uci

        legal = self.agent.features.legal_indices(obs)
        keys = list(legal.keys())
        if len(keys) == 1:
            return legal[keys[0]]

        encoded = self.agent.features.encode(obs, self.agent.memory)
        x = torch.from_numpy(encoded).float().unsqueeze(0)
        out = self.agent.net(x)
        logits = (out[0] if isinstance(out, tuple) else out)[0].cpu()
        l = logits[keys]

        # Smooth decay from 1.0 (opening) toward floors (endgame), half-life = 12 moves
        decay = 0.5 ** (self.ply / 12.0)
        temp = (0.35 + 0.65 * decay) * self.scale * self.jit
        win = (0.40 + 0.60 * decay) * self.scale

        # Keep moves within `win` logits of the best move (capped at top 5)
        cand = torch.where(l >= l.max() - win)[0]
        if len(cand) > 5:
            cand = torch.topk(l, k=5).indices

        probs = F.softmax(l[cand] / max(temp, 1e-3), dim=-1).tolist()
        chosen = self.rng.choices(cand.tolist(), weights=probs, k=1)[0]
        return legal[keys[chosen]]

    def close(self):
        pass


def normalize_uci(uci: str) -> str:
    return uci[:4] + "q" if len(uci) == 5 else uci


def teacher_move(engine, game, nodes):
    for mv in game.legal_uci():
        if game.board.piece_type_at(chess.parse_square(mv[2:4])) == chess.KING:
            return mv
    try:
        mv = engine.play(game.board.copy(stack=False), chess.engine.Limit(nodes=nodes)).move.uci()
    except Exception:
        return None
    return mv if mv in game.legal_uci() else None


def extract_winner(game: Game):
    """Returns 'white', 'black', or None (draw/timeout)."""
    if hasattr(game, "winner") and game.winner in ("white", "black"):
        return game.winner
    if hasattr(game, "result"):
        res = game.result
        if res in ("1-0", "white", 1):
            return "white"
        if res in ("0-1", "black", -1):
            return "black"
    # Fallback: check which king remains on the board
    w_king = game.board.king(chess.WHITE) is not None
    b_king = game.board.king(chess.BLACK) is not None
    if w_king and not b_king:
        return "white"
    if b_king and not w_king:
        return "black"
    return None


def worker(args):
    wid, n_games, nodes, teacher_rate, keep, out, seed, iteration, agent_dir, pool_dirs, has_trained_agent = args
    # Prevent multi-worker CPU thread oversubscription
    torch.set_num_threads(1)
    t0 = time.time()
    rng = random.Random(seed + iteration * 1000 + wid)

    features = load_local_module(agent_dir, "features")
    N_MOVES = features.N_MOVES
    encode = features.encode
    legal_indices = features.legal_indices
    move_index = features.move_index
    encode_belief_target = getattr(features, "encode_belief_target", None)

    

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path()) if teacher_rate > 0 else None
    if engine is not None:
        engine.configure({"Threads": 1, "Hash": 16})

    valid_pool = [d for d in pool_dirs if os.path.exists(os.path.join(d, "weights", "model.safetensors"))]
    matchups = ITERATIVE_MATCHUPS if has_trained_agent else WARMUP_MATCHUPS
    if not valid_pool:
        matchups = [m for m in matchups if "league" not in m]

    bots = {}
    xs, beliefs, y_playeds, y_teachers, has_teachers, zs, plies, game_ids, legals = [], [], [], [], [], [], [], [], []
    decisive_games = 0
    self_play_games, self_play_decisive = 0, 0

    for g in range(n_games):
        a, b = rng.choice(matchups)
        if rng.random() < 0.5:
            a, b = b, a

        resolved_specs = []
        for spec in (a, b):
            if spec == "league":
                resolved_specs.append(f"agent:{rng.choice(valid_pool)}")
            else:
                resolved_specs.append(spec)
        a_spec, b_spec = resolved_specs

        for spec in (a_spec, b_spec):
            if spec not in bots:
                bots[spec] = build_bot(spec, agent_dir)

        if a_spec == b_spec:
            dup_key = f"{a_spec}_black"
            if dup_key not in bots:
                bots[dup_key] = build_bot(a_spec, agent_dir)
            players = {"white": bots[a_spec], "black": bots[dup_key]}
        else:
            players = {"white": bots[a_spec], "black": bots[b_spec]}

        players["white"].reset("white", rng.randrange(1 << 30))
        players["black"].reset("black", rng.randrange(1 << 30))
        mem_cls = getattr(features, "AgentMemory", Memory)
        mem = {"white": mem_cls(), "black": mem_cls()}
        prev_obs = {"white": None, "black": None}
        pre_prev_obs = {"white": None, "black": None}
        game = Game()

        game_traj = []
        ply = 0
        gid = (iteration * 1_000_000) + (wid * 100_000) + g

        while not game.over:
            obs = game.observation()
            c = obs["color"]
            ply += 1
            mem[c].update(obs)

            keep_pos = (rng.random() < keep)
            t_idx = -1
            has_t = 0

            enc_x = encode(obs, mem[c]) if keep_pos else None

            if keep_pos and engine is not None and rng.random() < teacher_rate:
                t_uci = teacher_move(engine, game, nodes)
                if t_uci is None:
                    # If Stockfish crashed on a FOW board state, respawn it
                    try:
                        engine.ping()
                    except Exception:
                        try:
                            engine.quit()
                        except Exception:
                            pass
                        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path())
                        engine.configure({"Threads": 1, "Hash": 16})
                        t_uci = teacher_move(engine, game, nodes)
                if t_uci is not None:
                    t_idx = move_index(normalize_uci(t_uci), c)
                    has_t = 1

            leg_map = legal_indices(obs) if keep_pos else None

            played_uci = players[c].act(obs)
            if keep_pos:
                p_idx = move_index(normalize_uci(played_uci), c)
                mask = np.zeros(N_MOVES, np.uint8)
                mask[list(leg_map)] = 1
                b_target = encode_belief_target(game.board, c)
                packed_belief = np.packbits(b_target.reshape(-1))  # 384 bits -> 48 uint8 bytes
                game_traj.append((c, enc_x, p_idx, t_idx, has_t, ply, np.packbits(mask), packed_belief))

            game.play(played_uci)

        winner = extract_winner(game)
        if winner is not None:
            decisive_games += 1
        if a_spec == "self" and b_spec == "self":
            self_play_games += 1
            if winner is not None:
                self_play_decisive += 1

        for c, enc_x, p_idx, t_idx, has_t, p_ply, packed_mask, packed_belief in game_traj:
            if winner is None:
                z_val = 0
            else:
                z_val = 1 if winner == c else -1
            xs.append(enc_x)
            beliefs.append(packed_belief)
            y_playeds.append(p_idx)
            y_teachers.append(t_idx)
            has_teachers.append(has_t)
            zs.append(z_val)
            plies.append(p_ply)
            game_ids.append(gid)
            legals.append(packed_mask)

        if (g + 1) % 50 == 0:
            elapsed_hr = max((time.time() - t0) / 3600.0, 1e-6)
            g_per_hr = (g + 1) / elapsed_hr
            dec_rate = decisive_games / (g + 1)
            print(
                f"[iter {iteration}] worker {wid}: {g + 1}/{n_games} games "
                f"({g_per_hr:.0f} games/hr, decisive={dec_rate:.1%}), {len(xs):,} pos",
                flush=True,
            )

    if engine is not None:
        try:
            engine.quit()
        except Exception:
            pass
    for bot in bots.values():
        if hasattr(bot, "close"):
            try:
                bot.close()
            except Exception:
                pass

    path = os.path.join(out, f"iter_{iteration:03d}_shard_{seed}_{wid}.npz")
    np.savez_compressed(
        path,
        x=np.stack(xs),
        belief=np.stack(beliefs),
        y=np.array(y_teachers, np.int16),          # backward-compatible alias
        y_teacher=np.array(y_teachers, np.int16),
        y_played=np.array(y_playeds, np.int16),
        has_teacher=np.array(has_teachers, np.uint8),
        z=np.array(zs, np.int8),
        ply=np.array(plies, np.int16),
        game_id=np.array(game_ids, np.int32),
        legal=np.stack(legals),
    )
    return len(xs), decisive_games, n_games, self_play_decisive, self_play_games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, default=0, help="Current iteration number")
    ap.add_argument("--games", type=int, default=1500, help="Games to generate in this iteration")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nodes", type=int, default=5000, help="Stockfish nodes per teacher label")
    ap.add_argument("--teacher-rate", type=float, default=0.5, help="Fraction of kept positions labeled by Stockfish")
    ap.add_argument("--keep", type=float, default=1.0, help="Fraction of positions kept for outcome training (default 1.0)")
    ap.add_argument("--agent-dir", default="submission", help="Agent directory for features and self-play")
    ap.add_argument("--pool-dirs", nargs="*", default=[], help="Additional agent directories for league opponents")
    ap.add_argument("--out", default="data/replay_buffer", help="Replay buffer directory")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    weights_file = os.path.join(args.agent_dir, "weights", "model.safetensors")
    has_trained_agent = os.path.exists(weights_file) and args.iter > 0

    mode_str = f"DAgger/League Self-Play ({args.agent_dir})" if has_trained_agent else "Warmup Reference Bots"
    print(
        f"=== Generating Iteration {args.iter}: {args.games} games [{mode_str}] "
        f"(keep={args.keep}, teacher_rate={args.teacher_rate}) ===",
        flush=True,
    )

    per = [args.games // args.workers + (1 if i < args.games % args.workers else 0) for i in range(args.workers)]
    jobs = [
        (
            i, per[i], args.nodes, args.teacher_rate, args.keep,
            args.out, args.seed, args.iter, args.agent_dir, args.pool_dirs, has_trained_agent
        )
        for i in range(args.workers)
        if per[i] > 0
    ]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        res = pool.map(worker, jobs)

    n_pos = sum(r[0] for r in res)
    dec_all = sum(r[1] for r in res)
    tot_all = sum(r[2] for r in res)
    dec_sp = sum(r[3] for r in res)
    tot_sp = sum(r[4] for r in res)

    print(
        f"=== Iteration {args.iter} complete: added {n_pos:,} positions to {args.out} | "
        f"Decisive rate: {dec_all / max(1, tot_all):.1%} ({dec_all}/{tot_all}) ===",
        flush=True,
    )
    if tot_sp > 0:
        sp_rate = dec_sp / tot_sp
        warn = " [WARNING: <50% decisive, outcome signal may be weak!]" if sp_rate < 0.5 else ""
        print(f"    Self-Play decisive rate: {sp_rate:.1%} ({dec_sp}/{tot_sp}){warn}", flush=True)


if __name__ == "__main__":
    main()