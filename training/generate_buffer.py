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



WARMUP_MATCHUPS = (
    [("bot:guess-sf-40x500", "bot:greedy")] * 5 +    
    [("bot:guess-sf-40x500", "bot:naive-sf")] * 3 +  
    [("bot:guess-sf-40x500", "bot:guess-sf-8x400")] * 2  
)

ITERATIVE_MATCHUPS = (
    [("self", "self")] * 3 +             
    [("self", "self_guarded")] * 3 +
    [("self", "league")] * 2 +      
    [("self", "bot:greedy")] * 2 +  
    [("self_guarded", "bot:greedy")] *2        
)


def load_local_module(agent_dir: str, name: str):
    path = os.path.join(agent_dir, f"{name}.py")
    folder = os.path.basename(os.path.normpath(agent_dir))
    spec = importlib.util.spec_from_file_location(f"gen_{folder}_{name}_{ random.getrandbits(32) }", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PIECE_VALS = { # ONLY USE DURING TRAINING!!!
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}

def is_omniscient_blunder(board: chess.Board, uci: str) -> bool: #ONLY FOR TRAINING PLS
    """Returns True if playing `uci` on the true `board` leaves our King capturable
    next turn or blunders a high-value piece (Queen/Rook) for free or to a lesser piece.
    """
    mv = chess.Move.from_uci(uci)
    us = board.turn
    them = not us

    moved_piece = board.piece_at(mv.from_square)
    captured_piece = board.piece_at(mv.to_square)
    captured_val = PIECE_VALS.get(captured_piece.piece_type, 0) if captured_piece else 0

    b = board.copy(stack=False)
    b.push(mv)

    # 1. CRITICAL: Did we leave our King attacked on the next ply? (Instant loss in FoW!)
    king_sq = b.king(us)
    if king_sq is not None and b.is_attacked_by(them, king_sq):
        return True

    # 2. OPTIONAL TACTICAL VETO: Did we hang a Queen or Rook on the destination square?
    if moved_piece is not None and moved_piece.piece_type in (chess.QUEEN, chess.ROOK):
        my_val = PIECE_VALS[moved_piece.piece_type]
        if b.is_attacked_by(them, mv.to_square):
            # Check the cheapest enemy attacker on `to_square`
            attackers = b.attackers(them, mv.to_square)
            min_attacker_val = min(
                (PIECE_VALS.get(b.piece_type_at(sq), 100) for sq in attackers),
                default=100,
            )
            is_defended = b.is_attacked_by(us, mv.to_square)
            # If undefended and we didn't capture something of equal/greater value -> blunder
            if not is_defended and captured_val < my_val:
                return True
            # If attacked by a cheaper piece and what we captured doesn't cover the trade -> blunder
            if min_attacker_val < my_val and (captured_val + min_attacker_val) < my_val:
                return True

    return False

def build_bot(spec: str, agent_dir: str):
    if spec == "self":
        return ExploratoryAgentWrapper(agent_dir, scale=1.0, guarded=False)
    if spec == "self_guarded":
        return ExploratoryAgentWrapper(agent_dir, scale=0.0, guarded=True, topk=5)
    if spec == "self_hot":
        return ExploratoryAgentWrapper(agent_dir, scale=1.5, guarded=False)
    if spec.startswith("agent:"):
        return ExploratoryAgentWrapper(spec.split("agent:", 1)[1], scale=1.0, guarded=False)
    return make_bot(spec)


class ExploratoryAgentWrapper:
    """CPU self-play wrapper with smooth temperature/window decay and optional omniscient guardrail."""
    def __init__(self, agent_dir: str, scale: float = 1.0, guarded: bool = False, topk: int = 5):
        agent_mod = load_local_module(agent_dir, "agent")
        self.agent = agent_mod.Agent(agent_dir, device="cpu")
        self.scale = scale
        self.guarded = guarded
        self.topk = topk
        self.ply = 0
        self.jit = 1.0
        self.rng = random.Random(0)

    def reset(self, color: str, seed: int) -> None:
        self.agent.reset(color, seed)
        self.rng = random.Random(seed)
        self.ply = 0
        self.jit = self.rng.uniform(0.85, 1.15)

    @torch.no_grad()
    def act(self, obs: dict, board: chess.Board = None) -> str:
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

        # --- MODE A: OMNISCIENT TOP-K NON-BLUNDER GUARDRAIL ---
        if self.guarded and board is not None:
            k = min(self.topk, len(keys))
            top_indices = torch.topk(l, k=k).indices.tolist()

            # Walk down the top-k moves in order of policy preference;
            # play the first move that is NOT an omniscient blunder!
            for idx in top_indices:
                cand_uci = legal[keys[idx]]
                if not is_omniscient_blunder(board, cand_uci):
                    return cand_uci

            # Fallback: if all top-k moves are in danger, check if ANY legal move saves the King
            us = board.turn
            them = not us
            for idx in torch.argsort(l, descending=True).tolist():
                cand_uci = legal[keys[idx]]
                b = board.copy(stack=False)
                b.push(chess.Move.from_uci(cand_uci))
                king_sq = b.king(us)
                if king_sq is not None and not b.is_attacked_by(them, king_sq):
                    return cand_uci

            # If checkmated/unavoidable, play the #1 policy move
            return legal[keys[top_indices[0]]]

        # --- MODE B: STANDARD EXPLORATORY SAMPLING ---
        if self.scale <= 0.0:
            best_idx = l.argmax().item()
            return legal[keys[best_idx]]

        decay = 0.5 ** (self.ply / 7.0)
        temp = (0.35 + 0.65 * decay) * self.scale * self.jit
        win = (0.40 + 0.60 * decay) * self.scale

        cand = torch.where(l >= l.max() - win)[0]
        if len(cand) > self.topk:
            cand = torch.topk(l, k=self.topk).indices

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

# BASICALLY DON'T EVER TRY TO LEARN FROM WEAK BOTS. You can only learn from your end
def should_record_player(spec: str, has_trained_agent: bool) -> bool:
    """Never record bot:greedy moves! In warmup, record Stockfish bots; in RL, record our agents."""
    if spec == "bot:greedy" or spec == "bot:random":
        return False
    if has_trained_agent:
        return spec in ("self", "self_guarded", "self_hot") or spec.startswith("agent:")
    return True

def record_matchup(stats, left, right, winner):
    """Record the result from self's perspective for a matchup involving self."""
    if "self" not in (left, right):
        return

    opponent = right if left == "self" else left
    result = stats.setdefault(opponent, {"games": 0, "wins": 0, "draws": 0, "losses": 0})
    result["games"] += 1
    if winner is None:
        result["draws"] += 1
    elif left == right and left == "self":
        if winner == "white": # Misleading metric technically, but better than before. And it tells me if there's a white advantage
            result["wins"] += 1
        else:
            result["losses"] += 1
    elif left == "self" and winner == "white" or right == "self" and winner == "black":
        result["wins"] += 1
    else:
        result["losses"] += 1


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
    matchup_stats = {}

    # Separate accumulators for train shards (90% of games) and val shards (10% of games)
    tr_data = {k: [] for k in ("x", "belief", "y_played", "y_teacher", "has_teacher", "z", "ply", "game_id", "legal")}
    val_data = {k: [] for k in ("x", "belief", "y_played", "y_teacher", "has_teacher", "z", "ply", "game_id", "legal")}

    for g in range(n_games):
        is_val_game = (g % 10 == 0)  # 10% of games strictly isolated for validation
        target_buf = val_data if is_val_game else tr_data
        
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

        spec_by_color = {"white": a_spec, "black": b_spec}

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

            # ONLY keep positions played by agents we actually want to learn from!
            can_record = should_record_player(spec_by_color[c], has_trained_agent)
            keep_pos = can_record and (rng.random() < keep)
            t_idx = -1
            has_t = 0

            enc_x = encode(obs, mem[c]) if keep_pos else None

            if keep_pos and engine is not None and rng.random() < teacher_rate:
                t_uci = teacher_move(engine, game, nodes)
                ...
                if t_uci is not None:
                    t_idx = move_index(normalize_uci(t_uci), c)
                    has_t = 1

            leg_map = legal_indices(obs) if keep_pos else None

            # Pass game.board so self_guarded's omniscient blunder guardrail actually activates!
            bot = players[c]
            played_uci = bot.act(obs, game.board) if isinstance(bot, ExploratoryAgentWrapper) else bot.act(obs)

            if keep_pos:
                p_idx = move_index(normalize_uci(played_uci), c)
                mask = np.zeros(N_MOVES, np.uint8)
                mask[list(leg_map)] = 1
                b_target = encode_belief_target(game.board, c)
                packed_belief = np.packbits(b_target.reshape(-1))
                game_traj.append((c, enc_x, p_idx, t_idx, has_t, ply, np.packbits(mask), packed_belief))

            game.play(played_uci)

        winner = extract_winner(game)
        if winner is not None:
            decisive_games += 1
        record_matchup(matchup_stats, a, b, winner)
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
        z=np.array(zs, np.float32),
        ply=np.array(plies, np.int16),
        game_id=np.array(game_ids, np.int32),
        legal=np.stack(legals),
    )
    return len(xs), decisive_games, n_games, self_play_decisive, self_play_games, matchup_stats


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
    matchup_stats = {}
    for result in res:
        for opponent, stats in result[5].items():
            total = matchup_stats.setdefault(
                opponent, {"games": 0, "wins": 0, "draws": 0, "losses": 0}
            )
            for key in total:
                total[key] += stats[key]

    print(
        f"=== Iteration {args.iter} complete: added {n_pos:,} positions to {args.out} | "
        f"Decisive rate: {dec_all / max(1, tot_all):.1%} ({dec_all}/{tot_all}) ===",
        flush=True,
    )
    if tot_sp > 0:
        sp_rate = dec_sp / tot_sp
        warn = " [WARNING: <50% decisive, outcome signal may be weak!]" if sp_rate < 0.5 else ""
        print(f"    Self-Play decisive rate: {sp_rate:.1%} ({dec_sp}/{tot_sp}){warn}", flush=True)
    if matchup_stats:
        print("    Self matchup results:", flush=True)
        for opponent, stats in sorted(matchup_stats.items()):
            games = stats["games"]
            win_rate = stats["wins"] / max(1, games)
            print(
                f"      vs {opponent}: {stats['wins']}W-{stats['draws']}D-{stats['losses']}L "
                f"({win_rate:.1%} won)",
                flush=True,
            )


if __name__ == "__main__":
    main()