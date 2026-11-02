"""Generate outcome + decoupled teacher replay buffer shards.

    python training/generate_buffer.py --iter 0 --games 2000 --workers 4 --teacher-rate 0.3 --agent-dir submission --out data/replay_buffer
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

# OLD MATCHUPS:
# ITERATIVE_MATCHUPS = (
#     [("self", "self")] * 3 +              I'll lower bc self starts off bad
#     [("self", "self_guarded")] * 3 +      Can stay
#     [("self", "league")] * 2 +            Can stay, but adding self_guarded matchup version
#     [("self", "bot:greedy")] * 2 +        
#     [("self_guarded", "bot:greedy")] * 2
# )

# PROPOSED SPEEDY GROWTH MATCHUPS. 1:3 ratio
ITERATIVE_MATCHUPS = (
    [("self", "self")] +
    [("self", "league")] +
    [("self", "bot:greedy")] +
    [("self", "self_guarded")] * 3 +
    [("self_guarded", "league")] * 3 +
    [("self_guarded", "bot:greedy")] * 3
)


def load_local_module(agent_dir: str, name: str):
    path = os.path.join(agent_dir, f"{name}.py")
    folder = os.path.basename(os.path.normpath(agent_dir))
    spec = importlib.util.spec_from_file_location(f"gen_{folder}_{name}_{random.getrandbits(32)}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PIECE_VALS = {  # ONLY USED DURING TRAINING DATA GENERATION
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}

SYM_TO_TYPE = {
    "P": chess.PAWN,
    "N": chess.KNIGHT,
    "B": chess.BISHOP,
    "R": chess.ROOK,
    "Q": chess.QUEEN,
    "K": chess.KING,
}


def build_fog_board(obs: dict, memory) -> chess.Board:
    """Reconstructs a chess.Board containing ONLY information visible in the network's
    input planes: own pieces (planes 0-5), visible enemy pieces (planes 6-11),
    and remembered last_seen enemy pieces (planes 25-30).
    """
    b = chess.Board(None)
    us = chess.WHITE if obs["color"] == "white" else chess.BLACK
    them = not us
    b.turn = us

    for sq_name, sym in obs["own"].items():
        b.set_piece_at(chess.parse_square(sq_name), chess.Piece(SYM_TO_TYPE[sym], us))

    last_seen = memory if isinstance(memory, dict) else getattr(memory, "last_seen", {})
    for sq_name, sym in last_seen.items():
        sq = chess.parse_square(sq_name)
        if b.piece_at(sq) is None:
            b.set_piece_at(sq, chess.Piece(SYM_TO_TYPE[sym], them))

    for sq_name, sym in obs["enemy"].items():
        b.set_piece_at(chess.parse_square(sq_name), chess.Piece(SYM_TO_TYPE[sym], them))

    return b


def is_fog_or_king_blunder(obs: dict, memory, uci: str, true_board: chess.Board = None) -> bool:
    """Training-only guardrail:
    1. Checks the FOG-KNOWN board (own + visible enemy + last_seen memory) so any tactical
       veto on hanging a piece (N, B, R, Q, K) corresponds 100% to threats visible in input x.
    2. Also checks true_board strictly for 1-ply King suicide so games don't end prematurely.
    """
    mv = chess.Move.from_uci(uci)
    fog_b = build_fog_board(obs, memory)
    us = fog_b.turn
    them = not us

    moved_piece = fog_b.piece_at(mv.from_square)
    captured_piece = fog_b.piece_at(mv.to_square)
    captured_val = PIECE_VALS.get(captured_piece.piece_type, 0) if captured_piece else 0

    fog_b.push(mv)

    # 1. Did we leave our King attacked by a VISIBLE or REMEMBERED enemy piece?
    king_sq = fog_b.king(us)
    if king_sq is not None and fog_b.is_attacked_by(them, king_sq):
        return True

    # 2. Did we hang a Knight, Bishop, Rook, or Queen to a VISIBLE or REMEMBERED enemy piece?
    if moved_piece is not None and moved_piece.piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        my_val = PIECE_VALS[moved_piece.piece_type]
        if fog_b.is_attacked_by(them, mv.to_square):
            attackers = fog_b.attackers(them, mv.to_square)
            min_attacker_val = min(
                (PIECE_VALS.get(fog_b.piece_type_at(sq), 100) for sq in attackers),
                default=100,
            )
            is_defended = fog_b.is_attacked_by(us, mv.to_square)
            if not is_defended and captured_val < my_val:
                return True
            if min_attacker_val < my_val and (captured_val + min_attacker_val) < my_val:
                return True

    # 3. Prevent 1-ply King suicide on the true board so self_guarded games stay high quality
    if true_board is not None:
        tb = true_board.copy(stack=False)
        tb.push(mv)
        t_king_sq = tb.king(us)
        if t_king_sq is not None and tb.is_attacked_by(them, t_king_sq):
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
        return ExploratoryAgentWrapper(spec.split("agent:", 1)[1], scale=0.5, guarded=False)
    return make_bot(spec)


class ExploratoryAgentWrapper:
    """CPU self-play wrapper with natural logit temperature decay and fog-consistent blunder guardrail."""
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
        self.jit = self.rng.uniform(0.90, 1.10)

    @torch.no_grad()
    def act(self, obs: dict, board: chess.Board = None) -> str:
        self.ply += 1
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

        # --- MODE A: FOG-CONSISTENT TOP-K NON-BLUNDER GUARDRAIL ---
        if self.guarded:
            k = min(self.topk, len(keys))
            top_indices = torch.topk(l, k=k).indices.tolist()

            for idx in top_indices:
                cand_uci = legal[keys[idx]]
                if not is_fog_or_king_blunder(obs, self.agent.memory, cand_uci, board):
                    return cand_uci

            # Fallback: check if any legal move saves the King
            for idx in torch.argsort(l, descending=True).tolist():
                cand_uci = legal[keys[idx]]
                if not is_fog_or_king_blunder(obs, self.agent.memory, cand_uci, board):
                    return cand_uci

            return legal[keys[top_indices[0]]]

        # --- MODE B: NATURAL SOFTMAX EXPLORATION (RESPECTS TRUE LOGIT GAPS) ---
        if self.scale <= 0.0:
            best_idx = l.argmax().item()
            return legal[keys[best_idx]]

        # Smooth temperature decay: explores genuinely close moves in the opening,
        # but NEVER overrides a large tactical logit gap, and sharpens to near-argmax after ply 14.
        decay = 0.5 ** (self.ply / 6.0)
        temp = max(0.05, (0.05 + 0.75 * decay) * self.scale * self.jit)

        k = min(self.topk, len(keys))
        top_l, top_idx = torch.topk(l, k=k)
        probs = F.softmax((top_l - top_l[0]) / temp, dim=-1)

        # Only keep candidates that have at least 10% of the top move's probability
        keep_mask = probs >= 0.10
        cand_idx = top_idx[keep_mask]
        cand_probs = probs[keep_mask]

        chosen = self.rng.choices(cand_idx.tolist(), weights=cand_probs.tolist(), k=1)[0]
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
    w_king = game.board.king(chess.WHITE) is not None
    b_king = game.board.king(chess.BLACK) is not None
    if w_king and not b_king:
        return "white"
    if b_king and not w_king:
        return "black"
    return None


def should_record_player(spec: str, has_trained_agent: bool) -> bool:
    """Never record weak bots (bot:greedy/bot:random). DO record self, self_guarded, and league!"""
    if spec in ("bot:greedy", "bot:random"):
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
        if winner == "white":
            result["wins"] += 1
        else:
            result["losses"] += 1
    elif left == "self" and winner == "white" or right == "self" and winner == "black":
        result["wins"] += 1
    else:
        result["losses"] += 1


def save_shard_dict(path: str, buf: dict) -> int:
    if not buf["x"]:
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        x=np.stack(buf["x"]),
        belief=np.stack(buf["belief"]),
        y=np.array(buf["y_teacher"], np.int16),
        y_teacher=np.array(buf["y_teacher"], np.int16),
        y_played=np.array(buf["y_played"], np.int16),
        has_teacher=np.array(buf["has_teacher"], np.uint8),
        z=np.array(buf["z"], np.float32),
        ply=np.array(buf["ply"], np.int16),
        game_id=np.array(buf["game_id"], np.int32),
        legal=np.stack(buf["legal"]),
    )
    return len(buf["x"])


def worker(args):
    wid, n_games, nodes, teacher_rate, keep, out, seed, iteration, agent_dir, pool_dirs, has_trained_agent = args
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
    decisive_games = 0
    self_play_games, self_play_decisive = 0, 0
    matchup_stats = {}

    keys = ("x", "belief", "y_played", "y_teacher", "has_teacher", "z", "ply", "game_id", "legal")
    tr_data = {k: [] for k in keys}
    val_data = {k: [] for k in keys}

    for g in range(n_games):
        is_val_game = (g % 10 == 0)
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
        game = Game()

        game_traj = []
        ply = 0
        gid = (iteration * 1_000_000) + (wid * 100_000) + g

        while not game.over:
            obs = game.observation()
            c = obs["color"]
            ply += 1
            mem[c].update(obs)

            can_record = should_record_player(spec_by_color[c], has_trained_agent)
            keep_pos = can_record and (rng.random() < keep)
            t_idx = -1
            has_t = 0

            enc_x = encode(obs, mem[c]) if keep_pos else None

            if keep_pos and engine is not None and rng.random() < teacher_rate:
                t_uci = teacher_move(engine, game, nodes)
                if t_uci is not None:
                    t_idx = move_index(normalize_uci(t_uci), c)
                    has_t = 1

            leg_map = legal_indices(obs) if keep_pos else None

            bot = players[c]
            played_uci = bot.act(obs, game.board) if isinstance(bot, ExploratoryAgentWrapper) else bot.act(obs)

            if keep_pos:
                p_idx = move_index(normalize_uci(played_uci), c)
                # Use guess-sf (in warmup) or self_guarded (in self-play) as a free fog-aware teacher!
                if has_t == 0 and (spec_by_color[c].startswith("bot:guess-sf") or spec_by_color[c] == "self_guarded"):
                    t_idx = p_idx
                    has_t = 1
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
                z_val = 0.0
            else:
                z_val = 1.0 if winner == c else -1.0
            target_buf["x"].append(enc_x)
            target_buf["belief"].append(packed_belief)
            target_buf["y_played"].append(p_idx)
            target_buf["y_teacher"].append(t_idx)
            target_buf["has_teacher"].append(has_t)
            target_buf["z"].append(z_val)
            target_buf["ply"].append(p_ply)
            target_buf["game_id"].append(gid)
            target_buf["legal"].append(packed_mask)

        if (g + 1) % 50 == 0:
            elapsed_hr = max((time.time() - t0) / 3600.0, 1e-6)
            g_per_hr = (g + 1) / elapsed_hr
            dec_rate = decisive_games / (g + 1)
            n_saved = len(tr_data["x"]) + len(val_data["x"])
            print(
                f"[iter {iteration}] worker {wid}: {g + 1}/{n_games} games "
                f"({g_per_hr:.0f} games/hr, decisive={dec_rate:.1%}), {n_saved:,} pos",
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

    tr_path = os.path.join(out, f"iter_{iteration:03d}_shard_{seed}_{wid}.npz")
    val_path = os.path.join(out, "val", f"iter_{iteration:03d}_shard_{seed}_{wid}.npz")
    n_tr = save_shard_dict(tr_path, tr_data)
    n_val = save_shard_dict(val_path, val_data)
    return n_tr + n_val, decisive_games, n_games, self_play_decisive, self_play_games, matchup_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, default=0, help="Current iteration number")
    ap.add_argument("--games", type=int, default=1500, help="Games to generate in this iteration")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nodes", type=int, default=3000, help="Stockfish nodes per teacher label")
    ap.add_argument("--teacher-rate", type=float, default=0.3, help="Fraction of kept positions labeled by Stockfish")
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