"""Generate outcome + decoupled teacher replay buffer shards with 3-stage curriculum and opening book.

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

# ---------------------------------------------------------------------------
# 3-STAGE MATCHUP CURRICULUM (EARLY -> MID -> LATE)
# ---------------------------------------------------------------------------
# Stage 1 (Early): Heavy deterministic self_guarded + self_book to bootstrap high
# tactical accuracy and expose the buffer to 10 distinct opening structures.
EARLY_MATCHUPS = (
    [("self_guarded", "bot:greedy")] * 3 +
    [("self_guarded", "league")] * 2 +
    [("self_guarded", "self_mild")] +
    [("self_book", "self_mild")] +
    [("self_book", "bot:greedy")] +
    [("self_mild", "bot:greedy")]
)

# Stage 2 (Middle): Remove heavy training wheels; focus on self_mild and self
# fighting the league, bot:greedy, self_book, and self_guard_mild.
MID_MATCHUPS = (
    [("self_mild", "league")] * 3 +
    [("self_mild", "bot:greedy")] * 2 +
    [("self_mild", "self")] * 2 +
    [("self_mild", "self_book")] +
    [("self_mild", "self_guard_mild")] +
    [("self_guard_mild", "bot:greedy")]
)

# Stage 3 (Late): Stress-test weak spots with self_hot (unrecorded sparring partner)
# and self_book while keeping sharp self_mild matchups against league and greedy.
LATE_MATCHUPS = (
    [("self_mild", "league")] * 3 +
    [("self_mild", "bot:greedy")] * 2 +
    [("self_mild", "self_hot")] * 2 +
    [("self_mild", "self_book")] +
    [("self_mild", "self")] +
    [("self_mild", "self_guard_mild")]
)

# 10 sound Fog-of-War & classical opening plans per color (4 moves deep).
# `self_book` only plays a book move if it is legal AND passes `not is_fog_or_king_blunder`.
OPENING_BOOK = {
    "white": [
        ["e2e4", "g1f3", "f1c4", "b1c3"],  # 1. Italian / Giuoco Piano
        ["d2d4", "c2c4", "b1c3", "g1f3"],  # 2. Queen's Gambit
        ["e2e4", "d2d4", "c2c3", "g1f3"],  # 3. FoW Center Radar Phalanx (with Nf3 instead of Na3)
        ["d2d4", "g1f3", "c1f4", "e2e3"],  # 4. London System (solid central dark-square control)
        ["g1f3", "g2g3", "f1g2", "d2d3"],  # 5. King's Indian Attack Fianchetto
        ["c2c4", "b1c3", "g2g3", "f1g2"],  # 6. English Fianchetto
        ["e2e4", "g1f3", "d2d4", "f1c4"],  # 7. Scotch Gambit
        ["b2b3", "c1b2", "e2e3", "g1f3"],  # 8. Nimzo-Larsen Queen's Fianchetto
        ["e2e3", "d2d4", "f1d3", "g1f3"],  # 9. Horwitz / Colle System
        ["e2e4", "b1c3", "f1c4", "d2d3"],  # 10. Vienna Game
    ],
    "black": [
        ["e7e5", "b8c6", "g8f6", "f8c5"],  # 1. Two Knights / Italian Defense
        ["d7d5", "e7e6", "g8f6", "c7c6"],  # 2. Semi-Slav Fortress
        ["c7c5", "b8c6", "e7e6", "g8f6"],  # 3. Sicilian Taimanov / Paulsen
        ["e7e6", "d7d5", "g8f6", "c7c5"],  # 4. French Defense
        ["c7c6", "d7d5", "c8f5", "e7e6"],  # 5. Caro-Kann Classical
        ["g8f6", "g7g6", "f8g7", "d7d6"],  # 6. King's Indian Defense Fianchetto
        ["d7d5", "c7c6", "g8f6", "e7e6"],  # 7. Slav Defense
        ["e7e6", "b7b6", "c8b7", "g8f6"],  # 8. Owen's Queen's Fianchetto Defense
        ["c7c5", "g7g6", "f8g7", "b8c6"],  # 9. Accelerated Dragon
        ["d7d6", "g8f6", "e7e5", "b8d7"],  # 10. Philidor / Pirc Defense
    ],
}


def pick_matchup(rng, early_pool, mid_pool, late_pool, w_early, w_mid, w_late):
    """Pick the curriculum pool according to (w_early, w_mid, w_late), then sample uniformly."""
    if not mid_pool and not late_pool:
        return rng.choice(early_pool)
    r = rng.random()
    if r < w_early and early_pool:
        return rng.choice(early_pool)
    elif r < (w_early + w_mid) and mid_pool:
        return rng.choice(mid_pool)
    elif late_pool:
        return rng.choice(late_pool)
    return rng.choice(early_pool or mid_pool or late_pool)


def pick_league_opponent(rng, valid_pool):
    """Bias league opponent selection 65% toward the most recent 15 checkpoints so the model
    is continually tested against strong recent iterations, while keeping 35% across all history.
    """
    if len(valid_pool) > 15 and rng.random() < 0.65:
        return rng.choice(valid_pool[-15:])
    return rng.choice(valid_pool)


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
    if spec == "self_guarded":
        return ExploratoryAgentWrapper(agent_dir, scale=0.0, guarded=True, topk=5)
    if spec == "self_guard_mild":
        return ExploratoryAgentWrapper(agent_dir, scale=0.20, guarded=True, topk=5)
    if spec == "self_book":
        return ExploratoryAgentWrapper(agent_dir, scale=0.25, guarded=True, topk=5, use_book=True)
    if spec == "self_mild":
        return ExploratoryAgentWrapper(agent_dir, scale=0.40, guarded=False, topk=5)
    if spec == "self":
        return ExploratoryAgentWrapper(agent_dir, scale=0.80, guarded=False, topk=5)
    if spec == "self_hot":
        return ExploratoryAgentWrapper(agent_dir, scale=1.40, guarded=False, topk=6)
    if spec.startswith("agent:"):
        return ExploratoryAgentWrapper(spec.split("agent:", 1)[1], scale=0.45, guarded=False, topk=5)
    return make_bot(spec)


class ExploratoryAgentWrapper:
    """CPU self-play wrapper with optional opening book, fog-consistent blunder guardrail,
    and smooth temperature decay.
    """
    def __init__(
        self,
        agent_dir: str,
        scale: float = 1.0,
        guarded: bool = False,
        topk: int = 5,
        use_book: bool = False,
    ):
        agent_mod = load_local_module(agent_dir, "agent")
        self.agent = agent_mod.Agent(agent_dir, device="cpu")
        self.scale = scale
        self.guarded = guarded
        self.topk = topk
        self.use_book = use_book
        self.ply = 0
        self.jit = 1.0
        self.rng = random.Random(0)
        self.book_line = []

    def reset(self, color: str, seed: int) -> None:
        self.agent.reset(color, seed)
        self.rng = random.Random(seed)
        self.ply = 0
        self.jit = self.rng.uniform(0.90, 1.10)
        if self.use_book:
            lines = OPENING_BOOK.get(color, OPENING_BOOK["white"])
            self.book_line = self.rng.choice(lines)
        else:
            self.book_line = []

    @torch.no_grad()
    def act(self, obs: dict, board: chess.Board = None) -> str:
        self.ply += 1
        self.agent.memory.update(obs)

        # Always capture the enemy King immediately if visible and legal
        for uci in obs["legal_moves"]:
            if obs["enemy"].get(uci[2:4]) == "K":
                return uci

        # Optional predefined opening repertoire for the first 4 moves (if safe and legal)
        if self.use_book and self.ply <= len(self.book_line):
            book_uci = self.book_line[self.ply - 1]
            if book_uci in obs["legal_moves"]:
                if not is_fog_or_king_blunder(obs, self.agent.memory, book_uci, board):
                    return book_uci

        legal = self.agent.features.legal_indices(obs)
        keys = list(legal.keys())
        if len(keys) == 1:
            return legal[keys[0]]

        encoded = self.agent.features.encode(obs, self.agent.memory)
        x = torch.from_numpy(encoded).float().unsqueeze(0)
        out = self.agent.net(x)
        logits = (out[0] if isinstance(out, tuple) else out)[0].cpu()
        l = logits[keys]

        k = min(self.topk, len(keys))
        top_l, top_idx = torch.topk(l, k=k)

        # --- MODE A: FOG-CONSISTENT TOP-K NON-BLUNDER GUARDRAIL ---
        if self.guarded:
            safe_pos = [
                i for i, idx in enumerate(top_idx.tolist())
                if not is_fog_or_king_blunder(obs, self.agent.memory, legal[keys[idx]], board)
            ]
            if not safe_pos:
                # Fallback: check if any legal move outside top-k saves the King
                for idx in torch.argsort(l, descending=True).tolist():
                    cand_uci = legal[keys[idx]]
                    if not is_fog_or_king_blunder(obs, self.agent.memory, cand_uci, board):
                        return cand_uci
                return legal[keys[top_idx[0].item()]]

            if self.scale <= 0.0 or len(safe_pos) == 1:
                return legal[keys[top_idx[safe_pos[0]].item()]]

            # Restrict candidates strictly to safe moves, then apply scale temperature
            top_l = top_l[safe_pos]
            top_idx = top_idx[safe_pos]

        # --- MODE B: NATURAL SOFTMAX EXPLORATION (RESPECTS TRUE LOGIT GAPS) ---
        if self.scale <= 0.0:
            best_idx = l.argmax().item()
            return legal[keys[best_idx]]

        decay = 0.5 ** (self.ply / 6.0)
        temp = max(0.05, (0.05 + 0.75 * decay) * self.scale * self.jit)
        probs = F.softmax((top_l - top_l[0]) / temp, dim=-1)

        keep_mask = probs >= 0.10
        if not keep_mask.any():
            keep_mask[0] = True
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
    """Never record weak bots (bot:greedy/bot:random) or unguarded high-temp sparring partner (self_hot).
    DO record self_mild, self, self_guarded, self_guard_mild, self_book, and league!
    """
    if spec in ("bot:greedy", "bot:random", "self_hot"):
        return False
    if has_trained_agent:
        return (
            spec in ("self", "self_mild", "self_guarded", "self_guard_mild", "self_book")
            or spec.startswith("agent:")
        )
    return True


PRIMARY_ACTORS = ("self", "self_mild")


def record_matchup(stats, left, right, winner, actor="self"):
    """Record the result from `actor`'s perspective for a matchup involving that actor.
    When `actor == "self"`, both `"self"` and `"self_mild"` are grouped as the primary self agent
    so `plot_experiments.py` continues to track `vs bot:greedy`, `vs league`, `vs self`, etc. seamlessly.
    """
    actor_group = PRIMARY_ACTORS if actor == "self" else (actor,)
    if left not in actor_group and right not in actor_group:
        return

    if left in actor_group and right in actor_group:
        opponent = "self"
        is_mirror = True
        actor_is_white = True
    elif left in actor_group:
        opponent = right
        is_mirror = False
        actor_is_white = True
    else:
        opponent = left
        is_mirror = False
        actor_is_white = False

    if opponent in ("self_guarded", "self_guard_mild") and actor == "self":
        opponent = "self_guarded"

    result = stats.setdefault(opponent, {"games": 0, "wins": 0, "draws": 0, "losses": 0})
    result["games"] += 1
    if winner is None:
        result["draws"] += 1
    elif is_mirror:
        if winner == "white":
            result["wins"] += 1
        else:
            result["losses"] += 1
    elif (actor_is_white and winner == "white") or (not actor_is_white and winner == "black"):
        result["wins"] += 1
    else:
        result["losses"] += 1


def merge_matchup_stats(per_worker_stats):
    """Sum per-worker {opponent: {games, wins, draws, losses}} dicts."""
    merged = {}
    for worker_stats in per_worker_stats:
        for opponent, stats in worker_stats.items():
            total = merged.setdefault(opponent, {"games": 0, "wins": 0, "draws": 0, "losses": 0})
            for key in total:
                total[key] += stats[key]
    return merged


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
    (
        wid, n_games, nodes, teacher_rate, keep, keep_draw, out, seed,
        iteration, agent_dir, pool_dirs, has_trained_agent, w_early, w_mid, w_late
    ) = args
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
    if has_trained_agent:
        early_pool = list(EARLY_MATCHUPS)
        mid_pool = list(MID_MATCHUPS)
        late_pool = list(LATE_MATCHUPS)
    else:
        early_pool, mid_pool, late_pool = list(WARMUP_MATCHUPS), [], []

    if not valid_pool:
        early_pool = [m for m in early_pool if "league" not in m]
        mid_pool = [m for m in mid_pool if "league" not in m]
        late_pool = [m for m in late_pool if "league" not in m]

    bots = {}
    decisive_games = 0
    self_play_games, self_play_decisive = 0, 0
    matchup_stats = {}
    guarded_stats = {}

    keys = ("x", "belief", "y_played", "y_teacher", "has_teacher", "z", "ply", "game_id", "legal")
    tr_data = {k: [] for k in keys}
    val_data = {k: [] for k in keys}

    for g in range(n_games):
        is_val_game = (g % 10 == 0)
        target_buf = val_data if is_val_game else tr_data

        a, b = pick_matchup(rng, early_pool, mid_pool, late_pool, w_early, w_mid, w_late)
        if rng.random() < 0.5:
            a, b = b, a

        resolved_specs = []
        for spec in (a, b):
            if spec == "league":
                resolved_specs.append(f"agent:{pick_league_opponent(rng, valid_pool)}")
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
                # Only use guess-sf (warmup) or deterministic self_guarded (scale=0.0) as a free teacher
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
        record_matchup(matchup_stats, a, b, winner, actor="self")
        record_matchup(guarded_stats, a, b, winner, actor="self_guarded")
        if a_spec in PRIMARY_ACTORS and b_spec in PRIMARY_ACTORS:
            self_play_games += 1
            if winner is not None:
                self_play_decisive += 1

        for c, enc_x, p_idx, t_idx, has_t, p_ply, packed_mask, packed_belief in game_traj:
            if winner is None:
                if rng.random() >= keep_draw:
                    continue
                z_val = -0.5
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
    return n_tr + n_val, decisive_games, n_games, self_play_decisive, self_play_games, matchup_stats, guarded_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, default=0, help="Current iteration number")
    ap.add_argument("--games", type=int, default=1500, help="Games to generate in this iteration")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nodes", type=int, default=3000, help="Stockfish nodes per teacher label")
    ap.add_argument("--teacher-rate", type=float, default=0.3, help="Fraction of kept positions labeled by Stockfish")
    ap.add_argument("--keep", type=float, default=1.0, help="Fraction of positions kept for outcome training (default 1.0)")
    ap.add_argument("--keep-draw", type=float, default=0.20, help="Fraction of drawn positions kept in the buffer (default 0.20)")
    ap.add_argument("--agent-dir", default="submission", help="Agent directory for features and self-play")
    ap.add_argument("--pool-dirs", nargs="*", default=[], help="Additional agent directories for league opponents")
    ap.add_argument("--early-share", type=float, default=0.75, help="Share of games drawn from EARLY_MATCHUPS")
    ap.add_argument("--mid-share", type=float, default=0.25, help="Share of games drawn from MID_MATCHUPS")
    ap.add_argument("--late-share", type=float, default=0.0, help="Share of games drawn from LATE_MATCHUPS")
    ap.add_argument("--out", default="data/replay_buffer", help="Replay buffer directory")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    weights_file = os.path.join(args.agent_dir, "weights", "model.safetensors")
    has_trained_agent = os.path.exists(weights_file) and args.iter > 0

    w_e = max(0.0, args.early_share)
    w_m = max(0.0, args.mid_share)
    w_l = max(0.0, args.late_share)
    w_sum = w_e + w_m + w_l
    if w_sum > 0:
        w_e, w_m, w_l = w_e / w_sum, w_m / w_sum, w_l / w_sum
    else:
        w_e, w_m, w_l = 1.0, 0.0, 0.0

    if has_trained_agent:
        mode_str = (
            f"DAgger/League Self-Play ({args.agent_dir}) | "
            f"early:mid:late = {w_e:.0%}:{w_m:.0%}:{w_l:.0%}"
        )
    else:
        mode_str = "Warmup Reference Bots"
    print(
        f"=== Generating Iteration {args.iter}: {args.games} games [{mode_str}] "
        f"(keep={args.keep}, keep_draw={args.keep_draw}, teacher_rate={args.teacher_rate}) ===",
        flush=True,
    )

    per = [args.games // args.workers + (1 if i < args.games % args.workers else 0) for i in range(args.workers)]
    jobs = [
        (
            i, per[i], args.nodes, args.teacher_rate, args.keep, args.keep_draw,
            args.out, args.seed, args.iter, args.agent_dir, args.pool_dirs, has_trained_agent,
            w_e, w_m, w_l,
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
    matchup_stats = merge_matchup_stats(r[5] for r in res)
    guarded_stats = merge_matchup_stats(r[6] for r in res)

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
    if guarded_stats:
        print("    Self_guarded matchup results:", flush=True)
        for opponent, stats in sorted(guarded_stats.items()):
            games = stats["games"]
            win_rate = stats["wins"] / max(1, games)
            draw_rate = stats["draws"] / max(1, games)
            print(
                f"      self_guarded -> {opponent}: {stats['wins']} W / {stats['draws']} D / {stats['losses']} L "
                f"(n={games}, {win_rate:.1%} won, {draw_rate:.1%} drawn)",
                flush=True,
            )


if __name__ == "__main__":
    main()