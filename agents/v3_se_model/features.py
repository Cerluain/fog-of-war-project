"""Turn an observation (plus memory) into network input, and moves into policy indices.

Board orientation: the board is flipped for Black, so the side to move always plays "up".
Input planes (34 x 8 x 8):
   0-5   your pawns, knights, bishops, rooks, queens, king
   6-11  enemy pieces you can see now (same order)
   12    squares you can see
   13-18 enemy pieces seen 1 turn ago (prev)
   19-24 enemy pieces seen 2 turns ago (pre_prev)
   25-30 enemy pieces you saw earlier and have not seen move since (last_seen memory)
   31    your kingside castling right (all ones or all zeros)
   32    your queenside castling right
   33    the square where you lost a piece on the opponent's last move
Moves: index = from_square * 64 + to_square (oriented). Promotions always use the queen.
"""
import numpy as np
from fow.memory import Memory as BaseMemory


class AgentMemory(BaseMemory):
    """Wraps official fow.memory.Memory and tracks prior observations."""
    def __init__(self):
        super().__init__()
        self.prev = {"enemy": {}}
        self.pre_prev = {"enemy": {}}
        self._curr = None

    def update(self, obs: dict) -> None:
        super().update(obs)
        if self._curr is not None:
            self.pre_prev = self.prev
            self.prev = self._curr
        # Store a shallow copy of the current enemy dict for the next turn
        self._curr = {"enemy": dict(obs["enemy"])}


ORDER = "PNBRQK"
N_PLANES = 34
N_MOVES = 64 * 64


def sq_index(name: str) -> int:
    return (ord(name[1]) - ord("1")) * 8 + (ord(name[0]) - ord("a"))


def orient(sq: int, color: str) -> int:
    return sq if color == "white" else sq ^ 56


def encode_belief_target(board, color: str) -> np.ndarray:
    """Extracts true enemy piece locations as a (6, 8, 8) uint8 binary array
    oriented from the active player's perspective.
    """
    import chess
    enemy_color = chess.BLACK if color == "white" else chess.WHITE
    target = np.zeros((6, 64), dtype=np.uint8)

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None and piece.color == enemy_color:
            plane_idx = piece.piece_type - 1  # PAWN=1 -> 0, ..., KING=6 -> 5
            oriented_sq = orient(sq, color)
            target[plane_idx, oriented_sq] = 1

    return target.reshape(6, 8, 8)


def encode(obs: dict, memory) -> np.ndarray:
    c = obs["color"]
    x = np.zeros((N_PLANES, 64), np.uint8)

    if isinstance(memory, dict):
        last_seen = memory
        prev_enemy = {}
        pre_prev_enemy = {}
    else:
        last_seen = getattr(memory, "last_seen", {})
        prev_obs = getattr(memory, "prev", None) or {}
        pre_prev_obs = getattr(memory, "pre_prev", None) or {}
        prev_enemy = prev_obs.get("enemy", {})
        pre_prev_enemy = pre_prev_obs.get("enemy", {})

    for s, t in obs["own"].items():
        x[ORDER.index(t), orient(sq_index(s), c)] = 1
    for s, t in obs["enemy"].items():
        x[6 + ORDER.index(t), orient(sq_index(s), c)] = 1
    for s in obs["visible"]:
        x[12, orient(sq_index(s), c)] = 1
    for s, t in prev_enemy.items():
        x[13 + ORDER.index(t), orient(sq_index(s), c)] = 1
    for s, t in pre_prev_enemy.items():
        x[19 + ORDER.index(t), orient(sq_index(s), c)] = 1
    for s, t in last_seen.items():
        x[25 + ORDER.index(t), orient(sq_index(s), c)] = 1
    if obs["castling"]["kingside"]:
        x[31] = 1
    if obs["castling"]["queenside"]:
        x[32] = 1
    if obs.get("lost_on"):
        x[33, orient(sq_index(obs["lost_on"]), c)] = 1
    return x.reshape(N_PLANES, 8, 8)


def move_index(uci: str, color: str) -> int:
    return orient(sq_index(uci[0:2]), color) * 64 + orient(sq_index(uci[2:4]), color)


def legal_indices(obs: dict) -> dict:
    """policy index -> uci move, for the legal moves (under-promotions are left out)."""
    out = {}
    for mv in obs["legal_moves"]:
        if len(mv) == 5 and mv[4] != "q":
            continue
        out[move_index(mv, obs["color"])] = mv
    return out


# ---------------------------------------------------------------------------
# 1-ply neural lookahead tensor builder (100% policy-compliant, no heuristics)
# ---------------------------------------------------------------------------
def step_1ply_tensors(
    obs: dict,
    memory,
    b_probs: np.ndarray,
    candidate_ucis: list,
    use_continuous: bool = False,
) -> np.ndarray:
    """Step each candidate move one ply forward to construct the opponent's
    input tensor at ply t+1.

    b_probs        : (6, 8, 8) calibrated enemy-piece probabilities in OUR oriented view.
    use_continuous : If False, thresholds fog beliefs at 0.5 for binary {0, 1} planes.
                     If True, feeds continuous calibrated probabilities in the fog.
    Returns:
      next_x       : (K, N_PLANES, 8, 8) float32 input batch for the opponent at ply t+1
    """
    c = obs["color"]
    K = len(candidate_ucis)
    next_x = np.zeros((K, N_PLANES, 8, 8), dtype=np.float32)

    # 1. Enemy map: exact 0/1 on visible squares, neural network belief in the fog
    vis_2d = np.zeros(64, dtype=np.float32)
    for s in obs["visible"]:
        vis_2d[orient(sq_index(s), c)] = 1.0
    vis_2d = vis_2d.reshape(8, 8)

    enemy_cont = b_probs.astype(np.float32) * (1.0 - vis_2d[None])
    for s, t in obs["enemy"].items():
        r, f = divmod(orient(sq_index(s), c), 8)
        enemy_cont[:, r, f] = 0.0
        enemy_cont[ORDER.index(t), r, f] = 1.0

    if use_continuous:
        enemy_map = enemy_cont
    else:
        # Binary version for the network input (matches the training distribution)
        enemy_map = (enemy_cont > 0.5).astype(np.float32)
        kr, kf = divmod(int(np.argmax(enemy_cont[5])), 8)  # Guarantee one enemy king
        enemy_map[5] = 0.0
        enemy_map[5, kr, kf] = 1.0

    # Prior own piece positions (before our candidate move) for opponent's history planes
    own_before = np.zeros((6, 8, 8), dtype=np.float32)
    for s, t in obs["own"].items():
        r, f = divmod(orient(sq_index(s), c), 8)
        own_before[ORDER.index(t), r, f] = 1.0
    opp_enemy_before = np.flip(own_before, axis=1)

    for k, uci in enumerate(candidate_ucis):
        from_name, to_name = uci[0:2], uci[2:4]
        ff = orient(sq_index(from_name), c) % 8
        tr, tf = divmod(orient(sq_index(to_name), c), 8)

        # Our pieces after the candidate move
        own_after = dict(obs["own"])
        moved = own_after.pop(from_name, "P")
        if len(uci) == 5:
            moved = "Q"
        own_after[to_name] = moved
        if moved == "K" and abs(ff - tf) == 2:  # Castling: move the rook too
            rk = from_name[1]
            if tf > ff and f"h{rk}" in own_after:
                own_after[f"f{rk}"] = own_after.pop(f"h{rk}")
            elif tf < ff and f"a{rk}" in own_after:
                own_after[f"d{rk}"] = own_after.pop(f"a{rk}")

        # Remove any enemy piece captured on the destination square
        e_map = enemy_map.copy()
        captured = bool(enemy_cont[:, tr, tf].max() > 0.5)
        e_map[:, tr, tf] = 0.0

        own_planes = np.zeros((6, 8, 8), dtype=np.float32)
        for s, t in own_after.items():
            r, f = divmod(orient(sq_index(s), c), 8)
            own_planes[ORDER.index(t), r, f] = 1.0

        # Opponent's view (ranks flipped): their pieces = planes 0-5, ours = planes 6-11
        opp_own = np.flip(e_map, axis=1)
        opp_enemy = np.flip(own_planes, axis=1)

        next_x[k, 0:6] = opp_own
        next_x[k, 6:12] = opp_enemy
        next_x[k, 12] = np.clip(opp_own.sum(axis=0) + opp_enemy.sum(axis=0), 0.0, 1.0)
        next_x[k, 13:19] = opp_enemy_before  # Turn t-1 positions
        next_x[k, 19:25] = opp_enemy_before  # Turn t-2 proxy
        next_x[k, 25:31] = opp_enemy         # Memory
        next_x[k, 31] = 1.0
        next_x[k, 32] = 1.0
        if captured:
            next_x[k, 33, 7 - tr, tf] = 1.0

    return next_x