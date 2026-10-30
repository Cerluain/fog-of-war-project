"""Turn an observation (plus memory) into network input, and moves into policy indices.

Board orientation: the board is flipped for Black, so the side to move always plays "up".
Input planes (22 x 8 x 8):
   0-5   your pawns, knights, bishops, rooks, queens, king
   6-11  enemy pieces you can see now (same order)
   12    squares you can see
   13-18 enemy pieces you saw earlier and have not seen move since (memory)
   19    your kingside castling right (all ones or all zeros)
   20    your queenside castling right
   21    the square where you lost a piece on the opponent's last move
Moves: index = from_square * 64 + to_square (oriented). Promotions always use the queen.
"""
# Add to features.py
import numpy as np
# Add to features.py
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
N_PLANES = 34 # Updated to have last 2 hidden moves
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
    
    # Map python-chess piece types (1=PAWN ... 6=KING) to plane indices 0..5
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None and piece.color == enemy_color:
            plane_idx = piece.piece_type - 1  # PAWN=1 -> 0, ..., KING=6 -> 5
            oriented_sq = orient(sq, color)
            target[plane_idx, oriented_sq] = 1
            
    return target.reshape(6, 8, 8)

# In features.py
def encode(obs: dict, memory) -> np.ndarray:
    c = obs["color"]
    x = np.zeros((N_PLANES, 64), np.uint8)

    # Support both passing the full Memory object OR a legacy last_seen dict
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
# 1-ply lookahead helpers
# ---------------------------------------------------------------------------
_KNIGHT_DELTAS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
_KING_DELTAS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_ROOK_RAYS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_BISHOP_RAYS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

# PRETTY SURE THIS AINT ALLOWED SO GOTTA PURGE THIS CODE...
def is_king_attacked(king_r: int, king_f: int, own_occ: np.ndarray, enemy_planes: np.ndarray) -> float:
    """Game-rule check (no evaluation): probability that an enemy piece can capture our
    King at (king_r, king_f) on the next ply. Oriented view: enemy pawns move 'down'."""
    if king_r < 0:
        return 0.0
    max_att = 0.0

    # Enemy pawns attack from rank r+1
    if king_r + 1 < 8:
        for df in (-1, 1):
            nf = king_f + df
            if 0 <= nf < 8:
                max_att = max(max_att, float(enemy_planes[0, king_r + 1, nf]))

    # Enemy knights (plane 1) and king (plane 5)
    for plane_idx, deltas in ((1, _KNIGHT_DELTAS), (5, _KING_DELTAS)):
        for dr, df in deltas:
            nr, nf = king_r + dr, king_f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                max_att = max(max_att, float(enemy_planes[plane_idx, nr, nf]))

    # Sliders: bishop/queen on diagonals, rook/queen on files/ranks
    for planes, rays in (((2, 4), _BISHOP_RAYS), ((3, 4), _ROOK_RAYS)):
        for dr, df in rays:
            nr, nf = king_r + dr, king_f + df
            while 0 <= nr < 8 and 0 <= nf < 8:
                if own_occ[nr, nf] > 0.5:  # blocked by our own piece
                    break
                max_att = max(max_att, float(enemy_planes[planes[0], nr, nf]),
                              float(enemy_planes[planes[1], nr, nf]))
                if enemy_planes[:, nr, nf].max() > 0.5:  # blocked by a (near-)certain enemy piece
                    break
                nr += dr
                nf += df

    return max_att

# This simulation should be okay if I get rid of king risk
def step_1ply_tensors(obs: dict, memory, b_probs: np.ndarray, candidate_ucis: list):
    """Play each candidate move one ply forward.

    b_probs: (6, 8, 8) calibrated enemy-piece probabilities in OUR oriented view.
    Returns:
      next_x    : (K, N_PLANES, 8, 8) strictly binary input for the opponent at ply t+1
      king_risk : (K,) probability our King can be captured on ply t+1 (game rule only)
    `memory` is unused (kept so the call signature stays the same).
    """
    c = obs["color"]
    K = len(candidate_ucis)
    next_x = np.zeros((K, N_PLANES, 8, 8), dtype=np.float32)
    king_risk = np.zeros(K, dtype=np.float32)

    # Enemy map: exact 0/1 on visible squares, network belief in the fog
    vis_2d = np.zeros(64, dtype=np.float32)
    for s in obs["visible"]:
        vis_2d[orient(sq_index(s), c)] = 1.0
    vis_2d = vis_2d.reshape(8, 8)

    enemy_cont = b_probs.astype(np.float32) * (1.0 - vis_2d[None])
    for s, t in obs["enemy"].items():
        r, f = divmod(orient(sq_index(s), c), 8)
        enemy_cont[:, r, f] = 0.0
        enemy_cont[ORDER.index(t), r, f] = 1.0

    # Binary version for the network input (matches the training distribution)
    enemy_bin = (enemy_cont > 0.5).astype(np.float32)
    kr, kf = divmod(int(np.argmax(enemy_cont[5])), 8)  # exactly one enemy king
    enemy_bin[5] = 0.0
    enemy_bin[5, kr, kf] = 1.0

    for k, uci in enumerate(candidate_ucis):
        from_name, to_name = uci[0:2], uci[2:4]
        ff = orient(sq_index(from_name), c) % 8
        tr, tf = divmod(orient(sq_index(to_name), c), 8)

        # Our pieces after the move
        own_after = dict(obs["own"])
        moved = own_after.pop(from_name, "P")
        if len(uci) == 5:
            moved = "Q"
        own_after[to_name] = moved
        if moved == "K" and abs(ff - tf) == 2:  # castling: move the rook too
            rk = from_name[1]
            if tf > ff and f"h{rk}" in own_after:
                own_after[f"f{rk}"] = own_after.pop(f"h{rk}")
            elif tf < ff and f"a{rk}" in own_after:
                own_after[f"d{rk}"] = own_after.pop(f"a{rk}")

        # Remove any enemy piece captured on the destination
        e_cont = enemy_cont.copy()
        e_bin = enemy_bin.copy()
        captured = bool(e_bin[:, tr, tf].max() > 0.5)
        e_cont[:, tr, tf] = 0.0
        e_bin[:, tr, tf] = 0.0

        own_planes = np.zeros((6, 8, 8), dtype=np.float32)
        king_r, king_f = -1, -1
        for s, t in own_after.items():
            r, f = divmod(orient(sq_index(s), c), 8)
            own_planes[ORDER.index(t), r, f] = 1.0
            if t == "K":
                king_r, king_f = r, f

        king_risk[k] = is_king_attacked(king_r, king_f, own_planes.sum(axis=0), e_cont)

        # Opponent's view (ranks flipped): their pieces = planes 0-5, ours = planes 6-11
        opp_own = np.flip(e_bin, axis=1)
        opp_enemy = np.flip(own_planes, axis=1)
        next_x[k, 0:6] = opp_own
        next_x[k, 6:12] = opp_enemy
        next_x[k, 12] = np.maximum(np.clip(opp_own.sum(axis=0), 0, 1), opp_enemy.sum(axis=0))
        next_x[k, 13:19] = opp_enemy
        next_x[k, 19:25] = opp_enemy
        next_x[k, 25:31] = opp_enemy
        next_x[k, 31] = 1.0
        next_x[k, 32] = 1.0
        if captured:
            next_x[k, 33, 7 - tr, tf] = 1.0

    return next_x, king_risk