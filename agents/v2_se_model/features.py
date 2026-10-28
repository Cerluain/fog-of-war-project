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
