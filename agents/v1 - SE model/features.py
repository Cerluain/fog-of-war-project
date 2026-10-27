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
import numpy as np

ORDER = "PNBRQK"
N_PLANES = 22
N_MOVES = 64 * 64


def sq_index(name: str) -> int:
    return (ord(name[1]) - ord("1")) * 8 + (ord(name[0]) - ord("a"))


def orient(sq: int, color: str) -> int:
    return sq if color == "white" else sq ^ 56


def encode(obs: dict, last_seen: dict) -> np.ndarray:
    c = obs["color"]
    x = np.zeros((N_PLANES, 64), np.uint8)
    for s, t in obs["own"].items():
        x[ORDER.index(t), orient(sq_index(s), c)] = 1
    for s, t in obs["enemy"].items():
        x[6 + ORDER.index(t), orient(sq_index(s), c)] = 1
    for s in obs["visible"]:
        x[12, orient(sq_index(s), c)] = 1
    for s, t in last_seen.items():
        x[13 + ORDER.index(t), orient(sq_index(s), c)] = 1
    if obs["castling"]["kingside"]:
        x[19] = 1
    if obs["castling"]["queenside"]:
        x[20] = 1
    if obs.get("lost_on"):
        x[21, orient(sq_index(obs["lost_on"]), c)] = 1
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
