"""Interactive Fog-of-War Chess Visualizer, Human-vs-Bot Player, Shard Inspector & Model Brain Overlay.

Usage Examples:
  1. Play against your model (press 'b' to toggle Model Brain + Heatmaps, 'v' to toggle 3 boards):
     python tools/visualizer.py play agents/v3_se_model --color white --brain

  2. Watch two models/bots play step-by-step with live Brain + Heatmaps:
     python tools/visualizer.py watch agents/v3_se_model bot:greedy

  3. Replay a recorded game from `fow.arena --out games.jsonl` and run your model's brain on every ply:
     python tools/visualizer.py replay games.jsonl --index 0 --model agents/v3_se_model

  4. Inspect training positions & belief targets inside a `.npz` shard (with optional live model comparison):
     python tools/visualizer.py shard data/v3_megamodel/iter_004_w0.npz --model agents/v3_se_model
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import numpy as np
import torch
from safetensors.torch import load_file

# Ensure project root is on sys.path so `fow` can be imported
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import chess
import chess.svg
from fow.game import Game, legal_moves
from fow.memory import Memory
from fow.referee import make_player

# ANSI escape codes for terminal rendering
RESET = "\033[0m"
BOLD = "\033[1m"
FG_WHITE_PIECE = "\033[97m"          # Bright white
FG_BLACK_PIECE = "\033[96m"          # Bright cyan
FG_GHOST_PIECE = "\033[93m"          # Yellow for last_seen memory ghosts
FG_TRUE_MARKER = "\033[92m"          # Bright green for ground-truth piece marker on heatmaps
BG_VISIBLE_LIGHT = "\033[48;5;240m"  # Illuminated light square
BG_VISIBLE_DARK = "\033[48;5;237m"   # Illuminated dark square
BG_FOG = "\033[48;5;232m"            # Dark gray fog
BG_LOST = "\033[48;5;52m"            # Dark red highlight for `lost_on`

ORDER = "PNBRQK"
PIECE_NAMES = {"P": "Pawn", "N": "Knight", "B": "Bishop", "R": "Rook", "Q": "Queen", "K": "King", "ALL": "Any Enemy"}
UNICODE_PIECES = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}

# ASCII density characters from 0% (dark/empty) to 100% (solid white)
SHADE_CHARS = [" · ", "░░░", "▒▒▒", "▓▓▓", "███"]


# ---------------------------------------------------------------------------
# Direct Neural Brain Inspector (Loads any agent dir for white-box diagnostics)
# ---------------------------------------------------------------------------
class BrainInspector:
    """Loads an agent's PolicyNet + features module directly to extract Value, Policy,
    1-Ply Lookahead scores, and 6x8x8 Belief Heatmaps.
    """
    def __init__(self, agent_dir: str, device: str = "cpu", calibrate_pos_weight: float = 10.0):
        self.agent_dir = os.path.normpath(agent_dir)
        self.name = os.path.basename(self.agent_dir)
        with open(os.path.join(self.agent_dir, "config.json"), "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.device = torch.device(device)
        self.features = self._load_mod("features")
        model_mod = self._load_mod("model")

        self.net = model_mod.PolicyNet(
            self.cfg.get("in_planes", 34),
            self.cfg.get("channels", 64),
            self.cfg.get("blocks", 4),
        )
        w_path = os.path.join(self.agent_dir, "weights", "model.safetensors")
        self.net.load_state_dict(load_file(w_path))
        self.net.to(self.device).eval()

        self.top_k = self.cfg.get("top_k", 5)
        self.value_weight = self.cfg.get("value_weight", 1.0)
        self.use_continuous = self.cfg.get("use_continuous_belief", False)
        self.log_pos_weight = math.log(calibrate_pos_weight) if calibrate_pos_weight > 1.0 else 0.0

    def _load_mod(self, mod_name: str):
        path = os.path.join(self.agent_dir, f"{mod_name}.py")
        spec = importlib.util.spec_from_file_location(f"viz_{self.name}_{mod_name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def new_memory(self):
        mem_cls = getattr(self.features, "AgentMemory", Memory)
        return mem_cls()

    @torch.no_grad()
    def analyze(self, obs: dict, memory, fuse_visible: bool = True, raw_sigmoid: bool = False) -> dict:
        """Run Pass 1 + Pass 2 (1-ply lookahead) and return full diagnostic dictionary."""
        encoded = self.features.encode(obs, memory)
        x = torch.from_numpy(encoded).float().unsqueeze(0).to(self.device)
        out = self.net(x)

        logits = (out[0] if isinstance(out, tuple) else out)[0].cpu()
        v_curr = float(out[1][0].item()) if (isinstance(out, tuple) and len(out) > 1 and out[1] is not None) else 0.0
        has_belief = isinstance(out, tuple) and len(out) > 2 and out[2] is not None

        legal = self.features.legal_indices(obs)
        legal_idx = list(legal.keys())
        probs_tensor = torch.softmax(logits[legal_idx], dim=0)
        log_p_tensor = torch.log_softmax(logits[legal_idx], dim=0)
        prob_map = dict(zip(legal_idx, probs_tensor.tolist()))
        log_p_map = dict(zip(legal_idx, log_p_tensor.tolist()))

        top_indices = sorted(legal_idx, key=lambda i: logits[i].item(), reverse=True)[: self.top_k]
        cand_ucis = [legal[i] for i in top_indices]
        cand_probs = [prob_map[i] for i in top_indices]
        cand_log_probs = [log_p_map[i] for i in top_indices]

        # Extract 6x8x8 belief probabilities
        b_probs_oriented = np.zeros((6, 8, 8), dtype=np.float32)
        if has_belief:
            b_logits = out[2][0].cpu()
            shift = 0.0 if raw_sigmoid else self.log_pos_weight
            b_probs_oriented = torch.sigmoid(b_logits - shift).numpy()

        # Run 1-ply lookahead if supported by features.py
        our_1ply_val = np.zeros(len(cand_ucis), dtype=np.float32)
        if has_belief and hasattr(self.features, "step_1ply_tensors") and len(cand_ucis) > 0:
            step_res = self.features.step_1ply_tensors(obs, memory, b_probs_oriented, cand_ucis)
            next_x_np = step_res[0] if isinstance(step_res, tuple) else step_res
            next_x = torch.from_numpy(next_x_np).to(self.device)
            next_out = self.net(next_x)
            if isinstance(next_out, tuple) and len(next_out) > 1 and next_out[1] is not None:
                v_opp = next_out[1].squeeze(-1).cpu().numpy()
                our_1ply_val = np.atleast_1d(-v_opp)

        # Build candidate table rows
        candidates = []
        for i, uci in enumerate(cand_ucis):
            is_king_cap = obs["enemy"].get(uci[2:4]) == "K"
            score = 999.0 if is_king_cap else (cand_log_probs[i] + self.value_weight * float(our_1ply_val[i]))
            candidates.append({
                "uci": uci,
                "prob": cand_probs[i],
                "log_p": cand_log_probs[i],
                "v_1ply": float(our_1ply_val[i]),
                "score": score,
                "king_cap": is_king_cap,
            })
        candidates.sort(key=lambda d: d["score"], reverse=True)

        # Convert oriented belief map (6, 8, 8) to standard board coordinates (rank 0=rank 1, file 0=file a)
        c = obs["color"]
        b_display = b_probs_oriented.copy()
        if fuse_visible:
            # Zero out visible empty squares and set visible enemies to 1.0
            vis_mask = np.zeros((8, 8), dtype=np.float32)
            for s in obs["visible"]:
                r, f = divmod(self.features.orient(self.features.sq_index(s), c), 8)
                vis_mask[r, f] = 1.0
            b_display *= (1.0 - vis_mask[None, :, :])
            for s, t in obs["enemy"].items():
                r, f = divmod(self.features.orient(self.features.sq_index(s), c), 8)
                b_display[:, r, f] = 0.0
                b_display[ORDER.index(t), r, f] = 1.0

        # Un-flip vertically if Black so rank 7 is always Rank 8 (top) and rank 0 is Rank 1 (bottom)
        if c == "black":
            b_display = np.flip(b_display, axis=1)

        return {
            "color": c,
            "v_curr": v_curr,
            "win_pct": 50.0 * (v_curr + 1.0),
            "candidates": candidates,
            "b_display": b_display,  # (6, 8, 8) in standard board orientation
        }


# ---------------------------------------------------------------------------
# Heatmap Renderers (Both Numeric % and ASCII Whiteness Shading)
# ---------------------------------------------------------------------------
def get_true_enemy_squares(board: chess.Board | None, viewer_color: str, piece_key: str) -> set[tuple[int, int]]:
    """Return set of (rank, file) where the opponent's true piece(s) actually sit."""
    if board is None:
        return set()
    enemy_color = chess.BLACK if viewer_color == "white" else chess.WHITE
    coords = set()
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is not None and p.color == enemy_color:
            sym = p.symbol().upper()
            if piece_key == "ALL" or sym == piece_key:
                coords.add((chess.square_rank(sq), chess.square_file(sq)))
    return coords


def render_heatmap_numeric_lines(
    prob_grid: np.ndarray,
    title: str = "HEATMAP (NUMERIC %)",
    true_coords: set[tuple[int, int]] | None = None,
) -> list[str]:
    """Render an (8, 8) probability grid as 00-99 percentages with grayscale background."""
    true_coords = true_coords or set()
    lines = [f"   {BOLD}{title:^22s}{RESET}   "]
    for rank in range(7, -1, -1):
        row_str = f" {rank + 1} "
        for file in range(8):
            p = float(np.clip(prob_grid[rank, file], 0.0, 1.0))
            pct = int(round(p * 100))
            # 24-step ANSI grayscale ramp: 232 (black) -> 255 (bright white)
            gray_idx = 232 + int(round(p * 23))
            bg = f"\033[48;5;{gray_idx}m"
            # Switch text color to dark black when background gets bright white (p >= 0.55)
            fg = "\033[38;5;16m" if p >= 0.55 else "\033[38;5;250m"

            if (rank, file) in true_coords:
                # Mark actual ground-truth enemy location with '*' and green/bold text
                fg = "\033[38;5;22m" if p >= 0.55 else FG_TRUE_MARKER
                cell = f"*{min(pct, 99):02d}"
            elif pct >= 100:
                cell = "100"
            elif pct == 0:
                cell = " · "
            else:
                cell = f" {pct:2d}"

            row_str += f"{bg}{fg}{BOLD}{cell}{RESET}"
        lines.append(row_str)
    lines.append("    a  b  c  d  e  f  g  h  ")
    return lines


def render_heatmap_ascii_lines(
    prob_grid: np.ndarray,
    title: str = "HEATMAP (WHITENESS)",
    true_coords: set[tuple[int, int]] | None = None,
) -> list[str]:
    """Render an (8, 8) probability grid using ASCII density blocks (·, ░, ▒, ▓, █) + luminance."""
    true_coords = true_coords or set()
    lines = [f"   {BOLD}{title:^22s}{RESET}   "]
    for rank in range(7, -1, -1):
        row_str = f" {rank + 1} "
        for file in range(8):
            p = float(np.clip(prob_grid[rank, file], 0.0, 1.0))
            if p < 0.03:
                idx = 0
            elif p < 0.25:
                idx = 1
            elif p < 0.50:
                idx = 2
            elif p < 0.75:
                idx = 3
            else:
                idx = 4

            ch = SHADE_CHARS[idx]
            # Scale foreground brightness from dark gray (236) to pure white (255)
            fg_gray = 236 + int(round(p * 19))
            fg = f"\033[38;5;{fg_gray}m"
            bg = "\033[48;5;232m"

            if (rank, file) in true_coords:
                # Frame the middle character with true-location markers
                mid = ch[1]
                ch = f"{FG_TRUE_MARKER}[{fg}{mid}{FG_TRUE_MARKER}]"

            row_str += f"{bg}{fg}{BOLD}{ch}{RESET}"
        lines.append(row_str)
    lines.append("    a  b  c  d  e  f  g  h  ")
    return lines


def print_brain_summary(
    analysis: dict,
    agent_name: str,
    piece_key: str = "K",
    true_board: chess.Board | None = None,
) -> None:
    """Print Value Head bar, Top-K 1-Ply Candidate table, and Side-by-Side Heatmaps."""
    v = analysis["v_curr"]
    win_pct = analysis["win_pct"]
    c = analysis["color"].upper()

    # Value bar (20 chars wide)
    filled = int(round((win_pct / 100.0) * 20))
    bar = "█" * filled + "░" * (20 - filled)
    v_color = "\033[92m" if v > 0.15 else ("\033[91m" if v < -0.15 else "\033[93m")

    print(f"\n┌─ {BOLD}MODEL BRAIN OVERLAY: {agent_name} ({c}'s Turn){RESET} " + "─" * 28)
    print(f"│ {BOLD}Value Head V(s):{RESET} {v_color}{v:+.3f}{RESET}  [{bar}]  Est. Win Prob: {BOLD}{win_pct:.1f}%{RESET}")
    print("│")
    print(f"│ {BOLD}Top-{len(analysis['candidates'])} Candidate Moves (Pass 1 Policy + Pass 2 1-Ply Value Lookahead):{RESET}")
    print("│   Rank  Move     Policy π(a|s)   log π(a|s)   1-Ply -V(s')   Combined Score")
    print("│   ────  ──────   ─────────────   ──────────   ────────────   ──────────────")
    for rank_i, cand in enumerate(analysis["candidates"], 1):
        marker = "★" if rank_i == 1 else " "
        k_tag = " [WIN: KING CAPTURE]" if cand["king_cap"] else ""
        print(
            f"│  {marker} #{rank_i:<2d}  {BOLD}{cand['uci']:<6s}{RESET}   "
            f"{cand['prob'] * 100:6.1f}%         {cand['log_p']:+7.3f}      "
            f"{cand['v_1ply']:+7.3f}        {BOLD}{cand['score']:+7.3f}{RESET}{k_tag}"
        )
    print("└" + "─" * 76)

    # Extract requested piece plane (or max across all 6 planes for 'ALL')
    b_disp = analysis["b_display"]
    piece_key = piece_key.upper()
    if piece_key == "ALL":
        grid = b_disp.max(axis=0)
    else:
        p_idx = ORDER.index(piece_key) if piece_key in ORDER else 5
        grid = b_disp[p_idx]

    label = PIECE_NAMES.get(piece_key, piece_key)
    true_sq = get_true_enemy_squares(true_board, analysis["color"], piece_key)

    num_lines = render_heatmap_numeric_lines(grid, f"BELIEF %: ENEMY {label.upper()}", true_sq)
    asc_lines = render_heatmap_ascii_lines(grid, f"WHITENESS: ENEMY {label.upper()}", true_sq)

    print()
    for r_i in range(len(num_lines)):
        print("   │   ".join([num_lines[r_i], asc_lines[r_i]]))
    print(
        f"\nHeatmap Legend: {BOLD}00..99{RESET}=Prob %  |  "
        f"{BOLD}· ░ ▒ ▓ █{RESET}=Whiteness Scale  |  "
        f"{FG_TRUE_MARKER}{BOLD}*XX / [█]{RESET}=True Enemy {label} Location"
    )


# ---------------------------------------------------------------------------
# Standard Board Renderers
# ---------------------------------------------------------------------------
def get_side_obs(game: Game, color: str) -> dict:
    """Fog-of-War observation for `color`, even when it is not that side's turn.

    Game.observation() always describes the side to move and builds `visible` from the
    cached `game._legal`. For the other side we recompute the legal moves with the
    reference `legal_moves()` and let Game.observation() do the rest.
    """
    want = chess.WHITE if color == "white" else chess.BLACK
    if game.board.turn == want and not game.over:
        return game.observation()  # exactly what the bot sees

    ghost = Game.__new__(Game)  # skip __init__; set only what observation() reads
    ghost.board = game.board.copy(stack=False)
    ghost.board.turn = want
    if game.board.turn != want:
        ghost.board.ep_square = None  # the ep square belongs to the side really to move
    ghost.lost_on = game.lost_on
    ghost.last_move = game.last_move
    ghost._legal = legal_moves(ghost.board)

    obs = ghost.observation()
    obs["ply"] = game.board.ply()  # flipping the turn would shift ply() by one
    if game.board.turn != want:
        obs["lost_on"] = None  # stale for a side that isn't on move
    return obs


def render_true_board_lines(board: chess.Board, use_unicode: bool = False) -> list[str]:
    lines = [f"   {BOLD}FULL VISIBILITY (TRUE){RESET}   "]
    for rank in range(7, -1, -1):
        row_str = f" {rank + 1} "
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            bg = BG_VISIBLE_LIGHT if (rank + file) % 2 == 1 else BG_VISIBLE_DARK
            if piece is None:
                ch, fg = " · ", ""
            else:
                sym = piece.symbol()
                fg = FG_WHITE_PIECE if piece.color == chess.WHITE else FG_BLACK_PIECE
                disp = UNICODE_PIECES[sym] if use_unicode else sym
                ch = f" {disp} "
            row_str += f"{bg}{fg}{BOLD}{ch}{RESET}"
        lines.append(row_str)
    lines.append("    a  b  c  d  e  f  g  h  ")
    return lines


def render_fog_board_lines(
    obs: dict,
    last_seen: dict[str, str] | None = None,
    title: str = "FOG VIEW",
    use_unicode: bool = False,
) -> list[str]:
    color = obs["color"]
    vis_set = set(obs["visible"])
    own_dict = obs["own"]
    enemy_dict = obs["enemy"]
    lost_on = obs.get("lost_on")
    last_seen = last_seen or {}

    lines = [f"   {BOLD}{title:^22s}{RESET}   "]
    for rank in range(7, -1, -1):
        row_str = f" {rank + 1} "
        for file in range(8):
            sq_name = chess.square_name(chess.square(file, rank))
            is_vis = sq_name in vis_set

            if sq_name == lost_on:
                bg = BG_LOST
            elif is_vis:
                bg = BG_VISIBLE_LIGHT if (rank + file) % 2 == 1 else BG_VISIBLE_DARK
            else:
                bg = BG_FOG

            if sq_name in own_dict:
                p_letter = own_dict[sq_name]
                sym = p_letter if color == "white" else p_letter.lower()
                fg = FG_WHITE_PIECE if color == "white" else FG_BLACK_PIECE
                disp = UNICODE_PIECES[sym] if use_unicode else sym
                ch = f" {disp} "
            elif sq_name in enemy_dict:
                p_letter = enemy_dict[sq_name]
                sym = p_letter.lower() if color == "white" else p_letter
                fg = FG_BLACK_PIECE if color == "white" else FG_WHITE_PIECE
                disp = UNICODE_PIECES[sym] if use_unicode else sym
                ch = f" {disp} "
            elif not is_vis and sq_name in last_seen:
                p_letter = last_seen[sq_name]
                sym = p_letter.lower() if color == "white" else p_letter
                fg = FG_GHOST_PIECE
                disp = UNICODE_PIECES[sym] if use_unicode else sym
                ch = f"?{disp}?"
            else:
                fg = ""
                ch = " · " if is_vis else "░░░"

            row_str += f"{bg}{fg}{BOLD}{ch}{RESET}"
        lines.append(row_str)
    lines.append("    a  b  c  d  e  f  g  h  ")
    return lines


def print_boards_side_by_side(board_line_groups: list[list[str]]) -> None:
    print()
    for row_idx in range(len(board_line_groups[0])):
        print("   │   ".join(group[row_idx] for group in board_line_groups))
    print(
        f"\nBoard Legend: {FG_WHITE_PIECE}{BOLD}UPPER/White{RESET} | "
        f"{FG_BLACK_PIECE}{BOLD}lower/Cyan=Black{RESET} | "
        f"{FG_GHOST_PIECE}{BOLD}?x?=Remembered (last_seen){RESET} | "
        f"░░░=Fog | ·=Visible Empty"
    )


def try_load_brain(spec_or_dir: str | None, device: str = "cpu") -> BrainInspector | None:
    if not spec_or_dir or spec_or_dir.startswith("bot:"):
        return None
    if os.path.isfile(os.path.join(spec_or_dir, "config.json")):
        try:
            return BrainInspector(spec_or_dir, device=device)
        except Exception as e:
            print(f"[Warning] Could not load BrainInspector for {spec_or_dir}: {e}")
    return None


# ---------------------------------------------------------------------------
# MODE 1: Play interactively against any bot/agent
# ---------------------------------------------------------------------------
def cmd_play(args):
    opp_color = "black" if args.color == "white" else "white"
    print(f"Loading opponent '{args.opponent}' on {args.device}...")
    opponent = make_player(args.opponent, device=args.device)
    opponent.reset(opp_color, args.seed)

    brain = try_load_brain(args.model or args.opponent, args.device)
    game = Game()
    mem = {
        "white": brain.new_memory() if brain else Memory(),
        "black": brain.new_memory() if brain else Memory(),
    }
    show_all_three = args.all_boards
    show_brain = args.brain
    heat_piece = args.piece.upper()
    last_opp_analysis = None

    try:
        while not game.over:
            w_obs = get_side_obs(game, "white")
            b_obs = get_side_obs(game, "black")
            cur_color = game.turn
            cur_obs = game.observation()
            mem[cur_color].update(cur_obs)

            if cur_color == args.color:
                if show_all_three:
                    my_obs = w_obs if args.color == "white" else b_obs
                    en_obs = b_obs if args.color == "white" else w_obs
                    print_boards_side_by_side([
                        render_true_board_lines(game.board, args.unicode),
                        render_fog_board_lines(my_obs, mem[args.color].last_seen, f"MY FOG ({args.color.upper()})", args.unicode),
                        render_fog_board_lines(en_obs, mem[opp_color].last_seen, f"ENEMY FOG ({opp_color.upper()})", args.unicode),
                    ])
                else:
                    print_boards_side_by_side([
                        render_fog_board_lines(cur_obs, mem[args.color].last_seen, f"MY FOG ({args.color.upper()})", args.unicode)
                    ])

                if show_brain and brain is not None:
                    # Show either how the model evaluates YOUR current turn or what the opponent just thought
                    my_analysis = brain.analyze(cur_obs, mem[args.color])
                    print_brain_summary(
                        my_analysis,
                        f"{brain.name} (Advising {args.color.upper()})",
                        heat_piece,
                        game.board if show_all_three else None,
                    )
                    if last_opp_analysis is not None and show_all_three:
                        print_brain_summary(
                            last_opp_analysis,
                            f"{brain.name} (Opponent's Last Turn Thought)",
                            heat_piece,
                            game.board,
                        )

                print(f"\nPly {cur_obs['ply']} | Lost piece on: {cur_obs['lost_on']} | Legal moves ({len(cur_obs['legal_moves'])}): {', '.join(cur_obs['legal_moves'][:15])}...")
                while True:
                    cmd = input(
                        f"[{args.color.upper()}] Move UCI ('v'=3-boards, 'b'=brain, 'h P/N/B/R/Q/K/ALL'=heatmap piece, 'q'=quit): "
                    ).strip()
                    if cmd.lower() == "q":
                        return
                    if cmd.lower() == "v":
                        show_all_three = not show_all_three
                        break
                    if cmd.lower() == "b":
                        show_brain = not show_brain
                        break
                    if cmd.lower().startswith("h "):
                        heat_piece = cmd.split()[1].upper()
                        show_brain = True
                        break
                    if cmd in cur_obs["legal_moves"]:
                        game.play(cmd)
                        break
                    print(f"Illegal move '{cmd}'. Matching legal moves: {[m for m in cur_obs['legal_moves'] if m.startswith(cmd[:2])]}")
            else:
                if brain is not None:
                    last_opp_analysis = brain.analyze(cur_obs, mem[cur_color])
                move, flops = opponent.act(cur_obs, timeout=30.0)
                print(f"\nOpponent ({args.opponent}) played: {move} ({flops:,} FLOPs)")
                game.play(move)

        print(f"\n=== GAME OVER: {game.result} ({game.termination}) ===")
        print_boards_side_by_side([render_true_board_lines(game.board, args.unicode)])
    finally:
        opponent.close()


# ---------------------------------------------------------------------------
# MODE 2: Watch two bots play OR step through a recorded game's move list
# ---------------------------------------------------------------------------
def step_through_moves(
    moves: list[str],
    white_name: str,
    black_name: str,
    unicode: bool = False,
    w_brain: BrainInspector | None = None,
    b_brain: BrainInspector | None = None,
    init_piece: str = "K",
):
    states = []
    game = Game()
    mem = {
        "white": w_brain.new_memory() if w_brain else Memory(),
        "black": b_brain.new_memory() if b_brain else Memory(),
    }

    for ply in range(len(moves) + 1):
        w_obs = get_side_obs(game, "white")
        b_obs = get_side_obs(game, "black")
        brain_analysis = None
        active_brain_name = None

        if not game.over:
            cur_c = game.turn
            cur_obs = game.observation()
            mem[cur_c].update(cur_obs)
            active_brain = w_brain if cur_c == "white" else b_brain
            if active_brain is not None:
                brain_analysis = active_brain.analyze(cur_obs, mem[cur_c])
                active_brain_name = active_brain.name

        states.append((
            game.board.copy(),
            w_obs,
            b_obs,
            dict(mem["white"].last_seen),
            dict(mem["black"].last_seen),
            moves[ply - 1] if ply > 0 else "START",
            moves[ply] if ply < len(moves) else "END",
            brain_analysis,
            active_brain_name,
        ))
        if ply < len(moves):
            game.play(moves[ply])

    idx = 0
    show_brain = True
    heat_piece = init_piece.upper()

    while True:
        board, w_obs, b_obs, w_mem, b_mem, prev_mv, next_mv, b_anal, b_name = states[idx]
        print(f"\n=== {white_name} (White) vs {black_name} (Black) | Ply {idx}/{len(moves)} ===")
        print(f"Last Move Played: {prev_mv}  -->  Actual Move Chosen Next: {BOLD}{next_mv}{RESET}")
        print_boards_side_by_side([
            render_true_board_lines(board, unicode),
            render_fog_board_lines(w_obs, w_mem, "WHITE FOG VIEW", unicode),
            render_fog_board_lines(b_obs, b_mem, "BLACK FOG VIEW", unicode),
        ])

        if show_brain and b_anal is not None:
            print_brain_summary(b_anal, b_name, heat_piece, board)

        cmd = input(
            "[Enter/n]=next, [p]=prev, [ply #]=jump, [b]=toggle brain, [h P/N/B/R/Q/K/ALL]=heatmap piece, [q]=quit: "
        ).strip()
        if cmd.lower() == "q":
            break
        elif cmd.lower() == "p":
            idx = max(0, idx - 1)
        elif cmd.lower() == "b":
            show_brain = not show_brain
        elif cmd.lower().startswith("h "):
            heat_piece = cmd.split()[1].upper()
            show_brain = True
        elif cmd.isdigit():
            idx = max(0, min(len(states) - 1, int(cmd)))
        else:
            if idx < len(states) - 1:
                idx += 1
            else:
                print(f"End of game: {game.result} ({game.termination})")


def cmd_watch(args):
    print(f"Running match: {args.white} (White) vs {args.black} (Black) [seed={args.seed}]...")
    from fow.referee import TimeControl, play_game
    w = make_player(args.white, device=args.device)
    b = make_player(args.black, device=args.device)
    try:
        rec = play_game(w, b, args.seed, TimeControl())
        print(f"Match complete: {rec.result} ({rec.termination}) in {rec.plies} plies.")
        w_brain = try_load_brain(args.model or args.white, args.device)
        b_brain = try_load_brain(args.model or args.black, args.device)
        step_through_moves(rec.moves, args.white, args.black, args.unicode, w_brain, b_brain, args.piece)
    finally:
        w.close()
        b.close()


def cmd_replay(args):
    with open(args.jsonl_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        print("No games found in file.")
        return
    rec = records[args.index]
    print(f"Loaded game #{args.index}: {rec['white']} vs {rec['black']} -> {rec['result']} ({rec['termination']})")
    w_brain = try_load_brain(args.model or rec["white"], args.device)
    b_brain = try_load_brain(args.model or rec["black"], args.device)
    step_through_moves(rec["moves"], rec["white"], rec["black"], args.unicode, w_brain, b_brain, args.piece)


# ---------------------------------------------------------------------------
# MODE 3: Inspect encoded positions inside a `.npz` training shard
# ---------------------------------------------------------------------------
def decode_x_planes_to_obs(x_planes: np.ndarray) -> tuple[dict, dict[str, str]]:
    own, enemy, last_seen, visible = {}, {}, {}, []
    for r in range(8):
        for f in range(8):
            sq_name = chess.square_name(chess.square(f, r))
            if x_planes[12, r, f] > 0.5:
                visible.append(sq_name)
            for p_idx, p_char in enumerate(ORDER):
                if x_planes[p_idx, r, f] > 0.5:
                    own[sq_name] = p_char
                if x_planes[6 + p_idx, r, f] > 0.5:
                    enemy[sq_name] = p_char
                mem_offset = 25 if x_planes.shape[0] >= 34 else 13
                if x_planes[mem_offset + p_idx, r, f] > 0.5 and sq_name not in enemy:
                    last_seen[sq_name] = p_char

    lost_on = None
    lost_plane = 33 if x_planes.shape[0] >= 34 else 21
    if x_planes.shape[0] > lost_plane:
        ys, xs = np.where(x_planes[lost_plane] > 0.5)
        if len(ys) > 0:
            lost_on = chess.square_name(chess.square(int(xs[0]), int(ys[0])))

    obs = {"color": "white", "own": own, "enemy": enemy, "visible": sorted(visible), "lost_on": lost_on}
    return obs, last_seen


def index_to_uci(move_idx: int) -> str:
    from_sq, to_sq = divmod(int(move_idx), 64)
    return chess.square_name(from_sq) + chess.square_name(to_sq)


def cmd_shard(args):
    data = np.load(args.npz_path)
    print(f"Loaded shard: {args.npz_path}")
    print("Keys in shard:", {k: data[k].shape for k in data.files})
    xs = data["x"]
    N = len(xs)
    brain = try_load_brain(args.model, args.device)
    heat_piece = args.piece.upper()
    idx = args.index

    while True:
        x_planes = xs[idx]
        obs, last_seen = decode_x_planes_to_obs(x_planes)
        info_parts = [f"Sample {idx}/{N - 1}", f"Planes: {x_planes.shape[0]}"]
        for k in ("y", "y_played", "y_teacher"):
            if k in data:
                mv_id = int(data[k][idx])
                info_parts.append(f"{k}={mv_id} ({index_to_uci(mv_id)})")
        for k in ("z", "has_teacher"):
            if k in data:
                info_parts.append(f"{k}={data[k][idx]}")
        print("\n" + " | ".join(info_parts))

        board_lines = [render_fog_board_lines(obs, last_seen, "CANONICAL SHARD VIEW", args.unicode)]

        # Unpack ground-truth belief target (`belief` packed 48-byte uint8 or `b_target` 6x8x8)
        true_board = None
        b_t = None
        if "b_target" in data:
            b_t = data["b_target"][idx]
        elif "belief" in data:
            b_t = np.unpackbits(data["belief"][idx]).reshape(6, 8, 8)

        if b_t is not None:
            true_board = chess.Board(None)
            for sq_name, p_char in obs["own"].items():
                true_board.set_piece_at(chess.parse_square(sq_name), chess.Piece.from_symbol(p_char))
            for p_idx, p_char in enumerate(ORDER):
                for r in range(8):
                    for f in range(8):
                        if b_t[p_idx, r, f] > 0.5:
                            true_board.set_piece_at(chess.square(f, r), chess.Piece.from_symbol(p_char.lower()))
            board_lines.insert(0, render_true_board_lines(true_board, args.unicode))

        print_boards_side_by_side(board_lines)

        # If `--model` was provided, run the network directly on the shard's `x_planes`!
        if brain is not None and x_planes.shape[0] == brain.cfg.get("in_planes", 34):
            with torch.no_grad():
                xt = torch.from_numpy(x_planes).float().unsqueeze(0).to(brain.device)
                out = brain.net(xt)
                v_pred = float(out[1][0].item()) if len(out) > 1 else 0.0
                b_logits = out[2][0].cpu()
                b_probs = torch.sigmoid(b_logits - brain.log_pos_weight).numpy()

            p_idx = ORDER.index(heat_piece) if heat_piece in ORDER else 5
            grid = b_probs.max(axis=0) if heat_piece == "ALL" else b_probs[p_idx]
            true_sq = get_true_enemy_squares(true_board, "white", heat_piece)
            label = PIECE_NAMES.get(heat_piece, heat_piece)

            print(f"\nModel ({brain.name}) on Shard Sample #{idx} -> Value V(s) = {v_pred:+.3f}")
            num_l = render_heatmap_numeric_lines(grid, f"BELIEF %: ENEMY {label.upper()}", true_sq)
            asc_l = render_heatmap_ascii_lines(grid, f"WHITENESS: ENEMY {label.upper()}", true_sq)
            for r_i in range(len(num_l)):
                print("   │   ".join([num_l[r_i], asc_l[r_i]]))

        cmd = input("[Enter/n]=next, [p]=prev, [index #]=jump, [h P/N/B/R/Q/K/ALL]=piece, [q]=quit: ").strip()
        if cmd.lower() == "q":
            break
        elif cmd.lower() == "p":
            idx = max(0, idx - 1)
        elif cmd.lower().startswith("h "):
            heat_piece = cmd.split()[1].upper()
        elif cmd.isdigit():
            idx = max(0, min(N - 1, int(cmd)))
        else:
            idx = min(N - 1, idx + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unicode", action="store_true", help="use Unicode chess symbols (♔♕♖♗♘♙)")
    ap.add_argument("--piece", default="K", help="initial enemy piece for belief heatmap: P, N, B, R, Q, K, or ALL")
    sub = ap.add_subparsers(dest="mode", required=True)

    p_play = sub.add_parser("play", help="play interactively against a bot or agent")
    p_play.add_argument("opponent", help="e.g. agents/v3_se_model or bot:greedy")
    p_play.add_argument("--color", choices=["white", "black"], default="white")
    p_play.add_argument("--all-boards", action="store_true", help="start with all 3 boards visible")
    p_play.add_argument("--brain", action="store_true", help="start with Model Brain & Heatmaps enabled")
    p_play.add_argument("--model", default=None, help="optional agent dir to run Brain Overlay if playing vs bot:*")
    p_play.add_argument("--device", default="cpu")
    p_play.add_argument("--seed", type=int, default=42)
    p_play.set_defaults(func=cmd_play)

    p_watch = sub.add_parser("watch", help="run a game between two bots and step through all 3 boards + Brain")
    p_watch.add_argument("white", help="white player spec")
    p_watch.add_argument("black", help="black player spec")
    p_watch.add_argument("--model", default=None, help="override agent dir for Brain Overlay")
    p_watch.add_argument("--device", default="cpu")
    p_watch.add_argument("--seed", type=int, default=42)
    p_watch.set_defaults(func=cmd_watch)

    p_rep = sub.add_parser("replay", help="replay a game from an `arena.py --out` JSONL file")
    p_rep.add_argument("jsonl_file", help="path to JSONL file")
    p_rep.add_argument("--index", type=int, default=0, help="game index in file")
    p_rep.add_argument("--model", default=None, help="agent dir to run Brain Overlay during replay")
    p_rep.add_argument("--device", default="cpu")
    p_rep.set_defaults(func=cmd_replay)

    p_shard = sub.add_parser("shard", help="inspect positions inside a .npz training shard")
    p_shard.add_argument("npz_path", help="path to .npz shard")
    p_shard.add_argument("--index", type=int, default=0)
    p_shard.add_argument("--model", default=None, help="optional agent dir to test its belief heatmap on the shard")
    p_shard.add_argument("--device", default="cpu")
    p_shard.set_defaults(func=cmd_shard)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()