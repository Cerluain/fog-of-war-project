"""Diagnostic: Measure how often full-board Stockfish agrees with fog-aware guess-sf.

    python training/diagnose_labels.py --games 100 --workers 4 --guess-spec bot:guess-sf-8x500
"""
import argparse
import multiprocessing as mp
import os
import random
import sys

import chess
import chess.engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fow.bots import make_bot, stockfish_path  # noqa: E402
from fow.game import Game  # noqa: E402


def full_board_sf(engine, game, nodes):
    for mv in game.legal_uci():
        if game.board.piece_type_at(chess.parse_square(mv[2:4])) == chess.KING:
            return mv
    try:
        mv = engine.play(game.board.copy(stack=False), chess.engine.Limit(nodes=nodes)).move.uci()
    except Exception:
        return None
    return mv if mv in game.legal_uci() else None


def worker(args):
    wid, n_games, nodes, guess_spec, seed = args
    rng = random.Random(seed + wid)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path())
    engine.configure({"Threads": 1, "Hash": 16})

    guess_bot = make_bot(guess_spec)
    opp_bot = make_bot("bot:greedy")

    total, agree, king_cap_total = 0, 0, 0
    by_phase = {"opening (1-15)": [0, 0], "midgame (16-40)": [0, 0], "endgame (41+)": [0, 0]}

    for _ in range(n_games):
        guess_color = "white" if rng.random() < 0.5 else "black"
        guess_bot.reset(guess_color, rng.randrange(1 << 30))
        opp_color = "black" if guess_color == "white" else "white"
        opp_bot.reset(opp_color, rng.randrange(1 << 30))

        game = Game()
        ply = 0
        while not game.over:
            obs = game.observation()
            c = obs["color"]
            ply += 1
            if c == guess_color:
                sf_mv = full_board_sf(engine, game, nodes)
                g_mv = guess_bot.act(obs)
                if sf_mv is not None:
                    is_kcap = any(obs["enemy"].get(u[2:4]) == "K" for u in obs["legal_moves"])
                    if is_kcap:
                        king_cap_total += 1
                    total += 1
                    match = int(sf_mv[:4] == g_mv[:4])
                    agree += match
                    phase = "opening (1-15)" if ply <= 15 else ("midgame (16-40)" if ply <= 40 else "endgame (41+)")
                    by_phase[phase][0] += match
                    by_phase[phase][1] += 1
                game.play(g_mv)
            else:
                game.play(opp_bot.act(obs))

    engine.quit()
    for b in (guess_bot, opp_bot):
        if hasattr(b, "close"):
            b.close()
    return total, agree, king_cap_total, by_phase


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nodes", type=int, default=5000)
    ap.add_argument("--guess-spec", default="bot:guess-sf-8x500")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    per = [args.games // args.workers + (1 if i < args.games % args.workers else 0) for i in range(args.workers)]
    jobs = [(i, per[i], args.nodes, args.guess_spec, args.seed) for i in range(args.workers) if per[i] > 0]

    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        results = pool.map(worker, jobs)

    total = sum(r[0] for r in results)
    agree = sum(r[1] for r in results)
    kcap = sum(r[2] for r in results)

    print(f"\n=== Label Agreement Diagnostic ({args.guess_spec} vs Full-Board SF @ {args.nodes} nodes) ===")
    print(f"Positions evaluated : {total:,} (including {kcap:,} immediate king captures)")
    print(f"Overall agreement   : {agree / max(1, total):.3f} ({agree:,}/{total:,})")
    print(f"Implied noise bound : {1.0 - agree / max(1, total):.3f} hidden-info divergence")
    for phase in ("opening (1-15)", "midgame (16-40)", "endgame (41+)"):
        m = sum(r[3][phase][0] for r in results)
        n = sum(r[3][phase][1] for r in results)
        if n > 0:
            print(f"  {phase:16s}: {m / n:.3f} ({m:,}/{n:,})")


if __name__ == "__main__":
    main()