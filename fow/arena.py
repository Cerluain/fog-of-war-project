"""Play matches and estimate ratings.

Examples
    # your agent against two reference bots, 40 games each (half as white)
    python -m fow.arena submission bot:greedy bot:naive-sf --games 40

    # every player against every other player
    python -m fow.arena submission agents/baseline bot:greedy --mode round-robin --games 20

    # on a GPU node, 4 games at a time, with the tournament per-move limits
    python -m fow.arena submission bot:guess-sf-4x500 --device cuda --workers 4 --official

    # My template for many of these: 
    python -m fow.arena agents/baseline agents/v1_se_model bot:greedy bot:guess-sf-20x500 --games 100 --workers 3 --device cuda

Ratings are printed on the tournament scale: reference bots keep their published ratings
(fow/bots.py REFERENCE_RATINGS); everyone else is estimated from the games played here.
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from dataclasses import asdict

from . import elo
from .bots import REFERENCE_RATINGS
from .referee import TimeControl, make_player, play_game

_players: dict = {}
_cfg: dict = {}


def _init(cfg):
    _cfg.update(cfg)


def _run(job):
    white, black, seed = job
    for spec in (white, black):
        if spec not in _players:
            _players[spec] = make_player(spec, _cfg["device"], _cfg["log_dir"])
    tc = TimeControl(max_flops=_cfg["max_flops"], move_cap=_cfg["move_cap"],
                     clock=_cfg["clock"], increment=_cfg["increment"])
    rec = play_game(_players[white], _players[black], seed, tc)
    rec.white, rec.black = white, black
    return asdict(rec)


def _close_all(_):
    for p in _players.values():
        p.close()
    _players.clear()
    return True


def schedule(players, mode, games, seed):
    pairs = [(players[0], p) for p in players[1:]] if mode == "vs" else list(itertools.combinations(players, 2))
    jobs = []
    for a, b in pairs:
        for g in range(games):
            s = seed * 1_000_003 + len(jobs)
            jobs.append((a, b, s) if g % 2 == 0 else (b, a, s))
    return jobs


def report(records, pinned):
    score = defaultdict(lambda: [0.0, 0])
    for r in records:
        sw = 1.0 if r["result"] == "1-0" else 0.5 if r["result"] == "1/2-1/2" else 0.0
        key = tuple(sorted((r["white"], r["black"])))
        first_is_white = key[0] == r["white"]
        score[key][0] += sw if first_is_white else 1 - sw
        score[key][1] += 1
    print("\nResults (score of the first player in each pair):")
    for (a, b), (s, n) in sorted(score.items()):
        print(f"  {a:30s} vs {b:30s} {s:5.1f} / {n}")
    terms = defaultdict(int)
    for r in records:
        terms[r["termination"]] += 1
    print("\nHow games ended:", dict(terms))
    games = [(r["white"], r["black"], 1.0 if r["result"] == "1-0" else 0.5 if r["result"] == "1/2-1/2" else 0.0)
             for r in records]
    ratings = elo.fit_with_intervals(games, pinned)
    print("\nRatings (95% interval):")
    for p, (v, lo, hi) in sorted(ratings.items(), key=lambda x: -x[1][0]):
        tag = " (reference, fixed)" if p in pinned else ""
        print(f"  {p:30s} {v:7.0f}   [{lo:5.0f}, {hi:5.0f}]{tag}")
    return ratings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("players", nargs="+", help="agent directories or bot names (bot:random, bot:greedy, bot:naive-sf, bot:guess-sf-KxN)")
    ap.add_argument("--mode", choices=["vs", "round-robin"], default="vs",
                    help="vs: the first player against each of the others (default)")
    ap.add_argument("--games", type=int, default=20, help="games per pairing, colors alternate")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-flops", type=float, default=None,
                    help="per-move FLOP budget; tournaments use fow.limits.MAX_MOVE_FLOPS")
    ap.add_argument("--official", action="store_true", help="use the tournament limits (fow/limits.py)")
    ap.add_argument("--move-cap", type=float, default=None, help="wall-clock seconds per move (hang guard)")
    ap.add_argument("--clock", type=float, default=None, help="optional chess clock: seconds per side per game")
    ap.add_argument("--increment", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write one JSON line per game here")
    ap.add_argument("--log-dir", default="arena_logs", help="agents' stderr goes here")
    args = ap.parse_args(argv)

    jobs = schedule(args.players, args.mode, args.games, args.seed)
    if args.official:
        from .limits import MAX_MOVE_FLOPS, MOVE_WALL_SECONDS
        args.max_flops = args.max_flops or MAX_MOVE_FLOPS
        args.move_cap = args.move_cap or MOVE_WALL_SECONDS
    cfg = dict(device=args.device, log_dir=args.log_dir, clock=args.clock, increment=args.increment,
               move_cap=args.move_cap, max_flops=args.max_flops)
    print(f"Playing {len(jobs)} games with {args.workers} worker(s)...", flush=True)
    records = []
    out = open(args.out, "w") if args.out else None
    if args.workers <= 1:
        _init(cfg)
        it = map(_run, jobs)
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(args.workers, initializer=_init, initargs=(cfg,))
        it = pool.imap_unordered(_run, jobs)
    for i, rec in enumerate(it, 1):
        records.append(rec)
        if out:
            out.write(json.dumps(rec) + "\n")
            out.flush()
        if i % max(1, len(jobs) // 10) == 0:
            print(f"  {i}/{len(jobs)} games", flush=True)
    if args.workers <= 1:
        _close_all(None)
    else:
        pool.close()
        pool.terminate()
    pinned = {p: v for p, v in REFERENCE_RATINGS.items()} or {"bot:random": 0.0}
    report(records, pinned)


if __name__ == "__main__":
    main()
