"""Master iterative training loop with teacher-decay schedule and league opponent pool.

    python training/run_iterations.py --iterations 5 --warmup-games 2500 --games-per-iter 1500 \
        --workers 4 --agent-dir agents/v1_se_model --lambda-start 1.0 --lambda-end 0.2
"""
import argparse
import os
import shutil
import subprocess
import sys


def run_cmd(cmd):
    print(f"\n>>> Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def snapshot_agent(agent_dir: str, league_root: str, iteration: int) -> str:
    snap_dir = os.path.join(league_root, f"iter_{iteration:03d}")
    if os.path.exists(snap_dir):
        shutil.rmtree(snap_dir)
    shutil.copytree(agent_dir, snap_dir)
    return snap_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=5, help="Total Gen -> Train iterations")
    ap.add_argument("--warmup-games", type=int, default=2500, help="Games in Iteration 0 (warmup)")
    ap.add_argument("--games-per-iter", type=int, default=1500, help="New games added per subsequent iteration")
    ap.add_argument("--workers", type=int, default=4, help="CPU workers for data generation")
    ap.add_argument("--nodes", type=int, default=5000, help="Stockfish nodes per teacher label")
    ap.add_argument("--teacher-rate", type=float, default=0.5, help="Fraction of positions labeled by Stockfish")
    ap.add_argument("--keep", type=float, default=1.0, help="Fraction of positions kept for outcome training")
    ap.add_argument("--buffer-dir", default="data/replay_buffer")
    ap.add_argument("--league-dir", default="data/league", help="Directory storing past agent snapshots for league play")
    ap.add_argument("--max-buffer", type=int, default=500_000, help="FIFO buffer capacity (positions)")
    ap.add_argument("--evict-warmup-after", type=int, default=2, help="Evict iter 0 warmup shards starting at this iteration")
    ap.add_argument("--agent-dir", default="submission")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--steps-per-iter", type=int, default=800, help="2048-batch SGD steps per iteration")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--outcome-mode", choices=["win_only", "advantage", "none"], default="advantage")
    ap.add_argument("--lambda-start", type=float, default=1.0, help="Teacher weight lambda on Iteration 0")
    ap.add_argument("--lambda-end", type=float, default=0.2, help="Teacher weight lambda on final iteration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    py = sys.executable
    os.makedirs(args.league_dir, exist_ok=True)
    league_snapshots = []

    for it in range(args.iterations):
        n_games = args.warmup_games if it == 0 else args.games_per_iter

        # Linearly anneal teacher weight lambda from lambda_start -> lambda_end
        if args.iterations > 1:
            frac = it / (args.iterations - 1)
            lam = args.lambda_start + frac * (args.lambda_end - args.lambda_start)
        else:
            lam = args.lambda_start

        # On iteration 0 warmup, reference bots play, so weight teacher higher; outcome kicks in fully on iter >= 1
        outcome_w = 0.5 if it == 0 else 1.0
        min_iter = 1 if (args.evict_warmup_after > 0 and it >= args.evict_warmup_after) else 0

        # 1. Generate new shard(s) into the replay buffer
        gen_cmd = [
            py, "training/generate_buffer.py",
            "--iter", str(it),
            "--games", str(n_games),
            "--workers", str(args.workers),
            "--nodes", str(args.nodes),
            "--teacher-rate", str(args.teacher_rate),
            "--keep", str(args.keep),
            "--agent-dir", args.agent_dir,
            "--out", args.buffer_dir,
            "--seed", str(args.seed),
        ]
        if league_snapshots:
            gen_cmd.extend(["--pool-dirs"] + league_snapshots[-4:])
        run_cmd(gen_cmd)

        # 2. Train model on 2,048-position random batches sampled from the FIFO buffer
        train_cmd = [
            py, "training/train_buffer.py",
            "--buffer-dir", args.buffer_dir,
            "--agent-dir", args.agent_dir,
            "--max-buffer", str(args.max_buffer),
            "--min-iter", str(min_iter),
            "--channels", str(args.channels),
            "--blocks", str(args.blocks),
            "--steps", str(args.steps_per_iter),
            "--batch", str(args.batch),
            "--lr", str(args.lr),
            "--wd", str(args.wd),
            "--outcome-mode", args.outcome_mode,
            "--outcome-weight", f"{outcome_w:.3f}",
            "--teacher-weight", f"{lam:.3f}",
            "--seed", str(args.seed + it),
            "--device", args.device,
        ]
        if it > 0:
            train_cmd.append("--resume")
        run_cmd(train_cmd)

        # 3. Save checkpoint into the opponent league pool
        snap = snapshot_agent(args.agent_dir, args.league_dir, it)
        league_snapshots.append(snap)

    print(f"\n=== Completed all {args.iterations} iterations! Validating {args.agent_dir}... ===")
    run_cmd([py, "-m", "fow.validate", args.agent_dir])


if __name__ == "__main__":
    main()