"""Master iterative training loop (AlphaZero / DAgger style):
Alternates between generating new games into the FIFO replay buffer and training the network.

    python training/run_iterations.py --iterations 4 --games-per-iter 1500 --workers 4 --agent-dir agents/v1_se_model
"""
import argparse
import subprocess
import sys


def run_cmd(cmd):
    print(f"\n>>> Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=4, help="Total Gen -> Train iterations")
    ap.add_argument("--warmup-games", type=int, default=3000, help="Games in Iteration 0 (warmup)")
    ap.add_argument("--games-per-iter", type=int, default=1500, help="New games added per subsequent iteration")
    ap.add_argument("--workers", type=int, default=4, help="CPU workers for data generation")
    ap.add_argument("--nodes", type=int, default=5000, help="Stockfish nodes per label")
    ap.add_argument("--keep", type=float, default=0.5, help="Fraction of positions kept per game")
    ap.add_argument("--buffer-dir", default="data/replay_buffer")
    ap.add_argument("--max-buffer", type=int, default=500_000, help="FIFO buffer capacity (positions)")
    ap.add_argument("--agent-dir", default="submission")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--steps-per-iter", type=int, default=800, help="2048-batch SGD steps per iteration")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    py = sys.executable

    for it in range(args.iterations):
        n_games = args.warmup_games if it == 0 else args.games_per_iter

        # 1. Generate new shard(s) into the replay buffer
        gen_cmd = [
            py, "training/generate_buffer.py",
            "--iter", str(it),
            "--games", str(n_games),
            "--workers", str(args.workers),
            "--nodes", str(args.nodes),
            "--keep", str(args.keep),
            "--agent-dir", args.agent_dir,
            "--out", args.buffer_dir,
            "--seed", str(args.seed),
        ]
        run_cmd(gen_cmd)

        # 2. Train model on 2,048-position random batches sampled from the FIFO buffer
        train_cmd = [
            py, "training/train_buffer.py",
            "--buffer-dir", args.buffer_dir,
            "--agent-dir", args.agent_dir,
            "--max-buffer", str(args.max_buffer),
            "--channels", str(args.channels),
            "--blocks", str(args.blocks),
            "--steps", str(args.steps_per_iter),
            "--batch", str(args.batch),
            "--lr", str(args.lr),
            "--wd", str(args.wd),
            "--seed", str(args.seed + it),
            "--device", args.device,
        ]
        if it > 0:
            train_cmd.append("--resume")
        run_cmd(train_cmd)

    print(f"\n=== Completed all {args.iterations} iterations! Validating {args.agent_dir}... ===")
    run_cmd([py, "-m", "fow.validate", args.agent_dir])


if __name__ == "__main__":
    main()