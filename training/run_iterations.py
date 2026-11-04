"""Master iterative training loop with automatic logging to data/logs and plotting to data/plots."""
import argparse
import glob
import os
import shutil
import subprocess
import sys


def get_agent_name(agent_dir: str) -> str:
    return os.path.basename(os.path.normpath(agent_dir))


def run_cmd(cmd, log_file=None):
    header = f"\n>>> Running: {' '.join(cmd)}\n"
    print(header, end="", flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(header)
            lf.flush()
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            for line in proc.stdout:
                print(line, end="", flush=True)
                lf.write(line)
                lf.flush()
            proc.wait()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)
    else:
        subprocess.run(cmd, check=True)


def snapshot_agent(agent_dir: str, league_root: str, iteration: int) -> str:
    snap_dir = os.path.join(league_root, f"iter_{iteration:03d}")
    if os.path.exists(snap_dir):
        shutil.rmtree(snap_dir)
    shutil.copytree(agent_dir, snap_dir)
    return snap_dir

def compute_3stage_weights(it: int, stage1_end: int, stage2_end: int):
    """Smooth 3-block curriculum weights (w_early, w_mid, w_late)."""
    if it <= 1:
        return 0.85, 0.15, 0.0
    if it <= stage1_end:
        # Block 1 (Early -> Mid): Early ramps 0.85 -> 0.25, Mid ramps 0.15 -> 0.75
        f = (it - 1) / max(1, stage1_end - 1)
        w_e = 0.85 - 0.60 * f
        w_m = 0.15 + 0.60 * f
        return w_e, w_m, 0.0
    if it <= stage2_end:
        # Block 2 (Mid -> Late): Early 0.25 -> 0.10, Mid 0.75 -> 0.25, Late 0.0 -> 0.65
        f = (it - stage1_end) / max(1, stage2_end - stage1_end)
        w_e = 0.25 - 0.15 * f
        w_m = 0.75 - 0.50 * f
        w_l = 0.65 * f
        return w_e, w_m, w_l
    # Block 3 (Late Hardening): 10% Early, 20% Mid, 70% Late
    return 0.10, 0.20, 0.70


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=15, help="Total Gen -> Train iterations")
    ap.add_argument("--warmup-games", type=int, default=7500, help="Games in Iteration 0 (warmup)")
    ap.add_argument("--games-per-iter", type=int, default=1300, help="New games added per subsequent iteration")
    ap.add_argument("--workers", type=int, default=6, help="CPU workers for data generation")
    ap.add_argument("--nodes", type=int, default=5000, help="Stockfish nodes per teacher label")
    ap.add_argument("--teacher-rate-start", type=float, default=0.5, help="Stockfish labeling rate on Iteration 0")
    ap.add_argument("--teacher-rate-end", type=float, default=0.15, help="Stockfish labeling rate on final iteration")
    ap.add_argument(
        "--teacher-decay-power", type=float, default=0.5,
        help="Front-loads teacher_rate & lambda decay (<1.0 = steeper early drop; 0.5 = sqrt warp, 1.0 = standard exp)",
    )
    ap.add_argument(
        "--lock-early-matchups-by", type=int, default=30,
        help="Iteration by which early_share reaches early-share-end (0 = ramp across all iterations)",
    )
    ap.add_argument("--stage1-end", type=int, default=18, help="Iteration where Block 1 (Early focus) transitions into Block 2 (Mid focus)")
    ap.add_argument("--stage2-end", type=int, default=45, help="Iteration where Block 2 (Mid focus) transitions into Block 3 (Late focus)")
    ap.add_argument("--keep", type=float, default=1.0, help="Fraction of positions kept for outcome training")
    ap.add_argument("--keep-draw", type=float, default=0.20, help="Fraction of drawn positions kept in the replay buffer")
    ap.add_argument("--buffer-dir", default="data/replay_buffer")
    ap.add_argument("--league-dir", default="data/league", help="Directory storing past agent snapshots for league play")
    ap.add_argument("--log-dir", default="data/logs", help="Directory for agent training logs")
    ap.add_argument("--plot-dir", default="data/plots", help="Directory for output plots")
    ap.add_argument("--max-buffer", type=int, default=500_000, help="FIFO buffer capacity (positions)")
    ap.add_argument("--evict-warmup-after", type=int, default=0, help="Evict iter 0 warmup shards starting at this iteration (0 = keep in FIFO)")
    ap.add_argument("--agent-dir", default="submission")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--steps-per-iter", type=int, default=600, help="2048-batch SGD steps per iteration")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--outcome-mode", choices=["win_only", "advantage", "none"], default="advantage")
    ap.add_argument("--lambda-start", type=float, default=1.0, help="Teacher weight lambda on Iteration 0")
    ap.add_argument("--lambda-end", type=float, default=0.15, help="Teacher weight lambda on final iteration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true", help="Resume training from existing agent weights if they exist")
    ap.add_argument("--start-iter", type=int, default=0, help="Iteration we resume from")
    args = ap.parse_args()

    py = sys.executable
    os.makedirs(args.league_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    agent_name = get_agent_name(args.agent_dir)
    log_file = os.path.join(args.log_dir, f"{agent_name}.log")

    # Prevent old shards or weights from contaminating a fresh non-resumed run
    if not args.resume:
        if os.path.exists(log_file):
            os.remove(log_file)
        for old_shard in glob.glob(os.path.join(args.buffer_dir, "*.npz")) + glob.glob(os.path.join(args.buffer_dir, "val", "*.npz")):
            os.remove(old_shard)
        weights_file = os.path.join(args.agent_dir, "weights", "model.safetensors")
        if os.path.exists(weights_file):
            os.remove(weights_file)

    print(f"=== Piping all training & generation output to: {log_file} ===", flush=True)
    # Pre-populate league snapshots from previous iterations (0 .. start_iter - 1)
    league_snapshots = [
        os.path.join(args.league_dir, f"iter_{i:03d}")
        for i in range(args.start_iter)
        if os.path.exists(os.path.join(args.league_dir, f"iter_{i:03d}"))
    ]
    for it in range(args.start_iter, args.iterations):
        n_games = args.warmup_games if it == 0 else args.games_per_iter
        frac = it / max(1, args.iterations - 1)
        # Warped fraction: frac ** 0.5 drops steeply in early iterations and flattens out later
        t_frac = frac ** args.teacher_decay_power

        if args.iterations > 1 and args.lambda_start > 0 and args.lambda_end > 0:
            lam = args.lambda_start * ((args.lambda_end / args.lambda_start) ** t_frac)
        else:
            lam = args.lambda_start

        if args.iterations > 1:
            if args.teacher_rate_start > 0 and args.teacher_rate_end > 0:
                cur_t_rate = args.teacher_rate_start * ((args.teacher_rate_end / args.teacher_rate_start) ** t_frac)
            else:
                cur_t_rate = args.teacher_rate_start + t_frac * (args.teacher_rate_end - args.teacher_rate_start)
        else:
            cur_t_rate = args.teacher_rate_start

        # Smooth learning rate schedule from args.lr (1e-3) down to 25% (2.5e-4) in late iterations
        cur_lr = args.lr * (0.25 ** frac)
        outcome_w = 0.5 if it == 0 else 1.0
        min_iter = 1 if (args.evict_warmup_after > 0 and it >= args.evict_warmup_after) else 0
        iter_keep = 0.333 if it == 0 else args.keep

        # Matchup curriculum: Iteration 0 is warmup (reference bots only).
        # If --drop-early-matchups-by is set (e.g., 30), early_share ramps from early_share_start
        # on Iteration 1 down to early_share_end on Iteration `lock_early_matchups_by`, and stays
        # at early_share_end for all remaining iterations!
        w_early, w_mid, w_late = compute_3stage_weights(it, args.stage1_end, args.stage2_end)

        # 1. Generate new shard(s) into the replay buffer
        gen_cmd = [
            py, "training/generate_buffer.py",
            "--iter", str(it),
            "--games", str(n_games),
            "--workers", str(args.workers),
            "--nodes", str(args.nodes),
            "--teacher-rate", str(cur_t_rate),
            "--keep", str(iter_keep),
            "--keep-draw", str(args.keep_draw),
            "--early-share", f"{w_early:.4f}",
            "--mid-share", f"{w_mid:.4f}",
            "--late-share", f"{w_late:.4f}",
            "--agent-dir", args.agent_dir,
            "--out", args.buffer_dir,
            "--seed", str(args.seed),
        ]
        if league_snapshots:
            # Keep all league snapshots so opponent pool includes early, mid, and recent styles
            gen_cmd.extend(["--pool-dirs"] + league_snapshots)

        # Now should be safe from reusing shards if resume the training run
        existing_shards = glob.glob(os.path.join(args.buffer_dir, f"iter_{it:03d}_shard_{args.seed}_*.npz"))
        if args.resume and len(existing_shards) >= args.workers:
            print(f"=== Reusing existing Iteration {it} shards ({len(existing_shards)} files) ===", flush=True)
        else:
            run_cmd(gen_cmd, log_file=log_file)

        # 2. Train model on FIFO buffer
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
            "--lr", f"{cur_lr:.6f}",
            "--wd", str(args.wd),
            "--outcome-mode", args.outcome_mode,
            "--outcome-weight", f"{outcome_w:.3f}",
            "--teacher-weight", f"{lam:.3f}",
            "--seed", str(args.seed + it),
            "--device", args.device,
        ]
        if args.resume or it > 0:
            train_cmd.append("--resume")
        run_cmd(train_cmd, log_file=log_file)

        # 3. Save checkpoint into the opponent league pool
        snap = snapshot_agent(args.agent_dir, args.league_dir, it)
        league_snapshots.append(snap)

        # 4. Update JSON & plots in data/plots/ after each iteration
        run_cmd([
            py, "training/plot_experiments.py",
            "--parse-log", log_file,
            "--name", agent_name,
            "--agent-name", args.agent_dir,
            "--plot-dir", args.plot_dir,
        ])

    print(f"\n=== Completed all {args.iterations} iterations! Validating {args.agent_dir}... ===")
    run_cmd([py, "-m", "fow.validate", args.agent_dir], log_file=log_file)


if __name__ == "__main__":
    main()