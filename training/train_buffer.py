"""Train a policy (and optional value) network with outcome-weighted cloning + decaying teacher loss.

    python training/train_buffer.py --buffer-dir data/replay_buffer --agent-dir submission \
        --steps 800 --batch 2048 --teacher-weight 0.5 --outcome-mode advantage
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


def load_model_class(agent_dir):
    spec = importlib.util.spec_from_file_location("agent_model", os.path.join(agent_dir, "model.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.PolicyNet


def shard_iter_num(path: str) -> int:
    m = re.search(r"iter_(\d+)_shard", os.path.basename(path))
    return int(m.group(1)) if m else 0


def load_fifo_buffer(buffer_dir, max_positions=500_000, min_iter=0):
    """Load shards >= min_iter ordered by creation/iteration and keep the most recent max_positions."""
    all_shards = sorted(glob.glob(os.path.join(buffer_dir, "*.npz")), key=os.path.getmtime)
    shards = [s for s in all_shards if shard_iter_num(s) >= min_iter]
    if not shards:
        shards = all_shards
    if not shards:
        raise FileNotFoundError(f"No .npz shards found in {buffer_dir}")

    xs, beliefs, y_playeds, y_teachers, has_teachers, zs, plies, legals = [], [], [], [], [], [], [], []
    total = 0

    for s in reversed(shards):
        d = np.load(s)
        n_s = len(d["x"])
        xs.append(d["x"])
        legals.append(d["legal"])
        if "belief" in d:
            b_arr = d["belief"]
        else:
            # Fallback for older shards without belief labels
            b_arr = np.zeros((n_s, 48), dtype=np.uint8)
        beliefs.append(b_arr)

        if "y_teacher" in d:
            y_t = d["y_teacher"].astype(np.int64)
            has_t = d["has_teacher"].astype(np.bool_)
            y_p = d["y_played"].astype(np.int64)
            z_arr = d["z"].astype(np.float32)
            ply_arr = d["ply"].astype(np.float32)
        else:
            # Backward compatibility with legacy (x, y, legal) shards
            y_t = d["y"].astype(np.int64)
            has_t = y_t >= 0
            y_p = y_t.copy()
            z_arr = np.ones(n_s, dtype=np.float32)
            ply_arr = np.ones(n_s, dtype=np.float32)

        y_teachers.append(y_t)
        has_teachers.append(has_t)
        y_playeds.append(y_p)
        zs.append(z_arr)
        plies.append(ply_arr)

        total += n_s
        if total >= max_positions:
            break

    def cat_tail(lst):
        return np.concatenate(list(reversed(lst)))[-max_positions:]

    return (
        cat_tail(xs),
        cat_tail(beliefs),
        cat_tail(y_playeds),
        cat_tail(y_teachers),
        cat_tail(has_teachers),
        cat_tail(zs),
        cat_tail(plies),
        cat_tail(legals),
        len(shards),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer-dir", default="data/replay_buffer")
    ap.add_argument("--agent-dir", default="submission")
    ap.add_argument("--max-buffer", type=int, default=500_000, help="FIFO buffer capacity in positions")
    ap.add_argument("--min-iter", type=int, default=0, help="Minimum iteration shard to include (evicts older warmup)")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--steps", type=int, default=800, help="Number of random 2048-batch SGD steps")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--teacher-weight", type=float, default=1.0, help="Lambda weight for teacher cross-entropy")
    ap.add_argument("--outcome-weight", type=float, default=1.0, help="Weight for outcome-weighted policy loss")
    ap.add_argument("--outcome-mode", choices=["win_only", "advantage", "none"], default="advantage")
    ap.add_argument("--baseline", type=float, default=0.0, help="Advantage baseline b in (z - b)")
    ap.add_argument("--entropy-bonus", type=float, default=0.005, help="Entropy regularization coefficient") # Not sure what this one does, not using it
    ap.add_argument("--value-weight", type=float, default=0.3, help="Weight for value head MSE if model returns (logits, v)")
    ap.add_argument("--belief-weight", type=float, default=0.5, help="Weight for the auxiliary spatial belief loss")
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--resume", action="store_true", help="Resume weights from agent-dir if they exist")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    x, belief, y_played, y_teacher, has_teacher, z, ply, legal, n_shards = load_fifo_buffer(
        args.buffer_dir, args.max_buffer, args.min_iter
    )
    n = len(x)
    perm = rng.permutation(n)
    n_val = max(1, int(args.val * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    win_pct = (z == 1).mean()
    loss_pct = (z == -1).mean()
    draw_pct = (z == 0).mean()
    t_pct = has_teacher.mean()
    print(
        f"Buffer: {n:,} pos ({n_shards} shards, min_iter={args.min_iter}) | "
        f"W/D/L: {win_pct:.1%}/{draw_pct:.1%}/{loss_pct:.1%} | Teacher labeled: {t_pct:.1%}"
    )

    dev = torch.device(args.device)
    in_planes = x.shape[1]
    net = load_model_class(args.agent_dir)(in_planes, args.channels, args.blocks).to(dev)

    weights_path = os.path.join(args.agent_dir, "weights", "model.safetensors")
    if args.resume and os.path.exists(weights_path):
        net.load_state_dict(load_file(weights_path), strict=False)
        print(f"Resumed weights from {weights_path}")

    n_params = sum(p.numel() for p in net.parameters())
    print(
        f"Model: {n_params:,} params | mode={args.outcome_mode}, "
        f"outcome_w={args.outcome_weight:.2f}, teacher_lambda={args.teacher_weight:.2f}"
    )

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    use_amp = (dev.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def get_batch(idx):
        xb = torch.from_numpy(x[idx]).float().to(dev, non_blocking=True)
        ypb = torch.from_numpy(y_played[idx]).to(dev, non_blocking=True)
        ytb = torch.from_numpy(y_teacher[idx]).to(dev, non_blocking=True)
        htb = torch.from_numpy(has_teacher[idx]).to(dev, non_blocking=True)
        zb = torch.from_numpy(z[idx]).to(dev, non_blocking=True)
        mb = torch.from_numpy(np.unpackbits(legal[idx], axis=1)).bool().to(dev, non_blocking=True)
        # Unpack [B, 48] -> [B, 384] -> [B, 6, 8, 8]
        bb = torch.from_numpy(np.unpackbits(belief[idx], axis=1).reshape(-1, 6, 8, 8)).float().to(
            dev, non_blocking=True
        )
        return xb, ypb, ytb, htb, zb, mb, bb

    def forward_net(xb, mb):
        out = net(xb)
        if isinstance(out, tuple):
            logits = out[0]
            v_pred = out[1].squeeze(-1) if len(out) > 1 and out[1] is not None else None
            b_logits = out[2] if len(out) > 2 else None
        else:
            logits, v_pred, b_logits = out, None, None
        masked = logits.masked_fill(~mb, -1e4 if use_amp else -1e9)
        return masked, v_pred, b_logits

    os.makedirs(os.path.join(args.agent_dir, "weights"), exist_ok=True)
    json.dump(
        {"in_planes": in_planes, "channels": args.channels, "blocks": args.blocks},
        open(os.path.join(args.agent_dir, "config.json"), "w"),
        indent=1,
    )

    best_score = -1e9
    t0 = time.time()
    run_tot, run_out, run_teach = 0.0, 0.0, 0.0

    net.train()
    for step in range(1, args.steps + 1):
        batch_indices = rng.choice(tr_idx, size=min(args.batch, len(tr_idx)), replace=False)
        xb, ypb, ytb, htb, zb, mb, bb = get_batch(batch_indices)

        with torch.amp.autocast("cuda", enabled=use_amp):
            masked_logits, v_pred, b_logits = forward_net(xb, mb)

            # 1. Teacher Cross-Entropy Loss (scaled by teacher_rate density in the batch)
            if args.teacher_weight > 0 and htb.any():
                teacher_density = htb.float().mean()  # Equals teacher_rate (N_labeled / Batch_Size)
                ce_t = teacher_density * F.cross_entropy(masked_logits[htb], ytb[htb])
            else:
                ce_t = torch.tensor(0.0, device=dev)

            # 2. Outcome-Weighted Cloning Loss on Actually Played Moves
            if args.outcome_mode == "none" or args.outcome_weight <= 0:
                loss_out = torch.tensor(0.0, device=dev)
            else:
                ce_per_sample = F.cross_entropy(masked_logits, ypb, reduction="none")
                if args.outcome_mode == "win_only":
                    weights = (zb > 0.5).float()
                else:
                    # Advantage = z - V(s) if value head exists, else z - baseline
                    baseline_val = v_pred.detach() if v_pred is not None else args.baseline
                    # Clamp to non-negative weights so we don't do unbounded gradient ascent on losses
                    weights = torch.clamp(zb - baseline_val, min=0.0)
                w_sum = weights.sum()
                loss_out = (ce_per_sample * weights).sum() / w_sum if w_sum > 0 else torch.tensor(0.0, device=dev)

            # 3. Optional Value Head Loss & Policy Entropy Bonus
            loss_val = F.mse_loss(v_pred, zb) if v_pred is not None else torch.tensor(0.0, device=dev)
            probs = F.softmax(masked_logits, dim=-1)
            log_probs = F.log_softmax(masked_logits, dim=-1).masked_fill(~mb, 0.0)
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            # 4. Auxiliary Belief Head Loss (exact square, fog-weighted, and spatial proximity)
            if b_logits is not None and args.belief_weight > 0:
                pos_w = torch.tensor(10.0, device=dev)
                loss_b_exact = F.binary_cross_entropy_with_logits(b_logits, bb, pos_weight=pos_w)

                b_probs = torch.sigmoid(b_logits)
                pool_pred = F.avg_pool2d(b_probs, kernel_size=3, stride=1, padding=1)
                pool_true = F.avg_pool2d(bb, kernel_size=3, stride=1, padding=1)
                loss_b_prox = F.mse_loss(pool_pred, pool_true)

                fog_mask = (xb[:, 12:13, :, :] == 0).float()
                loss_b_fog = (
                    F.binary_cross_entropy_with_logits(
                        b_logits, bb, pos_weight=pos_w, reduction="none"
                    )
                    * fog_mask
                ).mean()
                loss_belief = 0.5 * loss_b_exact + 0.5 * loss_b_fog + 2.0 * loss_b_prox
            else:
                loss_belief = torch.tensor(0.0, device=dev)

            loss = (
                args.outcome_weight * loss_out
                + args.teacher_weight * ce_t
                + args.value_weight * loss_val
                + args.belief_weight * loss_belief
                - args.entropy_bonus * entropy
            )

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()

        run_tot += loss.item()
        run_out += loss_out.item()
        run_teach += ce_t.item()

        if step % args.eval_every == 0 or step == args.steps:
            net.eval()
            t_correct, t_total = 0, 0
            w_correct, w_total = 0, 0
            with torch.no_grad():
                for i in range(0, n_val, 1024):
                    v_xb, v_ypb, v_ytb, v_htb, v_zb, v_mb, _ = get_batch(val_idx[i : i + 1024])
                    v_logits, _, _ = forward_net(v_xb, v_mb)
                    preds = v_logits.argmax(1)

                    if v_htb.any():
                        t_correct += (preds[v_htb] == v_ytb[v_htb]).sum().item()
                        t_total += v_htb.sum().item()

                    win_mask = v_zb > 0.5
                    if win_mask.any():
                        w_correct += (preds[win_mask] == v_ypb[win_mask]).sum().item()
                        w_total += win_mask.sum().item()

            val_t_acc = t_correct / max(1, t_total)
            val_w_acc = w_correct / max(1, w_total)
            # Composite checkpoint selection metric
            score = val_t_acc if args.outcome_mode == "none" else (0.5 * val_t_acc + 0.5 * val_w_acc)

            marker = ""
            if score >= best_score:
                best_score = score
                save_file({k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}, weights_path)
                marker = " *saved*"

            print(
                f"step {step:4d}/{args.steps}: tot={run_tot / args.eval_every:.3f} "
                f"(out={run_out / args.eval_every:.3f}, teach={run_teach / args.eval_every:.3f}) | "
                f"val_teacher={val_t_acc:.3f}, val_win_move={val_w_acc:.3f} ({time.time() - t0:.0f}s){marker}",
                flush=True,
            )
            run_tot = run_out = run_teach = 0.0
            net.train()

    print(f"Finished training -> weights saved to {weights_path}")


if __name__ == "__main__":
    main()