"""Train or update a policy network by sampling 2,048-position mini-batches from a FIFO replay buffer.

    python training/train_buffer.py --buffer-dir data/replay_buffer --agent-dir submission --steps 1000 --batch 2048
"""
import argparse
import glob
import importlib.util
import json
import os
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


def load_fifo_buffer(buffer_dir, max_positions=500_000):
    """Load shards ordered by creation/iteration and keep the most recent max_positions (FIFO)."""
    shards = sorted(glob.glob(os.path.join(buffer_dir, "*.npz")), key=os.path.getmtime)
    if not shards:
        raise FileNotFoundError(f"No .npz shards found in {buffer_dir}")

    xs, ys, legals = [], [], []
    total = 0
    # Load from newest to oldest until we hit max_positions
    for s in reversed(shards):
        d = np.load(s)
        n_s = len(d["y"])
        xs.append(d["x"])
        ys.append(d["y"].astype(np.int64))
        legals.append(d["legal"])
        total += n_s
        if total >= max_positions:
            break

    x = np.concatenate(list(reversed(xs)))[-max_positions:]
    y = np.concatenate(list(reversed(ys)))[-max_positions:]
    legal = np.concatenate(list(reversed(legals)))[-max_positions:]
    return x, y, legal, len(shards)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer-dir", default="data/replay_buffer")
    ap.add_argument("--agent-dir", default="submission")
    ap.add_argument("--max-buffer", type=int, default=500_000, help="FIFO buffer capacity in positions")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--steps", type=int, default=1000, help="Number of random 2048-batch SGD steps")
    ap.add_argument("--eval-every", type=int, default=100, help="Evaluate validation accuracy every N steps")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--resume", action="store_true", help="Resume weights from agent-dir if they exist")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    x, y, legal, n_shards = load_fifo_buffer(args.buffer_dir, args.max_buffer)
    n = len(y)
    perm = rng.permutation(n)
    n_val = max(1, int(args.val * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    print(f"Buffer: {n:,} positions (from {n_shards} shards); {len(tr_idx):,} train / {n_val:,} val")

    dev = torch.device(args.device)
    in_planes = x.shape[1]
    net = load_model_class(args.agent_dir)(in_planes, args.channels, args.blocks).to(dev)

    weights_path = os.path.join(args.agent_dir, "weights", "model.safetensors")
    if args.resume and os.path.exists(weights_path):
        net.load_state_dict(load_file(weights_path))
        print(f"Resumed weights from {weights_path}")

    n_params = sum(p.numel() for p in net.parameters())
    print(f"Model: {n_params:,} trainable parameters")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)

    def get_batch(idx):
        xb = torch.from_numpy(x[idx]).float().to(dev)
        yb = torch.from_numpy(y[idx]).to(dev)
        mb = torch.from_numpy(np.unpackbits(legal[idx], axis=1)).bool().to(dev)
        return xb, yb, mb

    def masked_logits(xb, mb):
        return net(xb).masked_fill(~mb, -1e9)

    os.makedirs(os.path.join(args.agent_dir, "weights"), exist_ok=True)
    json.dump({"in_planes": in_planes, "channels": args.channels, "blocks": args.blocks},
              open(os.path.join(args.agent_dir, "config.json"), "w"), indent=1)

    best_val_acc = -1.0
    t0 = time.time()
    running_loss = 0.0

    net.train()
    for step in range(1, args.steps + 1):
        # Uniformly sample 2,048 random positions from the replay buffer
        batch_indices = rng.choice(tr_idx, size=min(args.batch, len(tr_idx)), replace=False)
        xb, yb, mb = get_batch(batch_indices)

        loss = F.cross_entropy(masked_logits(xb, mb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        running_loss += loss.item()

        if step % args.eval_every == 0 or step == args.steps:
            net.eval()
            correct = 0
            with torch.no_grad():
                for i in range(0, n_val, 1024):
                    v_xb, v_yb, v_mb = get_batch(val_idx[i:i + 1024])
                    correct += (masked_logits(v_xb, v_mb).argmax(1) == v_yb).sum().item()
            val_acc = correct / n_val
            avg_loss = running_loss / args.eval_every
            running_loss = 0.0

            marker = ""
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_file({k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}, weights_path)
                marker = " *saved best*"

            print(f"step {step:4d}/{args.steps}: train loss {avg_loss:.3f}, "
                  f"val teacher match {val_acc:.3f} ({time.time() - t0:.0f}s){marker}", flush=True)
            net.train()

    print(f"Best validation match: {best_val_acc:.3f} -> saved to {weights_path}")


if __name__ == "__main__":
    main()