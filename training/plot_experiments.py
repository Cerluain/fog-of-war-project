"""Parse training + generation logs in data/logs/ and plot bot matchups & validation metrics to data/plots/.

Usage:
  1. Single agent parse + plot (automatically called by run_iterations.py):
     python training/plot_experiments.py --parse-log data/logs/v3_se_model.log --name v3_se_model --agent-name agents/v3_se_model --plot-dir data/plots

  2. Compare multiple agents from saved JSONs:
     python training/plot_experiments.py --plot data/logs/*.json --plot-dir data/plots --out-prefix comparison
"""
import argparse
import glob
import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_path: str, exp_name: str, agent_name: str) -> dict:
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    data = {
        "name": exp_name,
        "agent_name": agent_name,
        "log_path": log_path,
        # Continuous step metrics
        "steps": [],
        "tot_loss": [],
        "out_loss": [],
        "teach_loss": [],
        "val_loss": [],
        "belief_loss": [],
        "val_teacher": [],
        "val_win_move": [],
        "val_out_loss": [],
        "val_teach_loss": [],
        "val_val_loss": [],
        "val_belief_loss": [],
        # Per-iteration max/best validation summaries (indexed by training iteration i)
        "iter_Indices": [],
        "iter_max_val_win_move": [],
        "iter_max_val_teacher": [],
        "iter_max_composite": [],
        "iter_min_out_loss": [],
        "iter_min_teach_loss": [],
        "iter_min_val_loss": [],
        "iter_min_belief_loss": [],
        # Bot performance of Iteration i (recorded during generation of Iteration i+1)
        "eval_iter_indices": [],
        "sp_decisive_rate": [],
        "bot_matchups": {},  # {opponent: {"iters": [i, ...], "win_rates": [wr, ...], "games": [g, ...]}}
        # Final fow.arena ratings
        "elo": None,
        "elo_ci_low": None,
        "elo_ci_high": None,
        "arena_win_rates": {},
    }

    # Split log by generation / training blocks to accurately associate Iteration i
    gen_blocks = re.split(r"(?==== Generating Iteration \d+:)", text)
    for block in gen_blocks:
        m_iter = re.search(r"=== Generating Iteration (\d+):", block)
        if not m_iter:
            continue
        gen_it = int(m_iter.group(1))
        # Generation at iter (i + 1) tests the model trained in Iteration i!
        if gen_it >= 1:
            trained_model_iter = gen_it - 1
            m_sp = re.search(r"Self-Play decisive rate:\s+([0-9.]+)%", block)
            if m_sp:
                data["eval_iter_indices"].append(trained_model_iter)
                data["sp_decisive_rate"].append(float(m_sp.group(1)))

            # Extract: vs bot:greedy: 226W-0D-52L (81.3% won)
            for m_vs in re.finditer(
                r"vs\s+(\S+):\s+(\d+)W-(\d+)D-(\d+)L\s+\(([0-9.]+)%\s+won\)", block
            ):
                opp = m_vs.group(1)
                w, d, l = int(m_vs.group(2)), int(m_vs.group(3)), int(m_vs.group(4))
                wr = float(m_vs.group(5))
                entry = data["bot_matchups"].setdefault(
                    opp, {"iters": [], "win_rates": [], "games": []}
                )
                entry["iters"].append(trained_model_iter)
                entry["win_rates"].append(wr)
                entry["games"].append(w + d + l)

    # Parse training steps and group by iteration
    # Supports both legacy logs and new logs with value/belief/val_losses
    step_pattern = re.compile(
        r"step\s+(\d+)/(\d+):\s+tot=([0-9.]+)\s+"
        r"\(out=([0-9.]+),\s*teach=([0-9.]+)"
        r"(?:,\s*value=([0-9.]+),\s*belief=([0-9.]+))?\)\s+\|\s+"
        r"val_teacher=([0-9.]+),\s*val_win_move=([0-9.]+)"
        r"(?:,\s*val_out=([0-9.]+),\s*val_teach=([0-9.]+),\s*val_val=([0-9.]+),\s*val_belief=([0-9.]+))?"
    )

    last_raw_step = 0
    iter_offset = 0
    cur_train_iter = 0
    cur_iter_steps = []

    def flush_iter_summary(it_idx, step_records):
        if not step_records:
            return
        data["iter_Indices"].append(it_idx)
        win_moves = [r["val_win"] for r in step_records]
        teachers = [r["val_teach"] for r in step_records]
        composites = [0.5 * w + 0.5 * t for w, t in zip(win_moves, teachers)]
        data["iter_max_val_win_move"].append(max(win_moves))
        data["iter_max_val_teacher"].append(max(teachers))
        data["iter_max_composite"].append(max(composites))
        data["iter_min_out_loss"].append(min(r["out_l"] for r in step_records))
        data["iter_min_teach_loss"].append(min(r["teach_l"] for r in step_records))
        data["iter_min_val_loss"].append(min(r["val_l"] for r in step_records))
        data["iter_min_belief_loss"].append(min(r["bel_l"] for r in step_records))

    for m in step_pattern.finditer(text):
        s, max_s = int(m.group(1)), int(m.group(2))
        if s < last_raw_step:
            flush_iter_summary(cur_train_iter, cur_iter_steps)
            cur_iter_steps = []
            cur_train_iter += 1
            iter_offset += max_s
        last_raw_step = s
        cum_step = iter_offset + s

        tot_l = float(m.group(3))
        out_l = float(m.group(4))
        teach_l = float(m.group(5))
        val_l = float(m.group(6)) if m.group(6) is not None else 0.0
        bel_l = float(m.group(7)) if m.group(7) is not None else 0.0
        v_teach_acc = float(m.group(8))
        v_win_acc = float(m.group(9))

        # Use validation losses if present; otherwise fall back to step losses
        v_out_l = float(m.group(10)) if m.group(10) is not None else out_l
        v_teach_l = float(m.group(11)) if m.group(11) is not None else teach_l
        v_val_l = float(m.group(12)) if m.group(12) is not None else val_l
        v_bel_l = float(m.group(13)) if m.group(13) is not None else bel_l

        data["steps"].append(cum_step)
        data["tot_loss"].append(tot_l)
        data["out_loss"].append(out_l)
        data["teach_loss"].append(teach_l)
        data["val_loss"].append(val_l)
        data["belief_loss"].append(bel_l)
        data["val_teacher"].append(v_teach_acc)
        data["val_win_move"].append(v_win_acc)
        data["val_out_loss"].append(v_out_l)
        data["val_teach_loss"].append(v_teach_l)
        data["val_val_loss"].append(v_val_l)
        data["val_belief_loss"].append(v_bel_l)

        cur_iter_steps.append({
            "val_win": v_win_acc,
            "val_teach": v_teach_acc,
            "out_l": v_out_l,
            "teach_l": v_teach_l,
            "val_l": v_val_l,
            "bel_l": v_bel_l,
        })

    flush_iter_summary(cur_train_iter, cur_iter_steps)

    # Extract Final Arena Elo Rating & head-to-head scores if present
    clean_agent = agent_name.rstrip("/\\")
    for candidate in (agent_name, clean_agent, exp_name):
        escaped = re.escape(candidate)
        elo_match = re.search(rf"^\s*{escaped}\s+(\d+)\s+\[\s*(\d+),\s*(\d+)\]", text, re.MULTILINE)
        if elo_match:
            data["elo"] = int(elo_match.group(1))
            data["elo_ci_low"] = int(elo_match.group(2))
            data["elo_ci_high"] = int(elo_match.group(3))
            break

    h2h_pattern = re.compile(r"^\s*(\S+)\s+vs\s+(\S+)\s+([0-9.]+)\s*/\s*(\d+)", re.MULTILINE)
    for m in h2h_pattern.finditer(text):
        p1, p2, score, total = m.group(1), m.group(2), float(m.group(3)), int(m.group(4))
        if p1 in (agent_name, clean_agent, exp_name):
            data["arena_win_rates"][p2] = round(100.0 * (score / total), 1)
        elif p2 in (agent_name, clean_agent, exp_name):
            data["arena_win_rates"][p1] = round(100.0 * ((total - score) / total), 1)

    return data


def plot_arena_and_bots(runs: list, out_path: str):
    """Plot how Iteration i performs against various bots during Iteration i+1 data generation."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Agent Performance Across Iterations (Evaluated During Next-Iteration Generation)", fontsize=14, fontweight="bold")

    # Panel 1: Win Rate vs Bot Types after each Iteration i
    ax1 = axes[0]
    bot_styles = {
        "bot:greedy": ("#2ca02c", "o", "-", "vs bot:greedy"),
        "self_guarded": ("#d62728", "s", "-", "vs self_guarded (Omniscient Guard)"),
        "league": ("#1f77b4", "^", "--", "vs League Pool (Past Checkpoints)"),
        "self": ("#9467bd", "d", ":", "vs self (White Win %)"),
    }

    for r in runs:
        prefix = f"[{r['name']}] " if len(runs) > 1 else ""
        for opp, stats in sorted(r.get("bot_matchups", {}).items()):
            color, marker, ls, label = bot_styles.get(opp, (None, "o", "-", f"vs {opp}"))
            ax1.plot(
                stats["iters"],
                stats["win_rates"],
                marker=marker,
                ls=ls,
                lw=2,
                color=color if len(runs) == 1 else None,
                label=f"{prefix}{label}",
            )

    ax1.axhline(50.0, color="gray", ls="--", lw=1, alpha=0.6, label="50% Parity")
    ax1.set_title("Iteration i Win Rate vs. Opponents (Measured in Iter i+1 Gen)")
    ax1.set_xlabel("Trained Model Iteration (i)")
    ax1.set_ylabel("Win Rate (%)")
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="lower right")

    # Panel 2: Self-Play Decisive Rate & Final Arena Win-Rates (if run)
    ax2 = axes[1]
    for r in runs:
        if r.get("eval_iter_indices") and r.get("sp_decisive_rate"):
            n = min(len(r["eval_iter_indices"]), len(r["sp_decisive_rate"]))
            ax2.plot(
                r["eval_iter_indices"][:n],
                r["sp_decisive_rate"][:n],
                marker="o",
                lw=2,
                label=f"{r['name']} Self-Play Decisive %",
            )
    ax2.axhline(50.0, color="red", ls=":", lw=1.2, label="50% Warning Threshold")
    ax2.set_title("Self-Play Decisive Game Rate Across Iterations")
    ax2.set_xlabel("Trained Model Iteration (i)")
    ax2.set_ylabel("Decisive Rate (%)")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc="lower left")

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved bot performance plot to: {out_path}")


def plot_validation_metrics(runs: list, out_path: str):
    """Plot Max Validation Scores per Iteration + Continuous Outcome, Teacher, Value, and Belief curves."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Validation Metrics: Max Iteration Scores, Outcome/Policy, Teacher, Value & Belief", fontsize=14, fontweight="bold")
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs), 3)))

    # Panel 1: Max Validation Accuracy per Iteration (Outcome Win-Move & Teacher Match)
    ax1 = axes[0, 0]
    for idx, r in enumerate(runs):
        iters = r.get("iter_Indices", [])
        if iters:
            c = colors[idx]
            ax1.plot(iters, r["iter_max_val_win_move"], marker="o", lw=2.2, color=c, label=f"{r['name']} Max Val Win-Move (Outcome)")
            ax1.plot(iters, r["iter_max_val_teacher"], marker="s", lw=1.8, ls="--", color=c, alpha=0.8, label=f"{r['name']} Max Val Teacher")
            ax1.plot(iters, r["iter_max_composite"], marker="^", lw=1.5, ls=":", color=c, alpha=0.7, label=f"{r['name']} Max Composite Score")
    ax1.set_title("Peak Validation Accuracy per Iteration (Outcome vs. Teacher)")
    ax1.set_xlabel("Training Iteration (i)")
    ax1.set_ylabel("Max Top-1 Validation Match")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # Panel 2: Continuous Validation Win-Move & Teacher Alignment over Steps
    ax2 = axes[0, 1]
    for idx, r in enumerate(runs):
        if r["steps"]:
            c = colors[idx]
            ax2.plot(r["steps"], r["val_win_move"], lw=2, color=c, label=f"{r['name']} Val Win-Move (Policy/Outcome)")
            ax2.plot(r["steps"], r["val_teacher"], lw=1.5, ls="--", color=c, alpha=0.75, label=f"{r['name']} Val Teacher")
    ax2.set_title("Continuous Validation Accuracy Across Training Steps")
    ax2.set_xlabel("Cumulative SGD Steps")
    ax2.set_ylabel("Top-1 Match Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    # Panel 3: Outcome & Teacher Validation Losses
    ax3 = axes[1, 0]
    for idx, r in enumerate(runs):
        if r["steps"]:
            c = colors[idx]
            ax3.plot(r["steps"], r["val_out_loss"], lw=2, color=c, label=f"{r['name']} Outcome/Policy Loss")
            ax3.plot(r["steps"], r["val_teach_loss"], lw=1.5, ls="--", color=c, alpha=0.75, label=f"{r['name']} Teacher CE Loss")
    ax3.set_title("Outcome/Policy & Teacher Losses Over Steps")
    ax3.set_xlabel("Cumulative SGD Steps")
    ax3.set_ylabel("Loss")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    # Panel 4: Value & Belief Losses (Continuous + Min per Iteration)
    ax4 = axes[1, 1]
    for idx, r in enumerate(runs):
        if r["steps"]:
            c = colors[idx]
            ax4.plot(r["steps"], r["val_val_loss"], lw=2, color=c, label=f"{r['name']} Value MSE Loss")
            ax4.plot(r["steps"], r["val_belief_loss"], lw=1.8, ls=":", color=c, alpha=0.85, label=f"{r['name']} Belief Spatial Loss")
    ax4.set_title("Value Head (MSE) & Auxiliary Belief Losses")
    ax4.set_xlabel("Cumulative SGD Steps")
    ax4.set_ylabel("Loss")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved validation metrics plot to: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse-log", help="Path to a raw .log file in data/logs/")
    ap.add_argument("--name", default="agent", help="Short display name for the agent")
    ap.add_argument("--agent-name", default="submission", help="Agent directory path")
    ap.add_argument("--plot-dir", default="data/plots", help="Directory to save generated plots")
    ap.add_argument("--plot", nargs="*", help="Optional list/glob of .json files to compare")
    ap.add_argument("--out-prefix", default=None, help="Filename prefix for plots")
    args = ap.parse_args()

    os.makedirs("data/logs", exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    runs_to_plot = []

    if args.parse_log:
        j_path = os.path.join("data/logs", f"{args.name}.json")
        parsed = parse_log_file(args.parse_log, args.name, args.agent_name)
        with open(j_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2)
        runs_to_plot.append(parsed)
        print(f"Parsed {args.parse_log} -> {j_path}")

    if args.plot:
        expanded = []
        for pattern in args.plot:
            expanded.extend(glob.glob(pattern))
        for jf in sorted(expanded if expanded else args.plot):
            with open(jf, "r", encoding="utf-8") as f:
                runs_to_plot.append(json.load(f))

    if runs_to_plot:
        prefix = args.out_prefix or (args.name if len(runs_to_plot) == 1 else "comparison")
        plot_arena_and_bots(runs_to_plot, os.path.join(args.plot_dir, f"{prefix}_arena_bots.png"))
        plot_validation_metrics(runs_to_plot, os.path.join(args.plot_dir, f"{prefix}_validation.png"))


if __name__ == "__main__":
    main()