"""Parse training + arena logs into persistent JSON files and plot multi-experiment comparisons.

Usage:
  1. Parse a finished log into JSON:
     python training/plot_experiments.py --parse-log experiments/logs/lam_1.0_0.2.log --name "lam_1.0_to_0.2" --agent-name "agents/exp_lam_1.0_0.2"

  2. Plot a comparison across multiple saved JSON files:
     python training/plot_experiments.py --plot experiments/json/*.json --out experiments/plots/lambda_sweep.png
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
        "iterations": [],
        "steps": [],
        "tot_loss": [],
        "out_loss": [],
        "teach_loss": [],
        "val_teacher": [],
        "val_win_move": [],
        "sp_decisive_rate": [],
        "lambda_schedule": [],
        "elo": None,
        "elo_ci_low": None,
        "elo_ci_high": None,
        "win_rates": {},
    }

    # 1. Extract teacher-weight lambda per iteration
    for m in re.finditer(r"--teacher-weight\s+([0-9.]+)", text):
        data["lambda_schedule"].append(float(m.group(1)))

    # 2. Extract Self-Play decisive rate per iteration
    for m in re.finditer(r"Self-Play decisive rate:\s+([0-9.]+)%", text):
        data["sp_decisive_rate"].append(float(m.group(1)))

    # 3. Extract step metrics
    # Format: step  100/650: tot=2.573 (out=1.466, teach=1.858) | val_teacher=0.426, val_win_move=0.514
    step_pattern = re.compile(
        r"step\s+(\d+)/(\d+):\s+tot=([0-9.]+)\s+\(out=([0-9.]+),\s+teach=([0-9.]+)\)\s+\|\s+"
        r"val_teacher=([0-9.]+),\s+val_win_move=([0-9.]+)"
    )
    cum_step = 0
    last_raw_step = 0
    iter_offset = 0

    for m in step_pattern.finditer(text):
        s, max_s = int(m.group(1)), int(m.group(2))
        if s < last_raw_step:
            iter_offset += max_s
        last_raw_step = s
        cum_step = iter_offset + s

        data["steps"].append(cum_step)
        data["tot_loss"].append(float(m.group(3)))
        data["out_loss"].append(float(m.group(4)))
        data["teach_loss"].append(float(m.group(5)))
        data["val_teacher"].append(float(m.group(6)))
        data["val_win_move"].append(float(m.group(7)))

    # 4. Extract Arena Elo Rating & 95% CI for this agent
    # Format:   agents/exp_lam_1.0_0.2    1207   [ 1184,  1233]
    escaped_agent = re.escape(agent_name)
    elo_match = re.search(
        rf"^\s*{escaped_agent}\s+(\d+)\s+\[\s*(\d+),\s*(\d+)\]", text, re.MULTILINE
    )
    if elo_match:
        data["elo"] = int(elo_match.group(1))
        data["elo_ci_low"] = int(elo_match.group(2))
        data["elo_ci_high"] = int(elo_match.group(3))

    # 5. Extract head-to-head match scores from Arena
    # Format:   player_a   vs player_b    134.0 / 200
    h2h_pattern = re.compile(r"^\s*(\S+)\s+vs\s+(\S+)\s+([0-9.]+)\s*/\s*(\d+)", re.MULTILINE)
    for m in h2h_pattern.finditer(text):
        p1, p2, score, total = m.group(1), m.group(2), float(m.group(3)), int(m.group(4))
        if p1 == agent_name:
            data["win_rates"][p2] = round(100.0 * (score / total), 1)
        elif p2 == agent_name:
            data["win_rates"][p1] = round(100.0 * ((total - score) / total), 1)

    return data


def plot_comparisons(json_files: list, out_path: str):
    runs = []
    for jf in sorted(json_files):
        with open(jf, "r", encoding="utf-8") as f:
            runs.append(json.load(f))

    if not runs:
        print("No valid JSON experiment files found to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Fog-of-War Chess: Lambda Decay & Outcome Learning Ablation Suite", fontsize=15, fontweight="bold")
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs), 3)))

    # Panel 1: Validation Win-Move Match vs Teacher Match over Training Steps
    ax1 = axes[0, 0]
    for idx, r in enumerate(runs):
        if r["steps"]:
            ax1.plot(r["steps"], r["val_win_move"], color=colors[idx], lw=2, label=f"{r['name']} (Win-Move)")
            ax1.plot(r["steps"], r["val_teacher"], color=colors[idx], lw=1.5, ls="--", alpha=0.75, label=f"{r['name']} (Teacher)")
    ax1.set_title("Validation Alignment: Win-Move (solid) vs. Teacher (dashed)")
    ax1.set_xlabel("Cumulative Training Steps (Batch = 2048)")
    ax1.set_ylabel("Top-1 Validation Match")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, ncol=2)

    # Panel 2: Outcome Loss & Teacher Loss Curves
    ax2 = axes[0, 1]
    for idx, r in enumerate(runs):
        if r["steps"]:
            ax2.plot(r["steps"], r["out_loss"], color=colors[idx], lw=2, label=f"{r['name']} (Outcome Loss)")
            ax2.plot(r["steps"], r["teach_loss"], color=colors[idx], lw=1.2, ls=":", alpha=0.7, label=f"{r['name']} (Teacher CE)")
    ax2.set_title("Training Loss Decomposition (Outcome vs. Teacher)")
    ax2.set_xlabel("Cumulative Training Steps")
    ax2.set_ylabel("Cross-Entropy Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, ncol=2)

    # Panel 3: Lambda Annealing Schedule & Self-Play Decisive Rate
    ax3 = axes[1, 0]
    for idx, r in enumerate(runs):
        if r["lambda_schedule"]:
            iters = list(range(len(r["lambda_schedule"])))
            ax3.plot(iters, r["lambda_schedule"], marker="o", color=colors[idx], lw=2, label=f"{r['name']} λ")
    ax3.set_title("Teacher Weight (λ) Decay Schedule Across Iterations")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Teacher Weight λ")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)

    # Panel 4: Arena Elo Rating with 95% Confidence Intervals
    ax4 = axes[1, 1]
    elo_runs = [r for r in runs if r.get("elo") is not None]
    if elo_runs:
        names = [r["name"] for r in elo_runs]
        elos = [r["elo"] for r in elo_runs]
        err_low = [r["elo"] - r.get("elo_ci_low", r["elo"]) for r in elo_runs]
        err_high = [r.get("elo_ci_high", r["elo"]) - r["elo"] for r in elo_runs]

        x_pos = np.arange(len(names))
        bars = ax4.bar(x_pos, elos, yerr=[err_low, err_high], capsize=6, color=colors[:len(names)], alpha=0.85, edgecolor="black")
        ax4.axhline(1154, color="red", ls="--", lw=1.5, label="Baseline Agent (~1154)")
        ax4.axhline(1060, color="gray", ls=":", lw=1.2, label="guess-sf-4x500 (1060)")
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(names, rotation=15, ha="right")
        ax4.set_ylim(min(950, min(elos) - 60), max(elos) + 60)
        ax4.set_ylabel("Arena Elo Rating (with 95% CI)")
        ax4.set_title("Final Tournament Elo Comparison")
        ax4.grid(True, axis="y", alpha=0.3)
        ax4.legend(fontsize=8)

        for bar, r in zip(bars, elo_runs):
            yval = bar.get_height()
            wr_sf = r.get("win_rates", {}).get("bot:guess-sf-4x500", "--")
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 8,
                f"{yval}\n(vs SF: {wr_sf}%)",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
    else:
        ax4.text(0.5, 0.5, "No Arena Elo data parsed yet", ha="center", va="center")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved comparison dashboard to: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse-log", help="Path to a raw .log file to parse into JSON")
    ap.add_argument("--name", default="exp", help="Short display name for the experiment")
    ap.add_argument("--agent-name", default="submission", help="Agent path as printed in fow.arena")
    ap.add_argument("--json-out", default=None, help="Output JSON path (defaults to experiments/json/<name>.json)")
    ap.add_argument("--plot", nargs="*", help="List or glob of .json files to compare and plot")
    ap.add_argument("--out", default="experiments/plots/lambda_sweep_summary.png", help="Output plot image path")
    args = ap.parse_args()

    if args.parse_log:
        os.makedirs("experiments/json", exist_ok=True)
        j_path = args.json_out or os.path.join("experiments/json", f"{args.name}.json")
        parsed = parse_log_file(args.parse_log, args.name, args.agent_name)
        with open(j_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2)
        print(f"Parsed {args.parse_log} -> {j_path} (Elo: {parsed['elo']}, Win-Rates: {parsed['win_rates']})")

    if args.plot:
        expanded = []
        for pattern in args.plot:
            expanded.extend(glob.glob(pattern))
        plot_comparisons(expanded if expanded else args.plot, args.out)


if __name__ == "__main__":
    main()