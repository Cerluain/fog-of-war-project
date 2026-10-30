#!/usr/bin/env bash
# run_guess_sf_sweep.sh
# Finds the exact Pareto-optimal (K, N) configuration
set -euo pipefail

GAMES_PER_PAIR=${1:-150}
WORKERS=${2:-8}

CANDIDATES=(
    "bot:greedy"
    "bot:guess-sf-45x500"   # Baseline reference (1242 Elo)
    "bot:guess-sf-40x500"   
    "bot:guess-sf-35x500"   
    "bot:guess-sf-30x500"   # 15.0k nodes (consensus emphasis)
)

mkdir -p experiments/arena_evals

echo "=================================================================="
echo " Running (K x N) Arena Tournament (${GAMES_PER_PAIR} games/pair, ${WORKERS} workers)"
echo "=================================================================="

python -m fow.arena "${CANDIDATES[@]}" \
    --mode round-robin \
    --games "$GAMES_PER_PAIR" \
    --workers "$WORKERS" \
    --out experiments/arena_evals/guess_sf_sweep.json \
    | tee experiments/arena_evals/guess_sf_sweep.log

echo "Tournament complete. Results saved to experiments/arena_evals/guess_sf_sweep.json"