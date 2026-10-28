#!/usr/bin/env bash
# Controlled ablation sweep for teacher-weight decay schedules (lambda_start -> lambda_end).
# Saves raw logs to experiments/logs/, structured data to experiments/json/, and plots to experiments/plots/.
set -euo pipefail

# Default sweep parameters (adjustable via environment variables)
ITERATIONS=${ITERATIONS:-4}
WARMUP_GAMES=${WARMUP_GAMES:-1600}
GAMES_PER_ITER=${GAMES_PER_ITER:-1200}
STEPS_PER_ITER=${STEPS_PER_ITER:-600}
ARENA_GAMES=${ARENA_GAMES:-150}
WORKERS=${WORKERS:-4}
BASE_AGENT=${BASE_AGENT:-"agents/v1_se_model"}

mkdir -p experiments/logs experiments/json experiments/plots data/shared_warmup_iter0

echo "=================================================================="
echo " STEP 0: Generating Shared Warmup Iteration 0 (${WARMUP_GAMES} games)"
echo "=================================================================="
if [ -z "$(ls -A data/shared_warmup_iter0/*.npz 2>/dev/null || true)" ]; then
    python training/generate_buffer.py --iter 0 --games "$WARMUP_GAMES" --workers "$WORKERS" --nodes 4000 --teacher-rate 0.5 --keep 1.0 --agent-dir "$BASE_AGENT" --out data/shared_warmup_iter0 --seed 42
else
    echo "Shared Iteration 0 shards already exist in data/shared_warmup_iter0/ -- reusing for controlled comparison!"
fi

# Define the (lambda_start, lambda_end, label) schedules to test
SCHEDULES=(
    "1.0 1.0 lam_1.0_to_1.0_fixed"
    "1.0 0.3 lam_1.0_to_0.3_moderate"
    "1.0 0.05 lam_1.0_to_0.05_steep"
    "0.5 0.1 lam_0.5_to_0.1_lowteacher"
)

TRAINED_AGENTS=()

for entry in "${SCHEDULES[@]}"; do
    read -r L_START L_END EXP_NAME <<< "$entry"
    EXP_AGENT="agents/exp_${EXP_NAME}"
    EXP_BUF="data/buf_${EXP_NAME}"
    EXP_LEAGUE="data/league_${EXP_NAME}"
    LOG_FILE="experiments/logs/${EXP_NAME}.log"

    echo ""
    echo "=================================================================="
    echo " RUNNING EXPERIMENT: ${EXP_NAME} (lambda: ${L_START} -> ${L_END})"
    echo "=================================================================="

    # 1. Create clean agent directory cloned from BASE_AGENT (without pretrained weights)
    rm -rf "$EXP_AGENT" "$EXP_BUF" "$EXP_LEAGUE"
    mkdir -p "$EXP_AGENT/weights" "$EXP_BUF" "$EXP_LEAGUE"
    cp "$BASE_AGENT/agent.py" "$BASE_AGENT/features.py" "$BASE_AGENT/model.py" "$EXP_AGENT/"
    if [ -f "$BASE_AGENT/MODEL.md" ]; then cp "$BASE_AGENT/MODEL.md" "$EXP_AGENT/"; fi

    # 2. Seed buffer with the identical shared Iteration 0 warmup shards
    cp data/shared_warmup_iter0/*.npz "$EXP_BUF/"

    # 3. Train Iteration 0 on shared warmup, then run Iterations 1..(ITERATIONS-1)
    : > "$LOG_FILE"
    for (( it=0; it<ITERATIONS; it++ )); do
        # Compute annealed lambda for iteration it
        LAM=$(python -c "iters=$ITERATIONS; it=$it; s=$L_START; e=$L_END; print(f'{s + (it / max(1, iters - 1)) * (e - s):.3f}')")
        OUT_W=$(python -c "print('0.500' if $it == 0 else '1.000')")
        MIN_IT=$(python -c "print(1 if $it >= 2 else 0)")

        if [ "$it" -gt 0 ]; then
            POOL_ARGS=()
            if [ -n "$(ls -d "$EXP_LEAGUE"/iter_* 2>/dev/null || true)" ]; then
                POOL_ARGS=(--pool-dirs "$EXP_LEAGUE"/iter_*)
            fi
            python training/generate_buffer.py --iter "$it" --games "$GAMES_PER_ITER" --workers "$WORKERS" --nodes 4000 --teacher-rate 0.4 --keep 1.0 --agent-dir "$EXP_AGENT" --out "$EXP_BUF" --seed 42 "${POOL_ARGS[@]}" >> "$LOG_FILE" 2>&1
        fi

        RESUME_FLAG=()
        if [ "$it" -gt 0 ]; then RESUME_FLAG=(--resume); fi

        python training/train_buffer.py --buffer-dir "$EXP_BUF" --agent-dir "$EXP_AGENT" --max-buffer 500000 --min-iter "$MIN_IT" --channels 128 --blocks 6 --steps "$STEPS_PER_ITER" --batch 2048 --outcome-mode advantage --outcome-weight "$OUT_W" --teacher-weight "$LAM" --seed "$((42 + it))" "${RESUME_FLAG[@]}" >> "$LOG_FILE" 2>&1

        # Snapshot into league
        cp -r "$EXP_AGENT" "$EXP_LEAGUE/iter_$(printf '%03d' $it)"
    done

    # 4. Validate and benchmark in Arena against fixed anchors
    python -m fow.validate "$EXP_AGENT" >> "$LOG_FILE" 2>&1
    python -m fow.arena "$EXP_AGENT" agents/baseline bot:greedy bot:guess-sf-4x500 --games "$ARENA_GAMES" --mode round-robin >> "$LOG_FILE" 2>&1

    # 5. Parse log into persistent JSON
    python training/plot_experiments.py --parse-log "$LOG_FILE" --name "$EXP_NAME" --agent-name "$EXP_AGENT"
    TRAINED_AGENTS+=("$EXP_AGENT")
done

echo ""
echo "=================================================================="
echo " GENERATING MATPLOTLIB COMPARISON DASHBOARD"
echo "=================================================================="
python training/plot_experiments.py --plot experiments/json/lam_*.json --out experiments/plots/lambda_sweep_summary.png

echo "Done! Open experiments/plots/lambda_sweep_summary.png to view the results."