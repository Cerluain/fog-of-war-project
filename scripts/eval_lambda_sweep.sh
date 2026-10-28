#!/usr/bin/env bash
# Extended ablation sweep: exponential teacher-weight decay schedules.
set -euo pipefail

ITERATIONS=${ITERATIONS:-4}
WARMUP_GAMES=${WARMUP_GAMES:-1600}
GAMES_PER_ITER=${GAMES_PER_ITER:-1200}
STEPS_PER_ITER=${STEPS_PER_ITER:-600}
ARENA_GAMES=${ARENA_GAMES:-150}
WORKERS=${WORKERS:-4}
BASE_AGENT=${BASE_AGENT:-"agents/v1_se_model"}
TEACHER_RATE_START=${TEACHER_RATE_START:-0.9}
TEACHER_RATE_END=${TEACHER_RATE_END:-0.01}

mkdir -p experiments/logs experiments/json experiments/plots data/shared_warmup_iter0

echo "=================================================================="
echo " STEP 0: Checking Shared Warmup Iteration 0 (${WARMUP_GAMES} games)"
echo "=================================================================="
if [ -z "$(ls -A data/shared_warmup_iter0/*.npz 2>/dev/null || true)" ]; then
    python training/generate_buffer.py --iter 0 --games "$WARMUP_GAMES" --workers "$WORKERS" --nodes 4000 --teacher-rate "$TEACHER_RATE_START" --keep 1.0 --agent-dir "$BASE_AGENT" --out data/shared_warmup_iter0 --seed 42
else
    echo "Shared Iteration 0 shards already exist in data/shared_warmup_iter0/ -- reusing!"
fi

# Format: "L_START L_END OUT_MODE EXP_NAME"
SCHEDULES=(
    "1.0 0.1 advantage lam_1.0_to_0.1"
    "1.0 0.05 advantage lam_1.0_to_0.05"
    "1.0 0.01 advantage lam_1.0_to_0.01"
    "0.5 0.05 advantage lam_0.5_to_0.05"
    "1.2 0.05 advantage lam_1.2_to_0.05"
    "1.6 0.05 advantage lam_1.6_to_0.05"
)

for entry in "${SCHEDULES[@]}"; do
    read -r L_START L_END OUT_MODE EXP_NAME <<< "$entry"
    EXP_TAG="B_${EXP_NAME}"
    EXP_AGENT="agents/exp_${EXP_TAG}"
    EXP_BUF="data/buf_${EXP_TAG}"
    EXP_LEAGUE="data/league_${EXP_TAG}"
    LOG_FILE="experiments/logs/${EXP_TAG}.log"
    JSON_FILE="experiments/json/${EXP_TAG}.json"

    echo ""
    echo "=================================================================="
    echo " EXPERIMENT: ${EXP_TAG} (exp λ: ${L_START} -> ${L_END}, mode: ${OUT_MODE})"
    echo "=================================================================="

    if [ -f "$JSON_FILE" ]; then
        echo "[SKIP] ${JSON_FILE} already exists! Keeping completed results."
        continue
    fi

    mkdir -p "$EXP_AGENT/weights" "$EXP_BUF" "$EXP_LEAGUE"
    if [ ! -d "${EXP_LEAGUE}/iter_0" ]; then
        echo "[INIT] Setting up fresh agent & warmup buffer for ${EXP_TAG}..."
        cp "$BASE_AGENT/agent.py" "$BASE_AGENT/features.py" "$BASE_AGENT/model.py" "$EXP_AGENT/"
        if [ -f "$BASE_AGENT/MODEL.md" ]; then cp "$BASE_AGENT/MODEL.md" "$EXP_AGENT/"; fi
        cp data/shared_warmup_iter0/*.npz "$EXP_BUF/"
    fi

    for (( it=0; it<ITERATIONS; it++ )); do
        SNAP_DIR="${EXP_LEAGUE}/iter_${it}"
        if [ -d "$SNAP_DIR" ]; then
            echo "  [RESUME] Iteration ${it} already finished (${SNAP_DIR} exists) -- skipping!"
            continue
        fi

        # Exponential decay for teacher loss weight (LAM) and Stockfish labeling rate (T_RATE)
        LAM=$(python -c "iters=$ITERATIONS; it=$it; s=$L_START; e=$L_END; frac=it/max(1, iters-1); print(f'{s * ((e / s) ** frac) if (s > 0 and e > 0) else 0.0:.3f}')")
        T_RATE=$(python -c "iters=$ITERATIONS; it=$it; s=$TEACHER_RATE_START; e=$TEACHER_RATE_END; frac=it/max(1, iters-1); print(f'{s * ((e / s) ** frac) if (s > 0 and e > 0) else 0.0:.3f}')")
        OUT_W=$(python -c "print('0.500' if $it == 0 else '1.000')")
        MIN_IT=$(python -c "print(1 if $it >= 2 else 0)")

        echo "  [ITER ${it}] Running ${EXP_TAG} (λ=${LAM}, teacher_rate=${T_RATE})..."

        if [ "$it" -gt 0 ]; then
            if [ -z "$(ls -A "$EXP_BUF"/iter_00${it}_shard_*.npz 2>/dev/null || true)" ]; then
                POOL_ARGS=()
                if [ -n "$(ls -d "$EXP_LEAGUE"/iter_* 2>/dev/null || true)" ]; then
                    POOL_ARGS=(--pool-dirs "$EXP_LEAGUE"/iter_*)
                fi
                python training/generate_buffer.py --iter "$it" --games "$GAMES_PER_ITER" --workers "$WORKERS" --nodes 4000 --teacher-rate "$T_RATE" --keep 1.0 --agent-dir "$EXP_AGENT" --out "$EXP_BUF" --seed 42 "${POOL_ARGS[@]}" >> "$LOG_FILE" 2>&1
            fi
        fi

        RESUME_FLAG=()
        if [ "$it" -gt 0 ]; then RESUME_FLAG=(--resume); fi

        python training/train_buffer.py --buffer-dir "$EXP_BUF" --agent-dir "$EXP_AGENT" --max-buffer 500000 --min-iter "$MIN_IT" --channels 128 --blocks 6 --steps "$STEPS_PER_ITER" --batch 2048 --outcome-mode "$OUT_MODE" --outcome-weight "$OUT_W" --teacher-weight "$LAM" --seed "$((42 + it))" "${RESUME_FLAG[@]}" >> "$LOG_FILE" 2>&1

        cp -r "$EXP_AGENT" "$SNAP_DIR"
    done

    if grep -q "Ratings (95% interval):" "$LOG_FILE" 2>/dev/null; then
        echo "[RESUME] Arena ratings already found in ${LOG_FILE}! Skipping Arena."
    else
        echo "[EVAL] Running validate and Arena for ${EXP_AGENT}..."
        python -m fow.validate "$EXP_AGENT" >> "$LOG_FILE" 2>&1
        python -m fow.arena "$EXP_AGENT" agents/baseline bot:greedy bot:guess-sf-4x500 --games "$ARENA_GAMES" --mode round-robin >> "$LOG_FILE" 2>&1
    fi

    echo "[PARSE] Parsing ${LOG_FILE} -> ${JSON_FILE}..."
    python training/plot_experiments.py --parse-log "$LOG_FILE" --name "$EXP_TAG" --agent-name "$EXP_AGENT"
done

echo ""
echo "=================================================================="
echo " GENERATING MATPLOTLIB COMPARISON DASHBOARD"
echo "=================================================================="
python training/plot_experiments.py --plot experiments/json/B_*.json --out experiments/plots/lambda_sweep_B_summary.png

echo "Done! Comparison dashboard saved to experiments/plots/lambda_sweep_B_summary.png"