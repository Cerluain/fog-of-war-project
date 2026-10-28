#!/usr/bin/env bash
# Extended ablation sweep: teacher-weight decay schedules, loss modes, and anchor strengths.
set -euo pipefail

ITERATIONS=${ITERATIONS:-4}
WARMUP_GAMES=${WARMUP_GAMES:-1600}
GAMES_PER_ITER=${GAMES_PER_ITER:-1200}
STEPS_PER_ITER=${STEPS_PER_ITER:-600}
ARENA_GAMES=${ARENA_GAMES:-150}
WORKERS=${WORKERS:-4}
BASE_AGENT=${BASE_AGENT:-"agents/v1_se_model"}

mkdir -p experiments/logs experiments/json experiments/plots data/shared_warmup_iter0

echo "=================================================================="
echo " STEP 0: Checking Shared Warmup Iteration 0 (${WARMUP_GAMES} games)"
echo "=================================================================="
if [ -z "$(ls -A data/shared_warmup_iter0/*.npz 2>/dev/null || true)" ]; then
    python training/generate_buffer.py --iter 0 --games "$WARMUP_GAMES" --workers "$WORKERS" --nodes 4000 --teacher-rate 0.5 --keep 1.0 --agent-dir "$BASE_AGENT" --out data/shared_warmup_iter0 --seed 42
else
    echo "Shared Iteration 0 shards already exist in data/shared_warmup_iter0/ -- reusing!"
fi

# Format: "L_START L_END OUT_MODE EXP_NAME"
SCHEDULES=(
    "1.0 1.0 advantage lam_1.0_to_1.0_fixed"
    "1.0 0.3 advantage lam_1.0_to_0.3_moderate"
    "1.0 0.05 advantage lam_1.0_to_0.05_steep"
    "0.5 0.1 advantage lam_0.5_to_0.1_lowteacher"
    "2.0 0.5 advantage lam_2.0_to_0.5_heavyanchor"
    "0.0 0.0 advantage lam_0.0_to_0.0_pureoutcome"
    "1.0 0.05 win_only lam_1.0_to_0.05_winonly"
    "1.5 0.1 advantage lam_1.5_to_0.1_annealed"
)

for entry in "${SCHEDULES[@]}"; do
    read -r L_START L_END OUT_MODE EXP_NAME <<< "$entry"
    EXP_AGENT="agents/exp_${EXP_NAME}"
    EXP_BUF="data/buf_${EXP_NAME}"
    EXP_LEAGUE="data/league_${EXP_NAME}"
    LOG_FILE="experiments/logs/${EXP_NAME}.log"
    JSON_FILE="experiments/json/${EXP_NAME}.json"

    echo ""
    echo "=================================================================="
    echo " EXPERIMENT: ${EXP_NAME} (λ: ${L_START} -> ${L_END}, mode: ${OUT_MODE})"
    echo "=================================================================="

    if [ -f "$JSON_FILE" ]; then
        echo "[SKIP] ${JSON_FILE} already exists! Keeping completed results."
        continue
    fi

    mkdir -p "$EXP_AGENT/weights" "$EXP_BUF" "$EXP_LEAGUE"
    if [ ! -d "${EXP_LEAGUE}/iter_0" ]; then
        echo "[INIT] Setting up fresh agent & warmup buffer for ${EXP_NAME}..."
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

        echo "  [ITER ${it}] Running generation & training for ${EXP_NAME}..."
        LAM=$(python -c "iters=$ITERATIONS; it=$it; s=$L_START; e=$L_END; print(f'{s + (it / max(1, iters - 1)) * (e - s):.3f}')")
        OUT_W=$(python -c "print('0.500' if $it == 0 else '1.000')")
        MIN_IT=$(python -c "print(1 if $it >= 2 else 0)")

        if [ "$it" -gt 0 ]; then
            if [ -z "$(ls -A "$EXP_BUF"/iter_00${it}_shard_*.npz 2>/dev/null || true)" ]; then
                POOL_ARGS=()
                if [ -n "$(ls -d "$EXP_LEAGUE"/iter_* 2>/dev/null || true)" ]; then
                    POOL_ARGS=(--pool-dirs "$EXP_LEAGUE"/iter_*)
                fi
                python training/generate_buffer.py --iter "$it" --games "$GAMES_PER_ITER" --workers "$WORKERS" --nodes 4000 --teacher-rate 0.4 --keep 1.0 --agent-dir "$EXP_AGENT" --out "$EXP_BUF" --seed 42 "${POOL_ARGS[@]}" >> "$LOG_FILE" 2>&1
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
    python training/plot_experiments.py --parse-log "$LOG_FILE" --name "$EXP_NAME" --agent-name "$EXP_AGENT"
done

echo ""
echo "=================================================================="
echo " GENERATING MATPLOTLIB COMPARISON DASHBOARD"
echo "=================================================================="
python training/plot_experiments.py --plot experiments/json/*.json --out experiments/plots/lambda_sweep_summary.png

echo "Done! Comparison dashboard saved to experiments/plots/lambda_sweep_summary.png"