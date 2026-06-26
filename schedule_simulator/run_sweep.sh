#!/bin/bash
# Parameter sweep: bin_pack (N x k x cap) + cache-aware baselines
# All experiments use full 491901 requests dataset

set -e

MAX_PARALLEL=20
DATASET="/sgl-workspace/claude_workspace/data/h21_32_256k_full.jsonl"
NUM_PROMPTS=491901
PREDICTOR="/sgl-workspace/claude_workspace/data/qwen36_predictor/qwen36_prefill_predictor_qwen_lookup.pkl"
BASE_DIR="/sgl-workspace/claude_workspace/data/qwen36_timeline_0608/sweep_results"
SIM_CMD="python3 /sgl-workspace/claude_workspace/tair-kvcache/schedule_simulator/scripts/run_simulation.py"

# Common parameters
COMMON_ARGS="--dataset $DATASET --num-prompts $NUM_PROMPTS --pool-capacity 0 --request-level --enable-hierarchical --enable-p2p --predictor-pkl $PREDICTOR --kv-bytes-per-token 46080 --hbm-capacity 533 --mem-capacity 340 --page-size 2048 --data-block-size 256 --quiet"

mkdir -p "$BASE_DIR"

# Track running jobs
RUNNING=0
PIDS=()

wait_for_slot() {
    while [ $RUNNING -ge $MAX_PARALLEL ]; do
        # Wait for any child to finish
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                wait "${PIDS[$i]}" 2>/dev/null || true
                unset 'PIDS[$i]'
                RUNNING=$((RUNNING - 1))
            fi
        done
        # Compact array
        PIDS=("${PIDS[@]}")
        if [ $RUNNING -ge $MAX_PARALLEL ]; then
            sleep 5
        fi
    done
}

launch_job() {
    local out_dir="$1"
    shift
    if [ -f "$out_dir/simulation_summary.json" ]; then
        echo "[SKIP] $out_dir (already exists)"
        return
    fi
    wait_for_slot
    echo "[START] $out_dir"
    mkdir -p "$out_dir"
    $SIM_CMD $COMMON_ARGS "$@" --output-dir "$out_dir" > "$out_dir/run.log" 2>&1 &
    PIDS+=($!)
    RUNNING=$((RUNNING + 1))
}

# ============================================================
# 1. cache-aware baselines: N = 80, 120, 160, 200
# ============================================================
for N in 80 120 160 200; do
    OUT="$BASE_DIR/sweep_cacheaware_N${N}"
    launch_job "$OUT" --num-p-instances $N --routing cache_aware
done

# ============================================================
# 2. bin_pack: N=[80,120,160,200] x k=[4,8,12,16,20] x cap=[3,5,7]
# ============================================================
for N in 80 120 160 200; do
    for k in 4 8 12 16 20; do
        for cap in 3 5 7; do
            OUT="$BASE_DIR/sweep_binpack_N${N}_k${k}_cap${cap}"
            launch_job "$OUT" --num-p-instances $N --routing bin_pack --pods-per-group $k --bin-capacity $cap
        done
    done
done

# ============================================================
# Wait for all remaining jobs
# ============================================================
echo "[INFO] All jobs launched. Waiting for completion..."
wait
echo "[DONE] All experiments complete. Results in: $BASE_DIR"
