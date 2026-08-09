#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
VERL_ROOT="${VERL_ROOT:-$ROOT_DIR/verl}"
AGGREGATION_SCRIPT="${AGGREGATION_SCRIPT:-$ROOT_DIR/main_agent_eval/evaluate_multitrajectory.py}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/multitrajectory}"

SEEDS="${SEEDS:-2026 2027 2028 2029}"
ANCHOR_SEED="${ANCHOR_SEED:-2026}"
TRAJECTORY_DIR_PREFIX="${TRAJECTORY_DIR_PREFIX:-$ROOT_DIR/validation_main_agent_qwen35_dual_seed}"
TRAJECTORY_FILENAME="${TRAJECTORY_FILENAME:-0.jsonl}"

AGG_MODEL="${AGG_MODEL:-$ROOT_DIR/models/Qwen3.5-35B-A3B}"
AGG_VISIBLE_DEVICES="${AGG_VISIBLE_DEVICES:-0,1}"
AGG_TENSOR_PARALLEL_SIZE="${AGG_TENSOR_PARALLEL_SIZE:-2}"
AGG_GPU_MEMORY_UTILIZATION="${AGG_GPU_MEMORY_UTILIZATION:-0.85}"
AGG_PORT="${AGG_PORT:-30010}"
AGG_ENDPOINT="${AGG_ENDPOINT:-http://127.0.0.1:$AGG_PORT}"
AGG_MAX_TOKENS="${AGG_MAX_TOKENS:-64}"
AGG_TEMPERATURE="${AGG_TEMPERATURE:-0}"
AGG_CONCURRENCY="${AGG_CONCURRENCY:-4}"
AGG_STARTUP_TIMEOUT_SECONDS="${AGG_STARTUP_TIMEOUT_SECONDS:-1800}"
START_AGGREGATOR_SERVER="${START_AGGREGATOR_SERVER:-1}"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file not found: $1" >&2
        exit 1
    fi
}

require_file "$AGGREGATION_SCRIPT"
for helper in \
    evaluate_browsecomp_answer_verifier.py \
    evaluate_browsecomp_candidate_selector.py \
    evaluate_browsecomp_multitrajectory.py; do
    require_file "$VERL_ROOT/scripts/$helper"
done

read -r -a SEED_LIST <<<"$SEEDS"
if (( ${#SEED_LIST[@]} < 2 || ${#SEED_LIST[@]} > 8 )); then
    echo "SEEDS must contain between two and eight values" >&2
    exit 1
fi

TRAJECTORY_ARGS=()
anchor_found=0
for seed in "${SEED_LIST[@]}"; do
    label="seed${seed}"
    override_name="TRAJECTORY_${seed}"
    trajectory_path="${!override_name:-${TRAJECTORY_DIR_PREFIX}${seed}/${TRAJECTORY_FILENAME}}"
    require_file "$trajectory_path"
    TRAJECTORY_ARGS+=(--trajectory "$label=$trajectory_path")
    if [[ "$seed" == "$ANCHOR_SEED" ]]; then
        anchor_found=1
    fi
done
if (( anchor_found == 0 )); then
    echo "ANCHOR_SEED=$ANCHOR_SEED is not present in SEEDS=$SEEDS" >&2
    exit 1
fi

trajectory_count="${#SEED_LIST[@]}"
run_name="${RUN_NAME:-browsecomp_qwen35_k${trajectory_count}}"
input_path="${AGG_INPUT:-$OUTPUT_DIR/${run_name}.input.jsonl}"
predictions_path="${AGG_PREDICTIONS:-$OUTPUT_DIR/${run_name}.predictions.jsonl}"
report_path="${AGG_REPORT:-$OUTPUT_DIR/${run_name}.report.json}"
server_log="${AGG_SERVER_LOG:-$OUTPUT_DIR/${run_name}.server.out}"
mkdir -p "$OUTPUT_DIR"

(
    cd "$VERL_ROOT"
    python3 "$AGGREGATION_SCRIPT" prepare \
        "${TRAJECTORY_ARGS[@]}" \
        --anchor "seed${ANCHOR_SEED}" \
        --output "$input_path"
)

if [[ "$START_AGGREGATOR_SERVER" == "1" || "$START_AGGREGATOR_SERVER" == "true" ]]; then
    if nc -z 127.0.0.1 "$AGG_PORT"; then
        echo "Using the existing aggregation server on port $AGG_PORT."
    else
        CUDA_VISIBLE_DEVICES="$AGG_VISIBLE_DEVICES" nohup python3 -m sglang.launch_server \
            --model-path "$AGG_MODEL" \
            --tensor-parallel-size "$AGG_TENSOR_PARALLEL_SIZE" \
            --mem-fraction-static "$AGG_GPU_MEMORY_UTILIZATION" \
            --port "$AGG_PORT" \
            >"$server_log" 2>&1 &
        echo "Aggregation server PID: $!"

        started_at=$SECONDS
        until nc -z 127.0.0.1 "$AGG_PORT"; do
            if (( SECONDS - started_at >= AGG_STARTUP_TIMEOUT_SECONDS )); then
                echo "Timed out waiting for the aggregation server; see $server_log" >&2
                exit 1
            fi
            sleep 1
        done
    fi
fi

(
    cd "$VERL_ROOT"
    python3 "$AGGREGATION_SCRIPT" run \
        --input "$input_path" \
        --output "$predictions_path" \
        --endpoint "$AGG_ENDPOINT" \
        --model "$AGG_MODEL" \
        --max-tokens "$AGG_MAX_TOKENS" \
        --temperature "$AGG_TEMPERATURE" \
        --concurrency "$AGG_CONCURRENCY"
)

(
    cd "$VERL_ROOT"
    python3 "$AGGREGATION_SCRIPT" evaluate \
        "${TRAJECTORY_ARGS[@]}" \
        --input "$input_path" \
        --predictions "$predictions_path"
) | tee "$report_path"

echo "Aggregation report: $report_path"
