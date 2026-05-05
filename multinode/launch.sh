#!/usr/bin/env bash
set -xeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "$0")/.." && pwd)"

RAY_ADDRESS="${RAY_ADDRESS:-auto}"
SERVICE_MODE="${SERVICE_MODE:-per-node}"
CONFIG_PATH="${CONFIG_PATH:-src/config}"
TOOL_CONFIG_TEMPLATE="${TOOL_CONFIG_TEMPLATE:-src/config/tool_config/search_tool_config.yaml}"
TRAINING_SCRIPT="${TRAINING_SCRIPT:-multinode/train_verl_sync.sh}"
TRAIN_DATA="${TRAIN_DATA:-data/asearcher_searchr1/train.parquet}"
VAL_DATA="${VAL_DATA:-data/asearcher_searchr1/test.parquet}"
RETRIEVAL_SCRIPT="${RETRIEVAL_SCRIPT:-retrieval/retrieval_server_sglang_summarize.py}"
INDEX_PATH="${INDEX_PATH:-data/e5_Flat.index}"
CORPUS_PATH="${CORPUS_PATH:-data/wiki-18.jsonl}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-models/e5-base-v2}"
SUMMARY_MODEL_PATH="${SUMMARY_MODEL_PATH:-models/Qwen3-1.7B}"
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-models/Qwen3-8B}"
PROJECT_NAME="${PROJECT_NAME:-search_r1_like_multinode_rl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-8b-search-multinode}"
SUMMARY_MODEL_NAME="${SUMMARY_MODEL_NAME:-default}"
SUMMARY_TP="${SUMMARY_TP:-1}"
SUMMARY_MEM_FRACTION="${SUMMARY_MEM_FRACTION:-0.5}"
SUMMARY_PORT="${SUMMARY_PORT:-30000}"
SUMMARY_STARTUP_TIMEOUT_S="${SUMMARY_STARTUP_TIMEOUT_S:-300}"
RETRIEVAL_PORT="${RETRIEVAL_PORT:-1249}"
RETRIEVAL_STARTUP_TIMEOUT_S="${RETRIEVAL_STARTUP_TIMEOUT_S:-900}"
RETRIEVAL_NUM_GPUS="${RETRIEVAL_NUM_GPUS:-1}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-4,5,6,7}"
TRAIN_NNODES="${TRAIN_NNODES:-1}"
TRAIN_GPUS_PER_NODE="${TRAIN_GPUS_PER_NODE:-4}"
SUMMARY_VISIBLE_DEVICES="${SUMMARY_VISIBLE_DEVICES:-0}"
RETRIEVAL_VISIBLE_DEVICES="${RETRIEVAL_VISIBLE_DEVICES:-1,2,3}"
NODE_GPUS="${NODE_GPUS:-8}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"

JOB_COMMAND=$(cat <<EOF
export PYTHONPATH=\"\$PWD\${PYTHONPATH:+:\$PYTHONPATH}\"
export RAY_ADDRESS=$(printf '%q' "$RAY_ADDRESS")
export SERVICE_MODE=$(printf '%q' "$SERVICE_MODE")
export CONFIG_PATH=$(printf '%q' "$CONFIG_PATH")
export TOOL_CONFIG_TEMPLATE=$(printf '%q' "$TOOL_CONFIG_TEMPLATE")
export TRAINING_SCRIPT=$(printf '%q' "$TRAINING_SCRIPT")
export TRAIN_DATA=$(printf '%q' "$TRAIN_DATA")
export VAL_DATA=$(printf '%q' "$VAL_DATA")
export RETRIEVAL_SCRIPT=$(printf '%q' "$RETRIEVAL_SCRIPT")
export INDEX_PATH=$(printf '%q' "$INDEX_PATH")
export CORPUS_PATH=$(printf '%q' "$CORPUS_PATH")
export RETRIEVER_MODEL=$(printf '%q' "$RETRIEVER_MODEL")
export SUMMARY_MODEL_PATH=$(printf '%q' "$SUMMARY_MODEL_PATH")
export ACTOR_MODEL_PATH=$(printf '%q' "$ACTOR_MODEL_PATH")
export PROJECT_NAME=$(printf '%q' "$PROJECT_NAME")
export EXPERIMENT_NAME=$(printf '%q' "$EXPERIMENT_NAME")
export SUMMARY_MODEL_NAME=$(printf '%q' "$SUMMARY_MODEL_NAME")
export SUMMARY_TP=$(printf '%q' "$SUMMARY_TP")
export SUMMARY_MEM_FRACTION=$(printf '%q' "$SUMMARY_MEM_FRACTION")
export SUMMARY_PORT=$(printf '%q' "$SUMMARY_PORT")
export SUMMARY_STARTUP_TIMEOUT_S=$(printf '%q' "$SUMMARY_STARTUP_TIMEOUT_S")
export RETRIEVAL_PORT=$(printf '%q' "$RETRIEVAL_PORT")
export RETRIEVAL_STARTUP_TIMEOUT_S=$(printf '%q' "$RETRIEVAL_STARTUP_TIMEOUT_S")
export RETRIEVAL_NUM_GPUS=$(printf '%q' "$RETRIEVAL_NUM_GPUS")
export TRAIN_CUDA_VISIBLE_DEVICES=$(printf '%q' "$TRAIN_CUDA_VISIBLE_DEVICES")
export TRAIN_NNODES=$(printf '%q' "$TRAIN_NNODES")
export TRAIN_GPUS_PER_NODE=$(printf '%q' "$TRAIN_GPUS_PER_NODE")
export SUMMARY_VISIBLE_DEVICES=$(printf '%q' "$SUMMARY_VISIBLE_DEVICES")
export RETRIEVAL_VISIBLE_DEVICES=$(printf '%q' "$RETRIEVAL_VISIBLE_DEVICES")
export NODE_GPUS=$(printf '%q' "$NODE_GPUS")
export RESUME_FROM_PATH=$(printf '%q' "$RESUME_FROM_PATH")
python3 src/multinode/ray_launch_sync_training.py --training-script $(printf '%q' "$TRAINING_SCRIPT")
EOF
)

ray job submit --address="$RAY_ADDRESS" --working-dir "$ROOT_DIR" -- bash -lc "$JOB_COMMAND"
