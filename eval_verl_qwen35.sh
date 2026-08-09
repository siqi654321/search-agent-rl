#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
VERL_ROOT="${VERL_ROOT:-$ROOT_DIR/verl}"
VERL_SRC_DIR="${VERL_SRC_DIR:-$VERL_ROOT/verl}"
MAIN_AGENT_EVAL_DIR="${MAIN_AGENT_EVAL_DIR:-$ROOT_DIR/main_agent_eval}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/src/config}"
CONFIG_NAME="${CONFIG_NAME:-search_multiturn_grpo}"

# Primary retriever and evaluation model.
INDEX_PATH="${INDEX_PATH:-$ROOT_DIR/data/browsecomp-plus-indexes/qwen3-embedding-8b}"
CORPUS_REPO_ID="${CORPUS_REPO_ID:-Tevatron/browsecomp-plus-corpus}"
QUERY_ENCODER_MODEL="${QUERY_ENCODER_MODEL:-$ROOT_DIR/models/Qwen3-Embedding-8B}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$ROOT_DIR/models/Qwen3.5-35B-A3B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

# Secondary reasoning-aware retriever.
DUAL_RETRIEVAL_ENABLE="${DUAL_RETRIEVAL_ENABLE:-1}"
AGENTIR_INDEX_PATH="${AGENTIR_INDEX_PATH:-$ROOT_DIR/data/agentir/AgentIR_browsecomp-plus}"
AGENTIR_QUERY_ENCODER_MODEL="${AGENTIR_QUERY_ENCODER_MODEL:-$ROOT_DIR/models/AgentIR-4B}"
AGENTIR_RETRIEVER_PORT="${AGENTIR_RETRIEVER_PORT:-1250}"
AGENTIR_RETRIEVER_VISIBLE_DEVICES="${AGENTIR_RETRIEVER_VISIBLE_DEVICES:-0,1,2,3}"
AGENTIR_FAISS_GPU="${AGENTIR_FAISS_GPU:-1}"
DUAL_SEARCH_RATE_LIMIT="${DUAL_SEARCH_RATE_LIMIT:-8}"
DUAL_RETRIEVAL_CANDIDATE_TOPK="${DUAL_RETRIEVAL_CANDIDATE_TOPK:-50}"
DUAL_RETRIEVAL_RRF_K="${DUAL_RETRIEVAL_RRF_K:-60}"
DUAL_RETRIEVAL_PRIMARY_WEIGHT="${DUAL_RETRIEVAL_PRIMARY_WEIGHT:-1.0}"
DUAL_RETRIEVAL_SECONDARY_WEIGHT="${DUAL_RETRIEVAL_SECONDARY_WEIGHT:-2.0}"
DUAL_RETRIEVAL_SECONDARY_TIMEOUT="${DUAL_RETRIEVAL_SECONDARY_TIMEOUT:-600}"
DUAL_REASONING_MAX_CHARS="${DUAL_REASONING_MAX_CHARS:-8000}"
DUAL_RETRIEVAL_SMOKE_TEST="${DUAL_RETRIEVAL_SMOKE_TEST:-1}"

# Repository-local code and data inputs.
RAW_VAL_DATA="${RAW_VAL_DATA:-$ROOT_DIR/data/browsecompplus_searchr1/test.parquet}"
VAL_DATA="${VAL_DATA:-$ROOT_DIR/data/main_agent_eval_browsecomp_test.parquet}"
TRAIN_DATA="${TRAIN_DATA:-$VAL_DATA}"
TOOL_CONFIG="${TOOL_CONFIG:-$MAIN_AGENT_EVAL_DIR/browsecomp_tool_config.yaml}"
RETRIEVAL_SERVER_SCRIPT="${RETRIEVAL_SERVER_SCRIPT:-$MAIN_AGENT_EVAL_DIR/retrieval_server.py}"
SEARCH_TOOL_SOURCE="${SEARCH_TOOL_SOURCE:-$MAIN_AGENT_EVAL_DIR/search_tool.py}"
TOOL_AGENT_LOOP_SOURCE="${TOOL_AGENT_LOOP_SOURCE:-$MAIN_AGENT_EVAL_DIR/tool_agent_loop.py}"

EVAL_SEED="${EVAL_SEED:-2026}"
ROLLOUT_RANDOM_SEED="${ROLLOUT_RANDOM_SEED:-$EVAL_SEED}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-$ROOT_DIR/validation_main_agent_qwen35_dual_seed${EVAL_SEED}}"

RETRIEVER_HOST="${RETRIEVER_HOST:-0.0.0.0}"
RETRIEVER_PORT="${RETRIEVER_PORT:-1249}"
RETRIEVER_VISIBLE_DEVICES="${RETRIEVER_VISIBLE_DEVICES:-0,1,2,3}"
RETRIEVER_FAISS_GPU="${RETRIEVER_FAISS_GPU:-1}"
EVAL_VISIBLE_DEVICES="${EVAL_VISIBLE_DEVICES:-2,3,4,5,6,7}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${EVAL_GPUS_PER_NODE:-6}}"
SERVICE_STARTUP_TIMEOUT_SECONDS="${SERVICE_STARTUP_TIMEOUT_SECONDS:-1800}"

TOPK="${TOPK:-50}"
SEARCH_OUTPUT_TOPK="${SEARCH_OUTPUT_TOPK:-10}"
SEARCH_RICH_TOPK="${SEARCH_RICH_TOPK:-5}"
SEARCH_RICH_BUDGET_CHARS="${SEARCH_RICH_BUDGET_CHARS:-2100}"
SEARCH_COMPACT_SNIPPET_CHARS="${SEARCH_COMPACT_SNIPPET_CHARS:-240}"
SEARCH_MAX_OUTPUT_CHARS="${SEARCH_MAX_OUTPUT_CHARS:-24576}"
SEARCH_CARD_OVERLAP_CHARS="${SEARCH_CARD_OVERLAP_CHARS:-240}"
BROWSER_LINE_CHARS="${BROWSER_LINE_CHARS:-160}"
BROWSER_DEFAULT_NUM_LINES="${BROWSER_DEFAULT_NUM_LINES:-40}"
BROWSER_MAX_NUM_LINES="${BROWSER_MAX_NUM_LINES:-64}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-110000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-122880}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-24576}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-50}"
MAX_USER_TURNS="${MAX_USER_TURNS:-50}"
MAX_PARALLEL_CALLS="${MAX_PARALLEL_CALLS:-3}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-2}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}"
PROJECT_NAME="${PROJECT_NAME:-main_agent_eval}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3.5-35b-a3b-browsecomp-dual-seed${EVAL_SEED}}"

# Match the four independent-trajectory reference runs. These can still be
# overridden explicitly by callers that want to test another controller setup.
export VERL_TOOL_AGENT_SCAFFOLD_V2="${VERL_TOOL_AGENT_SCAFFOLD_V2:-0}"
export VERL_TOOL_AGENT_VERIFICATION_CHECKPOINT="${VERL_TOOL_AGENT_VERIFICATION_CHECKPOINT:-0}"
export VERL_TOOL_AGENT_CONTEXT_RESET="${VERL_TOOL_AGENT_CONTEXT_RESET:-0}"
export VERL_TOOL_AGENT_EVIDENCE_STATE_REBUILD="${VERL_TOOL_AGENT_EVIDENCE_STATE_REBUILD:-0}"
export VERL_TOOL_AGENT_FRONTIER_OPEN_FIRST="${VERL_TOOL_AGENT_FRONTIER_OPEN_FIRST:-0}"
export VERL_TOOL_AGENT_CANDIDATE_WORKSPACE="${VERL_TOOL_AGENT_CANDIDATE_WORKSPACE:-0}"
export VERL_TOOL_AGENT_PI_NAVIGATION="${VERL_TOOL_AGENT_PI_NAVIGATION:-0}"

export VERL_TOOL_AGENT_SEARCH_FRONTIER="${VERL_TOOL_AGENT_SEARCH_FRONTIER:-1}"
export VERL_TOOL_AGENT_SEARCH_FRONTIER_BASE_TOPK="${VERL_TOOL_AGENT_SEARCH_FRONTIER_BASE_TOPK:-10}"
export VERL_TOOL_AGENT_SEARCH_FRONTIER_OUTPUT_TOPK="${VERL_TOOL_AGENT_SEARCH_FRONTIER_OUTPUT_TOPK:-10}"
export VERL_TOOL_AGENT_SEARCH_FRONTIER_CANDIDATE_TOPK="${VERL_TOOL_AGENT_SEARCH_FRONTIER_CANDIDATE_TOPK:-50}"
export VERL_TOOL_AGENT_SEARCH_FRONTIER_OVERLAP_THRESHOLD="${VERL_TOOL_AGENT_SEARCH_FRONTIER_OVERLAP_THRESHOLD:-0.90}"
export VERL_TOOL_AGENT_SEARCH_FRONTIER_COMPACT_SNIPPET_CHARS="${VERL_TOOL_AGENT_SEARCH_FRONTIER_COMPACT_SNIPPET_CHARS:-800}"
export VERL_TOOL_AGENT_SEARCH_FRONTIER_EXCLUDE_LIMIT="${VERL_TOOL_AGENT_SEARCH_FRONTIER_EXCLUDE_LIMIT:-1000}"

export VERL_TOOL_AGENT_GENERIC_ANSWER_RESCUE="${VERL_TOOL_AGENT_GENERIC_ANSWER_RESCUE:-1}"
export VERL_TOOL_AGENT_GENERIC_ANSWER_RESCUE_PREFILL="${VERL_TOOL_AGENT_GENERIC_ANSWER_RESCUE_PREFILL:-0}"
export VERL_TOOL_AGENT_GENERIC_ANSWER_RESCUE_EXTRA_ATTEMPTS="${VERL_TOOL_AGENT_GENERIC_ANSWER_RESCUE_EXTRA_ATTEMPTS:-0}"

CANDIDATE_WORKSPACE_ENABLE="${CANDIDATE_WORKSPACE_ENABLE:-0}"
CANDIDATE_WORKSPACE_MODE="${CANDIDATE_WORKSPACE_MODE:-legacy}"
TOOL_AGENT_PI_NAVIGATION="${TOOL_AGENT_PI_NAVIGATION:-0}"
export CANDIDATE_WORKSPACE_ENABLE CANDIDATE_WORKSPACE_MODE TOOL_AGENT_PI_NAVIGATION

export BROWSECOMP_SEARCH_SERVICE_URL="http://127.0.0.1:$RETRIEVER_PORT/search"
export BROWSECOMP_BROWSER_SERVICE_URL="http://127.0.0.1:$RETRIEVER_PORT"
export BROWSECOMP_SEARCH_TOPK="$TOPK"
export BROWSECOMP_DUAL_RETRIEVAL_ENABLE="$DUAL_RETRIEVAL_ENABLE"
export BROWSECOMP_DUAL_REASONING_MAX_CHARS="$DUAL_REASONING_MAX_CHARS"
if [[ "$DUAL_RETRIEVAL_ENABLE" == "1" || "$DUAL_RETRIEVAL_ENABLE" == "true" ]]; then
    export BROWSECOMP_SEARCH_RATE_LIMIT="$DUAL_SEARCH_RATE_LIMIT"
fi

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file not found: $1" >&2
        exit 1
    fi
}

wait_for_port() {
    local host="$1"
    local port="$2"
    local name="$3"
    local started_at=$SECONDS
    until nc -z "$host" "$port"; do
        if (( SECONDS - started_at >= SERVICE_STARTUP_TIMEOUT_SECONDS )); then
            echo "Timed out waiting for $name at $host:$port" >&2
            exit 1
        fi
        sleep 1
    done
}

for required_file in \
    "$CONFIG_PATH/$CONFIG_NAME.yaml" \
    "$RAW_VAL_DATA" \
    "$TOOL_CONFIG" \
    "$RETRIEVAL_SERVER_SCRIPT" \
    "$SEARCH_TOOL_SOURCE" \
    "$TOOL_AGENT_LOOP_SOURCE" \
    "$MAIN_AGENT_EVAL_DIR/__init__.py" \
    "$MAIN_AGENT_EVAL_DIR/agent_loop.py" \
    "$MAIN_AGENT_EVAL_DIR/corpus_browser_tool.py" \
    "$MAIN_AGENT_EVAL_DIR/main_agent_eval.py" \
    "$MAIN_AGENT_EVAL_DIR/prepare_eval_data.py" \
    "$MAIN_AGENT_EVAL_DIR/schemas.py" \
    "$MAIN_AGENT_EVAL_DIR/search_r1_like_qa_em.py" \
    "$MAIN_AGENT_EVAL_DIR/search_r1_like_utils.py" \
    "$MAIN_AGENT_EVAL_DIR/tool_parser.py" \
    "$MAIN_AGENT_EVAL_DIR/visit_tool.py"; do
    require_file "$required_file"
done

mkdir -p "$ROOT_DIR/data" "$VALIDATION_DATA_DIR"

PRIMARY_FAISS_ARGS=()
if [[ "$RETRIEVER_FAISS_GPU" == "1" || "$RETRIEVER_FAISS_GPU" == "true" ]]; then
    PRIMARY_FAISS_ARGS+=(--faiss_gpu)
fi

RETRIEVER_EXTRA_ARGS=(
    --search_global_enable
    --search_global_output_topk "$SEARCH_OUTPUT_TOPK"
    --search_global_rank_cutoff "$TOPK"
    --search_global_rerank_mode rrf_lexical
    --search_global_rich_topk "$SEARCH_RICH_TOPK"
    --search_global_rich_budget "$SEARCH_RICH_BUDGET_CHARS"
    --search_global_rich_mode chunks
    --search_global_compact_mode query
    --search_global_compact_snippet_chars "$SEARCH_COMPACT_SNIPPET_CHARS"
    --search_global_max_output_chars "$SEARCH_MAX_OUTPUT_CHARS"
    --search_card_chunk_chars "$SEARCH_RICH_BUDGET_CHARS"
    --search_card_overlap_chars "$SEARCH_CARD_OVERLAP_CHARS"
    --search_result_link_label OpenURL
    --search_result_open_tool_name open
    --browser_line_chars "$BROWSER_LINE_CHARS"
    --browser_default_num_lines "$BROWSER_DEFAULT_NUM_LINES"
    --browser_max_num_lines "$BROWSER_MAX_NUM_LINES"
)

if [[ "$DUAL_RETRIEVAL_ENABLE" == "1" || "$DUAL_RETRIEVAL_ENABLE" == "true" ]]; then
    RETRIEVER_EXTRA_ARGS+=(
        --dual_retrieval_enable
        --dual_secondary_rank_url "http://127.0.0.1:$AGENTIR_RETRIEVER_PORT/rank"
        --dual_candidate_topk "$DUAL_RETRIEVAL_CANDIDATE_TOPK"
        --dual_rrf_k "$DUAL_RETRIEVAL_RRF_K"
        --dual_primary_weight "$DUAL_RETRIEVAL_PRIMARY_WEIGHT"
        --dual_secondary_weight "$DUAL_RETRIEVAL_SECONDARY_WEIGHT"
        --dual_primary_rerank_mode rrf
        --dual_secondary_timeout "$DUAL_RETRIEVAL_SECONDARY_TIMEOUT"
    )

    AGENTIR_INSTRUCTION=$'Instruct: Given a user\'s reasoning followed by a web search query, retrieve relevant passages that answer the query while incorporating the user\'s reasoning\nQuery:'
    AGENTIR_FAISS_ARGS=()
    if [[ "$AGENTIR_FAISS_GPU" == "1" || "$AGENTIR_FAISS_GPU" == "true" ]]; then
        AGENTIR_FAISS_ARGS+=(--faiss_gpu)
    fi

    if nc -z 127.0.0.1 "$AGENTIR_RETRIEVER_PORT"; then
        echo "Using the existing secondary retriever on port $AGENTIR_RETRIEVER_PORT."
    else
        CUDA_VISIBLE_DEVICES="$AGENTIR_RETRIEVER_VISIBLE_DEVICES" nohup python3 "$RETRIEVAL_SERVER_SCRIPT" \
            --retrieval_method dense \
            --index_path "$AGENTIR_INDEX_PATH" \
            --corpus_repo_id "$CORPUS_REPO_ID" \
            "${AGENTIR_FAISS_ARGS[@]}" \
            --query_encoder_model "$AGENTIR_QUERY_ENCODER_MODEL" \
            --query_encoder_backend transformer_eos \
            --query_encoder_max_length 8192 \
            --query_encoder_batch_size 8 \
            --query_instruction "$AGENTIR_INSTRUCTION" \
            --host "$RETRIEVER_HOST" \
            --port "$AGENTIR_RETRIEVER_PORT" \
            --topk "$DUAL_RETRIEVAL_CANDIDATE_TOPK" \
            --search_global_rank_cutoff "$DUAL_RETRIEVAL_CANDIDATE_TOPK" \
            --search_global_rerank_mode rrf \
            >"$ROOT_DIR/retrieve_agentir_seed${EVAL_SEED}.out" 2>&1 &
        echo "Secondary retriever PID: $!"
        wait_for_port 127.0.0.1 "$AGENTIR_RETRIEVER_PORT" "secondary retriever"
    fi
fi

if nc -z 127.0.0.1 "$RETRIEVER_PORT"; then
    echo "Using the existing primary retriever on port $RETRIEVER_PORT."
else
    CUDA_VISIBLE_DEVICES="$RETRIEVER_VISIBLE_DEVICES" nohup python3 "$RETRIEVAL_SERVER_SCRIPT" \
        --retrieval_method dense \
        --index_path "$INDEX_PATH" \
        --corpus_repo_id "$CORPUS_REPO_ID" \
        "${PRIMARY_FAISS_ARGS[@]}" \
        --query_encoder_model "$QUERY_ENCODER_MODEL" \
        --host "$RETRIEVER_HOST" \
        --port "$RETRIEVER_PORT" \
        --topk "$TOPK" \
        "${RETRIEVER_EXTRA_ARGS[@]}" \
        >"$ROOT_DIR/retrieve_main_agent_seed${EVAL_SEED}.out" 2>&1 &
    echo "Primary retriever PID: $!"
    wait_for_port 127.0.0.1 "$RETRIEVER_PORT" "primary retriever"
fi

if [[ "$DUAL_RETRIEVAL_SMOKE_TEST" == "1" || "$DUAL_RETRIEVAL_SMOKE_TEST" == "true" ]]; then
    python3 - "$RETRIEVER_PORT" <<'PY'
import json
import sys
import urllib.request

request = urllib.request.Request(
    f"http://127.0.0.1:{sys.argv[1]}/search",
    data=json.dumps(
        {
            "queries": ["example publication current title year"],
            "reasoning": "Find the year in which the publication adopted its current title.",
            "topk": 50,
            "frontier_enable": False,
            "frontier_base_rank_cutoff": 10,
            "frontier_output_topk": 10,
            "frontier_candidate_topk": 50,
        }
    ).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=600) as response:
    payload = json.load(response)
fusion = payload.get("fusion") or {}
print("Dual retrieval smoke test:", json.dumps(fusion, ensure_ascii=False))
if not payload.get("success") or not fusion.get("applied"):
    raise SystemExit("Dual retrieval smoke test failed")
PY
fi

echo "Retrieval services are ready; preparing seed $EVAL_SEED evaluation."

cp "$MAIN_AGENT_EVAL_DIR/visit_tool.py" "$VERL_SRC_DIR/tools/visit_tool.py"
cp "$MAIN_AGENT_EVAL_DIR/search_r1_like_qa_em.py" "$VERL_SRC_DIR/utils/reward_score/search_r1_like_qa_em.py"
cp "$MAIN_AGENT_EVAL_DIR/__init__.py" "$VERL_SRC_DIR/utils/reward_score/__init__.py"
cp "$MAIN_AGENT_EVAL_DIR/agent_loop.py" "$VERL_SRC_DIR/experimental/agent_loop/agent_loop.py"
cp "$MAIN_AGENT_EVAL_DIR/tool_parser.py" "$VERL_SRC_DIR/experimental/agent_loop/tool_parser.py"
cp "$SEARCH_TOOL_SOURCE" "$VERL_SRC_DIR/tools/search_tool.py"
cp "$MAIN_AGENT_EVAL_DIR/schemas.py" "$VERL_SRC_DIR/tools/schemas.py"
cp "$MAIN_AGENT_EVAL_DIR/corpus_browser_tool.py" "$VERL_SRC_DIR/tools/corpus_browser_tool.py"
cp "$MAIN_AGENT_EVAL_DIR/search_r1_like_utils.py" "$VERL_SRC_DIR/tools/utils/search_r1_like_utils.py"
cp "$MAIN_AGENT_EVAL_DIR/main_agent_eval.py" "$VERL_SRC_DIR/trainer/main_agent_eval.py"
cp "$TOOL_AGENT_LOOP_SOURCE" "$VERL_SRC_DIR/experimental/agent_loop/tool_agent_loop.py"

python3 "$MAIN_AGENT_EVAL_DIR/prepare_eval_data.py" \
    --input "$RAW_VAL_DATA" \
    --output "$VAL_DATA" \
    --tool-prompt-mode native \
    --native-tool-profile search_open_find

export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

RESUME_ARGS=()
if [[ -n "$CHECKPOINT_PATH" ]]; then
    RESUME_ARGS+=(
        trainer.resume_mode=resume_path
        trainer.resume_from_path="$CHECKPOINT_PATH"
    )
fi

CUDA_VISIBLE_DEVICES="$EVAL_VISIBLE_DEVICES" python3 -m verl.trainer.main_agent_eval \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    algorithm.adv_estimator=grpo \
    data.shuffle=false \
    data.seed="$EVAL_SEED" \
    data.train_batch_size=256 \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    actor_rollout_ref.model.path="$BASE_MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TENSOR_PARALLEL_SIZE" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers="$AGENT_NUM_WORKERS" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length="$MAX_TOOL_RESPONSE_LENGTH" \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_ASSISTANT_TURNS" \
    actor_rollout_ref.rollout.multi_turn.max_user_turns="$MAX_USER_TURNS" \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls="$MAX_PARALLEL_CALLS" \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_5 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.logger='["console"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.random_seed="$ROLLOUT_RANDOM_SEED" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    '+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=["Qwen3_5MoeDecoderLayer"]' \
    '+actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=["Qwen3_5MoeDecoderLayer"]' \
    trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
    "${RESUME_ARGS[@]}" \
    "$@"
