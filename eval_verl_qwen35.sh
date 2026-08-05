#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
VERL_ROOT="${VERL_ROOT:-$ROOT_DIR/verl}"
VERL_SRC_DIR="${VERL_SRC_DIR:-$VERL_ROOT/verl}"
MAIN_AGENT_EVAL_DIR="${MAIN_AGENT_EVAL_DIR:-$ROOT_DIR/main_agent_eval}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/src/config}"
CONFIG_NAME="${CONFIG_NAME:-search_multiturn_grpo}"

INDEX_PATH="${INDEX_PATH:-$ROOT_DIR/data/browsecomp-plus-indexes/qwen3-embedding-8b}"
CORPUS_REPO_ID="${CORPUS_REPO_ID:-Tevatron/browsecomp-plus-corpus}"
QUERY_ENCODER_MODEL="${QUERY_ENCODER_MODEL:-$ROOT_DIR/models/Qwen3-Embedding-8B}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$ROOT_DIR/models/Qwen3.5-35B-A3B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

RAW_VAL_DATA="${RAW_VAL_DATA:-$ROOT_DIR/data/browsecompplus_searchr1/test.parquet}"
VAL_DATA="${VAL_DATA:-$ROOT_DIR/data/main_agent_eval_browsecomp_test.parquet}"
TRAIN_DATA="${TRAIN_DATA:-$VAL_DATA}"
TOOL_CONFIG="${TOOL_CONFIG:-$MAIN_AGENT_EVAL_DIR/browsecomp_tool_config.yaml}"
RETRIEVAL_SERVER_SCRIPT="${RETRIEVAL_SERVER_SCRIPT:-$MAIN_AGENT_EVAL_DIR/retrieval_server.py}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-$ROOT_DIR/validation_main_agent_qwen35_open_find}"

RETRIEVER_HOST="${RETRIEVER_HOST:-0.0.0.0}"
RETRIEVER_PORT="${RETRIEVER_PORT:-1249}"
RETRIEVER_VISIBLE_DEVICES="${RETRIEVER_VISIBLE_DEVICES:-0,1,2,3}"
EVAL_VISIBLE_DEVICES="${EVAL_VISIBLE_DEVICES:-2,3,4,5,6,7}"
EVAL_GPUS_PER_NODE="${EVAL_GPUS_PER_NODE:-6}"

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
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3.5-35b-a3b-browsecomp-open-find}"

# Clear controls from older experiments so this run only uses the settings below.
unset VERL_TOOL_AGENT_CONTEXT_RESET
unset VERL_TOOL_AGENT_GENERIC_ANSWER_RESCUE_EXTRA_ATTEMPTS
unset VERL_TOOL_AGENT_FRONTIER_OPEN_FIRST
unset VERL_TOOL_AGENT_FRONTIER_OPEN_FIRST_PROMPT_LIMIT
unset VERL_TOOL_AGENT_FRONTIER_OPEN_FIRST_ESCALATE_AFTER
unset VERL_TOOL_AGENT_FRONTIER_OPEN_FIRST_BLOCK_LIMIT
unset VERL_TOOL_AGENT_FRONTIER_OPEN_FIRST_SUGGESTIONS

TOOL_AGENT_DUPLICATE_STREAK_LIMIT="${TOOL_AGENT_DUPLICATE_STREAK_LIMIT:-2}"
TOOL_AGENT_FINAL_RESERVE_TOKENS="${TOOL_AGENT_FINAL_RESERVE_TOKENS:-8192}"
TOOL_AGENT_FINAL_MAX_ATTEMPTS="${TOOL_AGENT_FINAL_MAX_ATTEMPTS:-2}"
TOOL_AGENT_FINAL_ATTEMPT_TOKENS="${TOOL_AGENT_FINAL_ATTEMPT_TOKENS:-2048}"

export VERL_TOOL_AGENT_DUPLICATE_STREAK_LIMIT="$TOOL_AGENT_DUPLICATE_STREAK_LIMIT"
export VERL_TOOL_AGENT_FINAL_RESERVE_TOKENS="$TOOL_AGENT_FINAL_RESERVE_TOKENS"
export VERL_TOOL_AGENT_FINAL_MAX_ATTEMPTS="$TOOL_AGENT_FINAL_MAX_ATTEMPTS"
export VERL_TOOL_AGENT_FINAL_ATTEMPT_TOKENS="$TOOL_AGENT_FINAL_ATTEMPT_TOKENS"

export VERL_TOOL_AGENT_SCAFFOLD_V2="${VERL_TOOL_AGENT_SCAFFOLD_V2:-1}"
export VERL_TOOL_AGENT_SCAFFOLD_V2_MODE="${VERL_TOOL_AGENT_SCAFFOLD_V2_MODE:-soft}"
export VERL_TOOL_AGENT_SEARCH_OVERLAP_THRESHOLD="${VERL_TOOL_AGENT_SEARCH_OVERLAP_THRESHOLD:-0.90}"
export VERL_TOOL_AGENT_SEARCH_ONLY_TURN_LIMIT="${VERL_TOOL_AGENT_SEARCH_ONLY_TURN_LIMIT:-3}"
export VERL_TOOL_AGENT_SEARCH_TO_OPEN_PROMPT_LIMIT="${VERL_TOOL_AGENT_SEARCH_TO_OPEN_PROMPT_LIMIT:-2}"
export VERL_TOOL_AGENT_SEARCH_TO_OPEN_SUGGESTIONS="${VERL_TOOL_AGENT_SEARCH_TO_OPEN_SUGGESTIONS:-5}"

export VERL_TOOL_AGENT_SEARCH_FRONTIER="${VERL_TOOL_AGENT_SEARCH_FRONTIER:-1}"
export VERL_TOOL_AGENT_SEARCH_FRONTIER_BASE_TOPK="${VERL_TOOL_AGENT_SEARCH_FRONTIER_BASE_TOPK:-10}"
export VERL_TOOL_AGENT_SEARCH_FRONTIER_OUTPUT_TOPK="${VERL_TOOL_AGENT_SEARCH_FRONTIER_OUTPUT_TOPK:-10}"
export VERL_TOOL_AGENT_SEARCH_FRONTIER_CANDIDATE_TOPK="${VERL_TOOL_AGENT_SEARCH_FRONTIER_CANDIDATE_TOPK:-50}"
export VERL_TOOL_AGENT_SEARCH_FRONTIER_OVERLAP_THRESHOLD="${VERL_TOOL_AGENT_SEARCH_FRONTIER_OVERLAP_THRESHOLD:-0.90}"
export VERL_TOOL_AGENT_SEARCH_FRONTIER_COMPACT_SNIPPET_CHARS="${VERL_TOOL_AGENT_SEARCH_FRONTIER_COMPACT_SNIPPET_CHARS:-800}"
export VERL_TOOL_AGENT_SEARCH_FRONTIER_EXCLUDE_LIMIT="${VERL_TOOL_AGENT_SEARCH_FRONTIER_EXCLUDE_LIMIT:-1000}"

export VERL_TOOL_AGENT_GENERIC_ANSWER_RESCUE="${VERL_TOOL_AGENT_GENERIC_ANSWER_RESCUE:-1}"
export VERL_TOOL_AGENT_GENERIC_ANSWER_RESCUE_PREFILL="${VERL_TOOL_AGENT_GENERIC_ANSWER_RESCUE_PREFILL:-0}"

export VERL_TOOL_AGENT_VERIFICATION_CHECKPOINT="${VERL_TOOL_AGENT_VERIFICATION_CHECKPOINT:-1}"
export VERL_TOOL_AGENT_VERIFICATION_CHECKPOINT_EXHAUSTED_TURNS="${VERL_TOOL_AGENT_VERIFICATION_CHECKPOINT_EXHAUSTED_TURNS:-2}"
export VERL_TOOL_AGENT_VERIFICATION_CHECKPOINT_SEARCH_ONLY_TURNS="${VERL_TOOL_AGENT_VERIFICATION_CHECKPOINT_SEARCH_ONLY_TURNS:-3}"
export VERL_TOOL_AGENT_VERIFICATION_CHECKPOINT_SEARCH_OVERLAP_THRESHOLD="${VERL_TOOL_AGENT_VERIFICATION_CHECKPOINT_SEARCH_OVERLAP_THRESHOLD:-0.90}"
export VERL_TOOL_AGENT_VERIFICATION_CHECKPOINT_DUPLICATE_STREAK="${VERL_TOOL_AGENT_VERIFICATION_CHECKPOINT_DUPLICATE_STREAK:-1}"
export VERL_TOOL_AGENT_VERIFICATION_CHECKPOINT_SUGGESTIONS="${VERL_TOOL_AGENT_VERIFICATION_CHECKPOINT_SUGGESTIONS:-9}"

export BROWSECOMP_SEARCH_SERVICE_URL="http://127.0.0.1:$RETRIEVER_PORT/search"
export BROWSECOMP_BROWSER_SERVICE_URL="http://127.0.0.1:$RETRIEVER_PORT"
export BROWSECOMP_SEARCH_TOPK="$TOPK"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file not found: $1" >&2
        exit 1
    fi
}

for required_file in \
    "$CONFIG_PATH/$CONFIG_NAME.yaml" \
    "$RAW_VAL_DATA" \
    "$TOOL_CONFIG" \
    "$RETRIEVAL_SERVER_SCRIPT" \
    "$MAIN_AGENT_EVAL_DIR/__init__.py" \
    "$MAIN_AGENT_EVAL_DIR/agent_loop.py" \
    "$MAIN_AGENT_EVAL_DIR/corpus_browser_tool.py" \
    "$MAIN_AGENT_EVAL_DIR/main_agent_eval.py" \
    "$MAIN_AGENT_EVAL_DIR/prepare_eval_data.py" \
    "$MAIN_AGENT_EVAL_DIR/schemas.py" \
    "$MAIN_AGENT_EVAL_DIR/search_r1_like_qa_em.py" \
    "$MAIN_AGENT_EVAL_DIR/search_r1_like_utils.py" \
    "$MAIN_AGENT_EVAL_DIR/search_tool.py" \
    "$MAIN_AGENT_EVAL_DIR/tool_agent_loop.py" \
    "$MAIN_AGENT_EVAL_DIR/tool_parser.py" \
    "$MAIN_AGENT_EVAL_DIR/visit_tool.py"; do
    require_file "$required_file"
done

mkdir -p "$ROOT_DIR/data" "$VALIDATION_DATA_DIR"

CUDA_VISIBLE_DEVICES="$RETRIEVER_VISIBLE_DEVICES" nohup python3 "$RETRIEVAL_SERVER_SCRIPT" \
    --retrieval_method dense \
    --index_path "$INDEX_PATH" \
    --corpus_repo_id "$CORPUS_REPO_ID" \
    --faiss_gpu \
    --query_encoder_model "$QUERY_ENCODER_MODEL" \
    --host "$RETRIEVER_HOST" \
    --port "$RETRIEVER_PORT" \
    --topk "$TOPK" \
    --search_global_enable \
    --search_global_output_topk "$SEARCH_OUTPUT_TOPK" \
    --search_global_rank_cutoff "$TOPK" \
    --search_global_rerank_mode rrf_lexical \
    --search_global_rich_topk "$SEARCH_RICH_TOPK" \
    --search_global_rich_budget "$SEARCH_RICH_BUDGET_CHARS" \
    --search_global_rich_mode chunks \
    --search_global_compact_mode query \
    --search_global_compact_snippet_chars "$SEARCH_COMPACT_SNIPPET_CHARS" \
    --search_global_max_output_chars "$SEARCH_MAX_OUTPUT_CHARS" \
    --search_card_chunk_chars "$SEARCH_RICH_BUDGET_CHARS" \
    --search_card_overlap_chars "$SEARCH_CARD_OVERLAP_CHARS" \
    --search_result_link_label OpenURL \
    --search_result_open_tool_name open \
    --browser_line_chars "$BROWSER_LINE_CHARS" \
    --browser_default_num_lines "$BROWSER_DEFAULT_NUM_LINES" \
    --browser_max_num_lines "$BROWSER_MAX_NUM_LINES" >"$ROOT_DIR/retrieve_main_agent_eval.out" 2>&1 &

while ! nc -z 127.0.0.1 "$RETRIEVER_PORT"; do
    sleep 0.5
done

echo "Retrieval and browser services are ready; preparing Qwen3.5 evaluation."

cp "$MAIN_AGENT_EVAL_DIR/visit_tool.py" "$VERL_SRC_DIR/tools/visit_tool.py"
cp "$MAIN_AGENT_EVAL_DIR/search_r1_like_qa_em.py" "$VERL_SRC_DIR/utils/reward_score/search_r1_like_qa_em.py"
cp "$MAIN_AGENT_EVAL_DIR/__init__.py" "$VERL_SRC_DIR/utils/reward_score/__init__.py"
cp "$MAIN_AGENT_EVAL_DIR/agent_loop.py" "$VERL_SRC_DIR/experimental/agent_loop/agent_loop.py"
cp "$MAIN_AGENT_EVAL_DIR/tool_parser.py" "$VERL_SRC_DIR/experimental/agent_loop/tool_parser.py"
cp "$MAIN_AGENT_EVAL_DIR/search_tool.py" "$VERL_SRC_DIR/tools/search_tool.py"
cp "$MAIN_AGENT_EVAL_DIR/schemas.py" "$VERL_SRC_DIR/tools/schemas.py"
cp "$MAIN_AGENT_EVAL_DIR/corpus_browser_tool.py" "$VERL_SRC_DIR/tools/corpus_browser_tool.py"
cp "$MAIN_AGENT_EVAL_DIR/search_r1_like_utils.py" "$VERL_SRC_DIR/tools/utils/search_r1_like_utils.py"
cp "$MAIN_AGENT_EVAL_DIR/main_agent_eval.py" "$VERL_SRC_DIR/trainer/main_agent_eval.py"
cp "$MAIN_AGENT_EVAL_DIR/tool_agent_loop.py" "$VERL_SRC_DIR/experimental/agent_loop/tool_agent_loop.py"

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
    data.seed=2026 \
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
    trainer.n_gpus_per_node="$EVAL_GPUS_PER_NODE" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    '+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=["Qwen3_5MoeDecoderLayer"]' \
    '+actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=["Qwen3_5MoeDecoderLayer"]' \
    trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
    "${RESUME_ARGS[@]}" \
    "$@"
