#!/usr/bin/env bash
set -xeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
VERL_ROOT="${VERL_ROOT:-$ROOT_DIR/verl}"
VERL_SRC_DIR="${VERL_SRC_DIR:-$VERL_ROOT/verl}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/src/config}"
CONFIG_NAME="${CONFIG_NAME:-search_multiturn_grpo}"

INDEX_PATH="${INDEX_PATH:-$ROOT_DIR/data/e5_Flat.index}"
CORPUS_PATH="${CORPUS_PATH:-$ROOT_DIR/data/wiki-18.jsonl}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-$ROOT_DIR/models/e5-base-v2}"
SUMMARY_MODEL_PATH="${SUMMARY_MODEL_PATH:-$ROOT_DIR/models/Qwen3-1.7B}"
SUMMARY_MODEL_NAME="${SUMMARY_MODEL_NAME:-qwen3-1.7b}"
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-$ROOT_DIR/models/Qwen3-4B}"

RAW_TRAIN_DATA="${RAW_TRAIN_DATA:-$ROOT_DIR/data/asearcher_searchr1/train.parquet}"
RAW_VAL_DATA="${RAW_VAL_DATA:-$ROOT_DIR/data/asearcher_searchr1/test.parquet}"
TRAIN_DATA="${TRAIN_DATA:-$ROOT_DIR/data/mtools_train.parquet}"
VAL_DATA="${VAL_DATA:-$ROOT_DIR/data/mtools_test.parquet}"
TOOL_CONFIG="${TOOL_CONFIG:-$ROOT_DIR/src/mtools/mtool_config.yaml}"

RETRIEVER_HOST="${RETRIEVER_HOST:-0.0.0.0}"
RETRIEVER_PORT="${RETRIEVER_PORT:-1249}"
SUMMARY_PORT="${SUMMARY_PORT:-30000}"
SUMMARY_BASE_URL="${SUMMARY_BASE_URL:-http://127.0.0.1:$SUMMARY_PORT}"

SUMMARY_VISIBLE_DEVICES="${SUMMARY_VISIBLE_DEVICES:-1}"
RETRIEVER_VISIBLE_DEVICES="${RETRIEVER_VISIBLE_DEVICES:-0,1,2,3}"
TRAIN_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES:-4,5,6,7}"

TOPK="${TOPK:-10}"
SUMMARY_TIMEOUT="${SUMMARY_TIMEOUT:-600}"
SUMMARY_MAX_TOKENS="${SUMMARY_MAX_TOKENS:-1024}"
LITE_SEARCH_SNIPPET_CHARS="${LITE_SEARCH_SNIPPET_CHARS:-120}"
VISIT_EVIDENCE_MAX_CHARS="${VISIT_EVIDENCE_MAX_CHARS:-1200}"
VISIT_SUMMARY_MAX_CHARS="${VISIT_SUMMARY_MAX_CHARS:-600}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-110000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-122880}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-2048}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-100}"
MAX_USER_TURNS="${MAX_USER_TURNS:-100}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
ROLLOUT_N="${ROLLOUT_N:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE:-1}"
REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-1}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-50}"
PROJECT_NAME="${PROJECT_NAME:-mtools_train}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-4b-mtools-search-visit}"

mkdir -p "$ROOT_DIR/data"

CUDA_VISIBLE_DEVICES="$SUMMARY_VISIBLE_DEVICES" nohup python -m sglang.launch_server \
    --model-path "$SUMMARY_MODEL_PATH" \
    --tensor-parallel-size=1 \
    --mem-fraction-static 0.5 \
    --port "$SUMMARY_PORT" >"$ROOT_DIR/qwen3_1.7b.out" 2>&1 &

echo "等待摘要服务启动..."
while ! nc -z 127.0.0.1 "$SUMMARY_PORT"; do
    sleep 0.5
done

CUDA_VISIBLE_DEVICES="$RETRIEVER_VISIBLE_DEVICES" nohup python3 "$ROOT_DIR/src/mtools/retrieval_server_train.py" \
    --index_path "$INDEX_PATH" \
    --corpus_path "$CORPUS_PATH" \
    --faiss_gpu \
    --retriever_name e5 \
    --retriever_model "$RETRIEVER_MODEL" \
    --sglang_base_url "$SUMMARY_BASE_URL" \
    --sglang_model "$SUMMARY_MODEL_NAME" \
    --host "$RETRIEVER_HOST" \
    --port "$RETRIEVER_PORT" \
    --topk "$TOPK" \
    --sglang_timeout "$SUMMARY_TIMEOUT" \
    --sglang_max_tokens "$SUMMARY_MAX_TOKENS" \
    --lite_search_snippet_chars "$LITE_SEARCH_SNIPPET_CHARS" \
    --visit_evidence_max_chars "$VISIT_EVIDENCE_MAX_CHARS" \
    --visit_summary_max_chars "$VISIT_SUMMARY_MAX_CHARS" >"$ROOT_DIR/retrieve_mtools_train.out" 2>&1 &

echo "等待检索服务启动..."
while ! nc -z 127.0.0.1 "$RETRIEVER_PORT"; do
    sleep 0.5
done

echo "服务已就绪，开始准备训练环境"

cp "$ROOT_DIR/src/tool_parser.py" "$VERL_SRC_DIR/experimental/agent_loop/tool_parser.py"
cp "$ROOT_DIR/src/search_r1_like_qa_em.py" "$VERL_SRC_DIR/utils/reward_score/search_r1_like_qa_em.py"
cp "$ROOT_DIR/src/__init__.py" "$VERL_SRC_DIR/utils/reward_score/__init__.py"
cp "$ROOT_DIR/src/mtools/search_r1_like_utils.py" "$VERL_SRC_DIR/tools/utils/search_r1_like_utils.py"
cp "$ROOT_DIR/src/mtools/search_tool.py" "$VERL_SRC_DIR/tools/search_tool.py"
cp "$ROOT_DIR/src/mtools/visit_tool.py" "$VERL_SRC_DIR/tools/visit_tool.py"

python3 "$ROOT_DIR/src/mtools/prepare_mtool_data.py" --input "$RAW_TRAIN_DATA" --output "$TRAIN_DATA"
python3 "$ROOT_DIR/src/mtools/prepare_mtool_data.py" --input "$RAW_VAL_DATA" --output "$VAL_DATA"

export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES="$TRAIN_VISIBLE_DEVICES" python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    algorithm.adv_estimator=grpo \
    data.shuffle=true \
    data.seed=2026 \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers="$AGENT_NUM_WORKERS" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length="$MAX_TOOL_RESPONSE_LENGTH" \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_ASSISTANT_TURNS" \
    actor_rollout_ref.rollout.multi_turn.max_user_turns="$MAX_USER_TURNS" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_LOG_PROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=false \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize=false \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.nnodes="$NNODES" \
    trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs="$TOTAL_EPOCHS"
