#!/usr/bin/env bash
set -xeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
VERL_ROOT="${VERL_ROOT:-$ROOT_DIR/verl}"
VERL_SRC_DIR="${VERL_SRC_DIR:-$VERL_ROOT/verl}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/src/config}"
CONFIG_NAME="${CONFIG_NAME:-search_multiturn_grpo}"

INDEX_PATH="${INDEX_PATH:-$ROOT_DIR/data/browsecomp-plus-indexes/qwen3-embedding-8b}"
CORPUS_REPO_ID="${CORPUS_REPO_ID:-Tevatron/browsecomp-plus-corpus}"
QUERY_ENCODER_MODEL="${QUERY_ENCODER_MODEL:-$ROOT_DIR/models/Qwen3-Embedding-8B}"
SUMMARY_MODEL_PATH="${SUMMARY_MODEL_PATH:-$ROOT_DIR/models/Qwen3-1.7B}"
SUMMARY_MODEL_NAME="${SUMMARY_MODEL_NAME:-qwen3-1.7b}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$ROOT_DIR/models/Qwen3-4B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$ROOT_DIR/checkpoints/qwen3-4b-mtools/global_step_250}"
RAW_VAL_DATA="${RAW_VAL_DATA:-$ROOT_DIR/data/browsecompplus_searchr1/test.parquet}"
VAL_DATA="${VAL_DATA:-$ROOT_DIR/data/mtools_browsecomp_test.parquet}"
TRAIN_DATA="${TRAIN_DATA:-$VAL_DATA}"
TOOL_CONFIG="${TOOL_CONFIG:-$ROOT_DIR/src/mtools/mtool_config.yaml}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-$ROOT_DIR/validation_mtools}"

RETRIEVER_HOST="${RETRIEVER_HOST:-0.0.0.0}"
RETRIEVER_PORT="${RETRIEVER_PORT:-1249}"
SUMMARY_PORT_1="${SUMMARY_PORT_1:-30000}"
SUMMARY_PORT_2="${SUMMARY_PORT_2:-30001}"
SUMMARY_PORT_3="${SUMMARY_PORT_3:-30002}"
SUMMARY_BASE_URLS="${SUMMARY_BASE_URLS:-http://127.0.0.1:$SUMMARY_PORT_1,http://127.0.0.1:$SUMMARY_PORT_2,http://127.0.0.1:$SUMMARY_PORT_3}"

RETRIEVER_VISIBLE_DEVICES="${RETRIEVER_VISIBLE_DEVICES:-0,1,2,3}"
SUMMARY_VISIBLE_DEVICES_1="${SUMMARY_VISIBLE_DEVICES_1:-1}"
SUMMARY_VISIBLE_DEVICES_2="${SUMMARY_VISIBLE_DEVICES_2:-2}"
SUMMARY_VISIBLE_DEVICES_3="${SUMMARY_VISIBLE_DEVICES_3:-3}"
EVAL_VISIBLE_DEVICES="${EVAL_VISIBLE_DEVICES:-4,5,6,7}"

TOPK="${TOPK:-10}"
MAX_DOC_CHARS="${MAX_DOC_CHARS:-4000}"
SUMMARIZE_RETRY_MAX_DOC_CHARS="${SUMMARIZE_RETRY_MAX_DOC_CHARS:-4000,2000}"
MAX_TITLE_CHARS="${MAX_TITLE_CHARS:-120}"
LITE_SEARCH_SNIPPET_CHARS="${LITE_SEARCH_SNIPPET_CHARS:-120}"
VISIT_EVIDENCE_MAX_CHARS="${VISIT_EVIDENCE_MAX_CHARS:-1200}"
VISIT_SUMMARY_MAX_CHARS="${VISIT_SUMMARY_MAX_CHARS:-600}"
SUMMARY_TIMEOUT="${SUMMARY_TIMEOUT:-60000}"
SUMMARY_MAX_TOKENS="${SUMMARY_MAX_TOKENS:-2048}"

mkdir -p "$ROOT_DIR/data" "$VALIDATION_DATA_DIR"

CUDA_VISIBLE_DEVICES="$SUMMARY_VISIBLE_DEVICES_1" nohup python -m sglang.launch_server \
    --model-path "$SUMMARY_MODEL_PATH" \
    --tensor-parallel-size=1 \
    --mem-fraction-static 0.6 \
    --port "$SUMMARY_PORT_1" >"$ROOT_DIR/summary_${SUMMARY_PORT_1}.out" 2>&1 &
CUDA_VISIBLE_DEVICES="$SUMMARY_VISIBLE_DEVICES_2" nohup python -m sglang.launch_server \
    --model-path "$SUMMARY_MODEL_PATH" \
    --tensor-parallel-size=1 \
    --mem-fraction-static 0.6 \
    --port "$SUMMARY_PORT_2" >"$ROOT_DIR/summary_${SUMMARY_PORT_2}.out" 2>&1 &
CUDA_VISIBLE_DEVICES="$SUMMARY_VISIBLE_DEVICES_3" nohup python -m sglang.launch_server \
    --model-path "$SUMMARY_MODEL_PATH" \
    --tensor-parallel-size=1 \
    --mem-fraction-static 0.6 \
    --port "$SUMMARY_PORT_3" >"$ROOT_DIR/summary_${SUMMARY_PORT_3}.out" 2>&1 &

while ! nc -z 127.0.0.1 "$SUMMARY_PORT_1"; do
    sleep 0.5
done
while ! nc -z 127.0.0.1 "$SUMMARY_PORT_2"; do
    sleep 0.5
done
while ! nc -z 127.0.0.1 "$SUMMARY_PORT_3"; do
    sleep 0.5
done

CUDA_VISIBLE_DEVICES="$RETRIEVER_VISIBLE_DEVICES" nohup python3 "$ROOT_DIR/src/mtools/retrieval_server_browsecomp_plus.py" \
    --retrieval_method dense \
    --index_path "$INDEX_PATH" \
    --corpus_repo_id "$CORPUS_REPO_ID" \
    --faiss_gpu \
    --query_encoder_model "$QUERY_ENCODER_MODEL" \
    --sglang_base_url "$SUMMARY_BASE_URLS" \
    --sglang_model "$SUMMARY_MODEL_NAME" \
    --host "$RETRIEVER_HOST" \
    --port "$RETRIEVER_PORT" \
    --topk "$TOPK" \
    --max_doc_chars "$MAX_DOC_CHARS" \
    --summarize_retry_max_doc_chars "$SUMMARIZE_RETRY_MAX_DOC_CHARS" \
    --max_title_chars "$MAX_TITLE_CHARS" \
    --lite_search_snippet_chars "$LITE_SEARCH_SNIPPET_CHARS" \
    --visit_evidence_max_chars "$VISIT_EVIDENCE_MAX_CHARS" \
    --visit_summary_max_chars "$VISIT_SUMMARY_MAX_CHARS" \
    --sglang_timeout "$SUMMARY_TIMEOUT" \
    --sglang_max_tokens "$SUMMARY_MAX_TOKENS" >"$ROOT_DIR/retrieve_mtools_eval.out" 2>&1 &

while ! nc -z 127.0.0.1 "$RETRIEVER_PORT"; do
    sleep 0.5
done

echo "服务已就绪，开始准备评估环境"

cp "$ROOT_DIR/src/tool_parser.py" "$VERL_SRC_DIR/experimental/agent_loop/tool_parser.py"
cp "$ROOT_DIR/src/search_r1_like_qa_em.py" "$VERL_SRC_DIR/utils/reward_score/search_r1_like_qa_em.py"
cp "$ROOT_DIR/src/__init__.py" "$VERL_SRC_DIR/utils/reward_score/__init__.py"
cp "$ROOT_DIR/src/mtools/search_r1_like_utils.py" "$VERL_SRC_DIR/tools/utils/search_r1_like_utils.py"
cp "$ROOT_DIR/src/mtools/search_tool.py" "$VERL_SRC_DIR/tools/search_tool.py"
cp "$ROOT_DIR/src/mtools/visit_tool.py" "$VERL_SRC_DIR/tools/visit_tool.py"

python3 "$ROOT_DIR/src/mtools/prepare_mtool_data.py" --input "$RAW_VAL_DATA" --output "$VAL_DATA"

export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES="$EVAL_VISIBLE_DEVICES" python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    algorithm.adv_estimator=grpo \
    data.shuffle=false \
    data.seed=2026 \
    data.train_batch_size=256 \
    data.val_batch_size=8 \
    data.max_prompt_length=8192 \
    data.max_response_length=110000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=122880 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=120 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=120 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
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
    trainer.project_name='mtools_eval' \
    trainer.experiment_name='qwen3-4b-mtools-browsecomp-eval' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$CHECKPOINT_PATH" \
    trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
    "$@"
