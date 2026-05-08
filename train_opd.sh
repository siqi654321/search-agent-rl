#!/usr/bin/env bash
set -xeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
VERL_ROOT="${VERL_ROOT:-$ROOT_DIR/verl}"
VERL_SRC_DIR="${VERL_SRC_DIR:-$VERL_ROOT/verl}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/src/config}"

INDEX_PATH="${INDEX_PATH:-$ROOT_DIR/data/e5_Flat.index}"
CORPUS_PATH="${CORPUS_PATH:-$ROOT_DIR/data/wiki-18.jsonl}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-$ROOT_DIR/models/e5-base-v2}"
SGLANG_MODEL_PATH="${SGLANG_MODEL_PATH:-$ROOT_DIR/models/Qwen3-1.7B}"
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-$ROOT_DIR/models/Qwen3-4B}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-$ROOT_DIR/models/Qwen3-8B}"

TRAIN_DATA="${TRAIN_DATA:-$ROOT_DIR/data/asearcher_searchr1/train.parquet}"
VAL_DATA="${VAL_DATA:-$ROOT_DIR/data/asearcher_searchr1/test.parquet}"
TOOL_CONFIG="${TOOL_CONFIG:-$CONFIG_PATH/tool_config/search_tool_config.yaml}"

PROJECT_NAME="${PROJECT_NAME:-search_r1_like_opd_rl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-4b-asearcher-opd-from-8b}"
CKPTS_DIR="${CKPTS_DIR:-$ROOT_DIR/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME}"

RETRIEVAL_VISIBLE_DEVICES="${RETRIEVAL_VISIBLE_DEVICES:-0,1,2,3}"
SGLANG_VISIBLE_DEVICES="${SGLANG_VISIBLE_DEVICES:-1}"
TRAIN_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES:-4,5,6,7}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-35000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40000}"

PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"

MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-2048}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-100}"
MAX_USER_TURNS="${MAX_USER_TURNS:-100}"
MULTI_TURN_FORMAT="${MULTI_TURN_FORMAT:-hermes}"

NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-50}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"

OPD_ENABLE="${OPD_ENABLE:-true}"
OPD_KL_COEF="${OPD_KL_COEF:-0.05}"
OPD_PURE_DISTILL="${OPD_PURE_DISTILL:-false}"
TEACHER_REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${TEACHER_REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
TEACHER_REF_LOGPROB_MAX_TOKEN_LEN_PER_GPU="${TEACHER_REF_LOGPROB_MAX_TOKEN_LEN_PER_GPU:-39096}"
TEACHER_REF_LOGPROB_USE_DYNAMIC_BSZ="${TEACHER_REF_LOGPROB_USE_DYNAMIC_BSZ:-false}"
TEACHER_PARAM_OFFLOAD="${TEACHER_PARAM_OFFLOAD:-true}"
TEACHER_ULYSSES_SEQUENCE_PARALLEL_SIZE="${TEACHER_ULYSSES_SEQUENCE_PARALLEL_SIZE:-2}"

CUDA_VISIBLE_DEVICES="$RETRIEVAL_VISIBLE_DEVICES" nohup python3 "$ROOT_DIR/retrieval/retrieval_server_sglang_summarize.py" \
    --index_path "$INDEX_PATH" \
    --corpus_path "$CORPUS_PATH" \
    --faiss_gpu \
    --retriever_name e5 \
    --retriever_model "$RETRIEVER_MODEL" \
    --sglang_base_url http://127.0.0.1:30000 \
    --sglang_model qwen3-1.7b \
    --host 0.0.0.0 \
    --port 1249 >"$ROOT_DIR/retrieve.out" &

echo "等待服务启动..."
while ! nc -z 127.0.0.1 1249; do
    sleep 0.5
done

CUDA_VISIBLE_DEVICES="$SGLANG_VISIBLE_DEVICES" nohup python -m sglang.launch_server \
    --model-path "$SGLANG_MODEL_PATH" \
    --tensor-parallel-size=1 \
    --mem-fraction-static 0.5 \
    --port 30000 >"$ROOT_DIR/qwen3_1.7b.out" &

while ! nc -z 127.0.0.1 30000; do
    sleep 0.5
done

echo "服务已就绪，开始执行 OPD 训练"

cp "$ROOT_DIR/src/tool_parser.py" "$VERL_SRC_DIR/experimental/agent_loop/tool_parser.py"
cp "$ROOT_DIR/src/search_tool.py" "$VERL_SRC_DIR/tools/search_tool.py"
cp "$ROOT_DIR/src/search_r1_like_utils.py" "$VERL_SRC_DIR/tools/utils/search_r1_like_utils.py"
cp "$ROOT_DIR/src/search_r1_like_qa_em.py" "$VERL_SRC_DIR/utils/reward_score/search_r1_like_qa_em.py"
cp "$ROOT_DIR/src/__init__.py" "$VERL_SRC_DIR/utils/reward_score/__init__.py"
cp "$ROOT_DIR/src/opd/fsdp_workers.py" "$VERL_SRC_DIR/workers/fsdp_workers.py"
cp "$ROOT_DIR/src/opd/main_ppo.py" "$VERL_SRC_DIR/trainer/main_ppo.py"
cp "$ROOT_DIR/src/opd/ray_trainer.py" "$VERL_SRC_DIR/trainer/ppo/ray_trainer.py"
cp "$ROOT_DIR/src/opd/utils.py" "$VERL_SRC_DIR/trainer/ppo/utils.py"

export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES="$TRAIN_VISIBLE_DEVICES" python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.shuffle=true \
    data.seed=2026 \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
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
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=$MAX_TOOL_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_ASSISTANT_TURNS \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_USER_TURNS \
    actor_rollout_ref.rollout.multi_turn.format=$MULTI_TURN_FORMAT \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
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
    trainer.nnodes=$NNODES \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.default_local_dir="$CKPTS_DIR" \
    trainer.total_epochs=$TOTAL_EPOCHS \
    +on_policy_distill.enable=$OPD_ENABLE \
    +on_policy_distill.kl_coef=$OPD_KL_COEF \
    +on_policy_distill.pure_distill=$OPD_PURE_DISTILL \
    +on_policy_distill.teacher.model.path="$TEACHER_MODEL_PATH" \
    +on_policy_distill.teacher.ref.log_prob_micro_batch_size_per_gpu=$TEACHER_REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    +on_policy_distill.teacher.ref.log_prob_max_token_len_per_gpu=$TEACHER_REF_LOGPROB_MAX_TOKEN_LEN_PER_GPU \
    +on_policy_distill.teacher.ref.log_prob_use_dynamic_bsz=$TEACHER_REF_LOGPROB_USE_DYNAMIC_BSZ \
    +on_policy_distill.teacher.ref.fsdp_config.param_offload=$TEACHER_PARAM_OFFLOAD \
    +on_policy_distill.teacher.actor.ulysses_sequence_parallel_size=$TEACHER_ULYSSES_SEQUENCE_PARALLEL_SIZE
