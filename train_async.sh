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
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-$ROOT_DIR/models/Qwen3-8B}"

TRAIN_DATA="${TRAIN_DATA:-$ROOT_DIR/data/asearcher_searchr1/train.parquet}"
VAL_DATA="${VAL_DATA:-$ROOT_DIR/data/asearcher_searchr1/test.parquet}"
TOOL_CONFIG="${TOOL_CONFIG:-$CONFIG_PATH/tool_config/search_tool_config.yaml}"
MODEL_PATH="${MODEL_PATH:-$ACTOR_MODEL_PATH}"
CKPTS_DIR="${CKPTS_DIR:-$ROOT_DIR/checkpoints}"

PROJECT_NAME="${PROJECT_NAME:-search_r1_like_async_rl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-8b-asearcher-tis-datarand-token-mis-flashattn-full-async-staleness0.5}"

ROLLOUT_NAME="${ROLLOUT_NAME:-sglang}"
ROLLOUT_MODE="${ROLLOUT_MODE:-async}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"

if [ "$ROLLOUT_NAME" = "vllm" ] && [ "$ROLLOUT_MODE" = "async" ]; then
    export VLLM_USE_V1=1
fi

NNODES_TRAIN="${NNODES_TRAIN:-1}"
NNODES_ROLLOUT="${NNODES_ROLLOUT:-1}"
TRAIN_GPUS_PER_NODE="${TRAIN_GPUS_PER_NODE:-4}"
ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-2}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-35000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40000}"
TOTAL_SEQ_LEN="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$TOTAL_SEQ_LEN}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-2048}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-100}"
MAX_USER_TURNS="${MAX_USER_TURNS:-100}"
MULTI_TURN_FORMAT="${MULTI_TURN_FORMAT:-hermes}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-0}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-$TOTAL_SEQ_LEN}"
ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU="${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU:-$TOTAL_SEQ_LEN}"
REF_LOGPROB_MAX_TOKEN_LEN_PER_GPU="${REF_LOGPROB_MAX_TOKEN_LEN_PER_GPU:-$TOTAL_SEQ_LEN}"
ROLLOUT_N="${ROLLOUT_N:-8}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-100000000}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-50}"

STALENESS_THRESHOLD="${STALENESS_THRESHOLD:-0.5}"
TRIGGER_PARAMETER_SYNC_STEP="${TRIGGER_PARAMETER_SYNC_STEP:-1}"
REQUIRE_BATCHES="${REQUIRE_BATCHES:-4}"
PARTIAL_ROLLOUT="${PARTIAL_ROLLOUT:-false}"
PARTIAL_ROLLOUT_RESUME="${PARTIAL_ROLLOUT_RESUME:-false}"
SYNC_ALIGNED_GROUPING="${SYNC_ALIGNED_GROUPING:-false}"
AGENT_LOOP_NUM_WORKERS="${AGENT_LOOP_NUM_WORKERS:-4}"
MAX_CONCURRENT_SAMPLES="${MAX_CONCURRENT_SAMPLES:-192}"

CUDA_VISIBLE_DEVICES="0,1,2,3" nohup python3 "$ROOT_DIR/retrieval/retrieval_server_sglang_summarize.py" \
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

CUDA_VISIBLE_DEVICES=1 nohup python -m sglang.launch_server \
    --model-path "$SGLANG_MODEL_PATH" \
    --tensor-parallel-size=1 \
    --mem-fraction-static 0.5 \
    --port 30000 >"$ROOT_DIR/qwen3_1.7b.out" &

while ! nc -z 127.0.0.1 30000; do
    sleep 0.5
done

echo "服务已就绪，开始执行后续程序"

cp "$ROOT_DIR/src/tool_parser.py" "$VERL_SRC_DIR/experimental/agent_loop/tool_parser.py"
cp "$ROOT_DIR/src/search_tool.py" "$VERL_SRC_DIR/tools/search_tool.py"
cp "$ROOT_DIR/src/search_r1_like_utils.py" "$VERL_SRC_DIR/tools/utils/search_r1_like_utils.py"
cp "$ROOT_DIR/src/search_r1_like_qa_em.py" "$VERL_SRC_DIR/utils/reward_score/search_r1_like_qa_em.py"
cp "$ROOT_DIR/src/__init__.py" "$VERL_SRC_DIR/utils/reward_score/__init__.py"

cp "$ROOT_DIR/src/async/fully_async_main.py" "$VERL_SRC_DIR/experimental/fully_async_policy/fully_async_main.py"
cp "$ROOT_DIR/src/async/param_sync.py" "$VERL_SRC_DIR/experimental/fully_async_policy/param_sync.py"
cp "$ROOT_DIR/src/async/sglang_async_server.py" "$VERL_SRC_DIR/experimental/fully_async_policy/sglang_rollout/sglang_async_server.py"
cp "$ROOT_DIR/src/async/verl__experimental__fully_async_policy__agent_loop__agent_loop.py" "$VERL_SRC_DIR/experimental/fully_async_policy/agent_loop/agent_loop.py"
cp "$ROOT_DIR/src/async/fully_async_rollouter.py" "$VERL_SRC_DIR/experimental/fully_async_policy/fully_async_rollouter.py"
cp "$ROOT_DIR/src/async/fully_async_trainer.py" "$VERL_SRC_DIR/experimental/fully_async_policy/fully_async_trainer.py"
cp "$ROOT_DIR/src/async/replica.py" "$VERL_SRC_DIR/workers/rollout/replica.py"
cp "$ROOT_DIR/src/async/async_sglang_server.py" "$VERL_SRC_DIR/workers/rollout/sglang_rollout/async_sglang_server.py"
cp "$ROOT_DIR/src/async/tool_agent_loop.py" "$VERL_SRC_DIR/experimental/agent_loop/tool_agent_loop.py"
cp "$ROOT_DIR/src/async/single_turn_agent_loop.py" "$VERL_SRC_DIR/experimental/agent_loop/single_turn_agent_loop.py"
cp "$ROOT_DIR/src/async/partial_tool_agent_loop.py" "$VERL_SRC_DIR/experimental/fully_async_policy/agent_loop/partial_tool_agent_loop.py"
cp "$ROOT_DIR/src/async/partial_single_turn_agent_loop.py" "$VERL_SRC_DIR/experimental/fully_async_policy/agent_loop/partial_single_turn_agent_loop.py"
cp "$ROOT_DIR/src/async/verl__experimental__agent_loop__agent_loop.py" "$VERL_SRC_DIR/experimental/agent_loop/agent_loop.py"
cp "$ROOT_DIR/src/async/metric_utils.py" "$VERL_SRC_DIR/trainer/ppo/metric_utils.py"
cp "$ROOT_DIR/src/async/detach_utils.py" "$VERL_SRC_DIR/experimental/fully_async_policy/detach_utils.py"
cp "$ROOT_DIR/src/async/fully_async_ppo_trainer.yaml" "$VERL_SRC_DIR/experimental/fully_async_policy/config/fully_async_ppo_trainer.yaml"
cp "$ROOT_DIR/src/async/verl__trainer__ppo__ray_trainer.py" "$VERL_SRC_DIR/trainer/ppo/ray_trainer.py"
cp "$ROOT_DIR/src/async/verl__experimental__separation__ray_trainer.py" "$VERL_SRC_DIR/experimental/separation/ray_trainer.py"

export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

rm -rf "$CKPTS_DIR"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python3 -m verl.experimental.fully_async_policy.fully_async_main \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.prompt_key=prompt \
    data.shuffle=true \
    data.seed=2026 \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.gen_batch_size=$GEN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.val_batch_size=256 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME \
    actor_rollout_ref.rollout.mode=$ROLLOUT_MODE \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1280 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=$MAX_TOOL_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_ASSISTANT_TURNS \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_USER_TURNS \
    actor_rollout_ref.rollout.multi_turn.format=$MULTI_TURN_FORMAT \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$REF_LOGPROB_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=false \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize=false \
    algorithm.rollout_correction.rollout_rs=token_k1 \
    algorithm.rollout_correction.rollout_rs_threshold=0.5_2.0 \
    actor_rollout_ref.rollout.agent.num_workers=$AGENT_LOOP_NUM_WORKERS \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.save_freq=$SAVE_FREQ \
    trainer.default_local_dir="$CKPTS_DIR" \
    trainer.resume_mode=auto \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.nnodes=$NNODES_TRAIN \
    trainer.n_gpus_per_node=$TRAIN_GPUS_PER_NODE \
    rollout.nnodes=$NNODES_ROLLOUT \
    rollout.n_gpus_per_node=$ROLLOUT_GPUS_PER_NODE \
    rollout.total_rollout_steps=$TOTAL_ROLLOUT_STEPS \
    rollout.total_epochs=$TOTAL_EPOCHS \
    rollout.test_freq=$TEST_FREQ \
    async_training.staleness_threshold=$STALENESS_THRESHOLD \
    async_training.trigger_parameter_sync_step=$TRIGGER_PARAMETER_SYNC_STEP \
    async_training.require_batches=$REQUIRE_BATCHES \
    async_training.sync_aligned_grouping=$SYNC_ALIGNED_GROUPING \
    async_training.max_concurrent_samples=$MAX_CONCURRENT_SAMPLES \
    async_training.partial_rollout=$PARTIAL_ROLLOUT \
    async_training.partial_rollout_resume=$PARTIAL_ROLLOUT_RESUME

sleep infinity
