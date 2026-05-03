#!/usr/bin/env bash

ROOT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
INDEX_PATH="${INDEX_PATH:-$ROOT_DIR/data/e5_Flat.index}"
CORPUS_PATH="${CORPUS_PATH:-$ROOT_DIR/data/wiki-18.jsonl}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-$ROOT_DIR/models/e5-base-v2}"
SGLANG_MODEL_PATH="${SGLANG_MODEL_PATH:-$ROOT_DIR/models/Qwen3-1.7B}"
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-$ROOT_DIR/models/Qwen3-8B}"

CUDA_VISIBLE_DEVICES="0,1,2,3" nohup python3 "$ROOT_DIR/retrieval/retrieval_server_sglang_summarize.py" --index_path "$INDEX_PATH" --corpus_path "$CORPUS_PATH" --faiss_gpu --retriever_name e5 --retriever_model "$RETRIEVER_MODEL" --sglang_base_url http://127.0.0.1:30000 --sglang_model qwen3-1.7b --host 0.0.0.0 --port 1249 >"$ROOT_DIR/retrieve.out" &

echo "等待服务启动..."
while ! nc -z 127.0.0.1 1249; do   
  sleep 0.5 # 每隔0.5秒检查一次
done

CUDA_VISIBLE_DEVICES=1 nohup python -m sglang.launch_server --model-path "$SGLANG_MODEL_PATH" --tensor-parallel-size=1 --mem-fraction-static 0.5 --port 30000 >"$ROOT_DIR/qwen3_1.7b.out" &
while ! nc -z 127.0.0.1 30000; do   
  sleep 0.5 # 每隔0.5秒检查一次
done

echo "服务已就绪，开始执行后续程序"
CONFIG_PATH="$ROOT_DIR/src/config"
cp "$ROOT_DIR/src/tool_parser.py" "$ROOT_DIR/verl/experimental/agent_loop/tool_parser.py"
cp "$ROOT_DIR/src/search_tool.py" "$ROOT_DIR/verl/tools/search_tool.py"
cp "$ROOT_DIR/src/search_r1_like_utils.py" "$ROOT_DIR/verl/tools/utils/search_r1_like_utils.py"
cp "$ROOT_DIR/src/search_r1_like_qa_em.py"  "$ROOT_DIR/verl/utils/reward_score/search_r1_like_qa_em.py"
cp "$ROOT_DIR/src/__init__.py" "$ROOT_DIR/verl/utils/reward_score/__init__.py"
TRAIN_DATA="${TRAIN_DATA:-$ROOT_DIR/data/asearcher_searchr1/train.parquet}"
VAL_DATA="${VAL_DATA:-$ROOT_DIR/data/asearcher_searchr1/test.parquet}"

TOOL_CONFIG="${TOOL_CONFIG:-$CONFIG_PATH/tool_config/search_tool_config.yaml}"



CUDA_VISIBLE_DEVICES="4,5,6,7" python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.shuffle=true \
    data.seed=2026 \
    data.train_batch_size=128 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=35000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2  \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.max_model_len=40000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=100 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=100 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='search_r1_like_async_rl' \
    trainer.experiment_name='qwen3-8b-asearcher-datarand-flashattn' \
    trainer.n_gpus_per_node=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA"  \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    algorithm.rollout_correction.bypass_mode=false \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize=false \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    trainer.total_epochs=3
