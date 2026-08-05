# Copyright 2026
"""Standalone main-agent evaluation entrypoint.

This is a lightweight alternative to ``verl.trainer.main_ppo`` for validation-only
multi-turn/tool-use evaluation.  It intentionally does NOT initialize actor FSDP,
reference policy, critic, optimizer, or checkpoint update/sync.  Instead it starts
standalone rollout servers (SGLang/vLLM via AgentLoopManager) and runs the same
AgentLoop validation path over ``data.val_files``.

Typical use:

    python3 -m verl.trainer.main_agent_eval \
      --config-path=/path/to/config --config-name=search_multiturn_grpo \
      actor_rollout_ref.model.path=/path/to/hf_model \
      actor_rollout_ref.rollout.name=sglang \
      actor_rollout_ref.rollout.nnodes=1 \
      actor_rollout_ref.rollout.n_gpus_per_node=4 \
      actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
      actor_rollout_ref.rollout.load_format=auto \
      trainer.validation_data_dir=./validation_agent_eval
"""

from __future__ import annotations

import json
import os
import socket
import uuid
from collections import defaultdict
from pprint import pprint

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.reward_loop import RewardLoopManager, migrate_legacy_reward_impl
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import create_rl_dataset
from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.trainer.ppo.reward import extract_reward
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.device import auto_set_device, is_cuda_available
from verl.utils.fs import copy_to_local


def _init_ray(config):
    if ray.is_initialized():
        return
    default_runtime_env = get_ppo_ray_runtime_env()
    ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
    runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
    # AgentLoop instances live in Ray workers. Explicitly forward opt-in
    # controller variables so behavior does not depend on whether the local Ray
    # runtime was started before or after the shell exported them.
    agent_loop_env = {
        key: value for key, value in os.environ.items() if key.startswith("VERL_TOOL_AGENT_")
    }
    runtime_env = OmegaConf.merge(
        default_runtime_env,
        {"env_vars": agent_loop_env},
        runtime_env_kwargs,
    )
    ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
    print(f"ray init kwargs: {ray_init_kwargs}")
    ray.init(**OmegaConf.to_container(ray_init_kwargs))


def _ensure_standalone_rollout_config(config):
    """Make the existing PPO config safe for standalone validation.

    AgentLoopManager(worker_group=None) requires rollout.nnodes > 0.  The normal
    PPO config often leaves rollout.nnodes=0 because hybrid rollout uses trainer
    worker_group.  For direct infer we default it to trainer.nnodes and use the
    trainer GPU count unless explicitly overridden.
    """
    with open_dict(config):
        rollout = config.actor_rollout_ref.rollout
        if rollout.get("nnodes", 0) <= 0:
            rollout.nnodes = config.trainer.nnodes
        if rollout.get("n_gpus_per_node", None) in (None, 0):
            rollout.n_gpus_per_node = config.trainer.n_gpus_per_node
        # In standalone eval the server should load real HF weights itself.
        # SGLangHttpServer also auto-corrects dummy->auto for non-HYBRID, but
        # setting it here makes logs/config explicit.
        if rollout.get("load_format", "dummy") == "dummy":
            rollout.load_format = "auto"
        # Avoid sleep/resume paths in eval-only direct infer.
        rollout.free_cache_engine = False
        # This entrypoint only validates.
        config.trainer.val_only = True
        config.trainer.val_before_train = True


def _get_gen_batch(batch: DataProto) -> DataProto:
    reward_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()
    non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
    gen_batch = batch.pop(batch_keys=[], non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop))
    gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
    return gen_batch


_TRAJECTORY_TELEMETRY_KEYS = (
    "duplicate_tool_calls",
    "max_duplicate_only_streak",
    "final_answer_attempts",
    "finalization_reason",
    "finalization_exhausted",
    "finalization_rejected_tool_calls",
    "finalization_balance_failures",
    "finalization_tool_compactions",
    "generic_answer_rescue_enabled",
    "generic_answer_rescue_prefill",
    "generic_answer_rescue_triggers",
    "generic_answer_rescue_rejections",
    "generic_answer_rescue_missing_rejections",
    "generic_answer_rescue_generic_rejections",
    "generic_answer_rescue_successes",
    "generic_answer_rescue_exhausted",
    "generic_answer_rescue_events",
    "final_answer_prefill_enabled",
    "finalization_structure_closure_enabled",
    "search_calls",
    "open_calls",
    "visit_calls",
    "find_calls",
    "search_doc_refs",
    "searched_docids",
    "opened_docids",
    "find_docids",
    "document_flow",
    "unique_search_doc_ratio",
    "search_result_overlap_events",
    "max_search_result_overlap",
    "search_frontier_enabled",
    "search_frontier_requests",
    "search_frontier_base_pages",
    "search_frontier_unseen_pages",
    "search_frontier_exhausted_calls",
    "search_frontier_selected_docids",
    "search_frontier_events",
    "frontier_open_first_enabled",
    "frontier_open_first_prompt_limit",
    "frontier_open_first_escalate_after",
    "frontier_open_first_block_limit",
    "frontier_open_first_active",
    "frontier_open_first_pending_docids",
    "frontier_open_first_ignore_streak",
    "frontier_open_first_triggers",
    "frontier_open_first_prompts",
    "frontier_open_first_ignored_unseen_pages",
    "frontier_open_first_blocked_calls",
    "frontier_open_first_compliances",
    "frontier_open_first_pending_compliances",
    "frontier_open_first_other_inspection_releases",
    "frontier_open_first_novel_search_releases",
    "frontier_open_first_suggested_docids",
    "frontier_open_first_events",
    "scaffold_v2_enabled",
    "scaffold_v2_mode",
    "search_overlap_threshold",
    "search_only_turn_limit",
    "search_to_open_prompt_limit",
    "consecutive_search_only_turns",
    "max_consecutive_search_only_turns",
    "last_search_turn_overlap",
    "search_to_open_required",
    "search_overlap_notices",
    "search_to_open_triggers",
    "search_to_open_interventions",
    "search_to_open_blocked_calls",
    "search_to_open_compliances",
    "search_to_open_soft_releases",
    "search_to_open_suggested_docids",
    "search_to_open_events",
    "verification_checkpoint_enabled",
    "verification_checkpoint_exhausted_turns",
    "verification_checkpoint_search_only_turns",
    "verification_checkpoint_search_overlap_threshold",
    "verification_checkpoint_duplicate_streak",
    "verification_checkpoint_triggered",
    "verification_checkpoint_pending",
    "verification_checkpoint_reason",
    "verification_checkpoint_triggers",
    "verification_checkpoint_responses",
    "verification_checkpoint_structured_responses",
    "verification_checkpoint_soft_releases",
    "verification_checkpoint_consecutive_exhausted_turns",
    "verification_checkpoint_consecutive_search_only_turns",
    "verification_checkpoint_last_search_overlap",
    "verification_checkpoint_max_search_overlap",
    "verification_checkpoint_suggested_docids",
    "verification_checkpoint_action_counts",
    "verification_checkpoint_events",
)


def _extract_sample_identifiers(extra_infos, batch_size):
    """Extract stable dataset identifiers without feeding them into metric aggregation."""
    identifiers = {"query_id": [], "dataset_index": [], "question": []}
    if isinstance(extra_infos, np.ndarray):
        extra_infos = extra_infos.tolist()
    elif not isinstance(extra_infos, list):
        extra_infos = [extra_infos] * batch_size
    if len(extra_infos) < batch_size:
        extra_infos = [*extra_infos, *([None] * (batch_size - len(extra_infos)))]
    elif len(extra_infos) > batch_size:
        extra_infos = extra_infos[:batch_size]

    for item in extra_infos:
        info = item if isinstance(item, dict) else {}
        query_id = info.get("query_id")
        dataset_index = info.get("index")
        question = info.get("question")
        identifiers["query_id"].append(None if query_id is None else str(query_id))
        if isinstance(dataset_index, np.generic):
            dataset_index = dataset_index.item()
        identifiers["dataset_index"].append(dataset_index)
        identifiers["question"].append(None if question is None else str(question))
    return identifiers


def _dump_generations(
    inputs,
    outputs,
    gts,
    scores,
    reward_extra_infos_dict,
    dump_path,
    global_steps=0,
    trajectory_extra_infos_dict=None,
):
    os.makedirs(dump_path, exist_ok=True)
    filename = os.path.join(dump_path, f"{global_steps}.jsonl")
    n = len(inputs)
    base_data = {
        "input": inputs,
        "output": outputs,
        "gts": gts,
        "score": scores,
        "step": [global_steps] * n,
    }
    for k, v in reward_extra_infos_dict.items():
        if len(v) == n:
            base_data[k] = v
    for k, v in (trajectory_extra_infos_dict or {}).items():
        if len(v) == n:
            base_data[k] = v

    with open(filename, "w", encoding="utf-8") as f:
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Dumped generations to {filename}")


def _val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
    data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
    metric_dict = {}
    for data_source, var2metric2val in data_src2var2metric2val.items():
        core_var = "acc" if "acc" in var2metric2val else "reward"
        for var_name, metric2val in var2metric2val.items():
            n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
            for metric_name, metric_val in metric2val.items():
                if (
                    (var_name == core_var)
                    and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                    and (f"@{n_max}" in metric_name)
                ):
                    metric_sec = "val-core"
                else:
                    metric_sec = "val-aux"
                metric_dict[f"{metric_sec}/{data_source}/{var_name}/{metric_name}"] = metric_val

    if len(sample_turns) > 0:
        sample_turns = np.concatenate(sample_turns)
        metric_dict["val-aux/num_turns/min"] = sample_turns.min()
        metric_dict["val-aux/num_turns/max"] = sample_turns.max()
        metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()
    return metric_dict


def _validate(config, tokenizer, val_dataloader, async_rollout_manager, reward_loop_manager=None, global_steps=0):
    data_source_lst = []
    reward_extra_infos_dict: dict[str, list] = defaultdict(list)
    trajectory_extra_infos_dict: dict[str, list] = defaultdict(list)
    sample_inputs, sample_outputs, sample_gts, sample_scores, sample_turns, sample_uids = [], [], [], [], [], []

    for test_data in val_dataloader:
        test_batch = DataProto.from_single_dict(test_data)
        if "uid" not in test_batch.non_tensor_batch:
            test_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
            )

        test_batch = test_batch.repeat(repeat_times=config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)
        sample_gts.extend([
            item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
        ])

        test_gen_batch = _get_gen_batch(test_batch)
        test_gen_batch.meta_info = {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            "validate": True,
            "global_steps": global_steps,
        }
        print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

        size_divisor = config.actor_rollout_ref.rollout.agent.num_workers
        test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
        test_output_gen_batch_padded = async_rollout_manager.generate_sequences(test_gen_batch_padded)
        test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
        if "rm_scores" not in test_output_gen_batch.batch.keys():
            if reward_loop_manager is None:
                raise RuntimeError(
                    "AgentLoop output has no rm_scores and no reward_loop_manager is available. "
                    "Pass RewardLoopManager workers to AgentLoopManager or compute reward after generation."
                )
            print("AgentLoop output has no rm_scores; computing reward with RewardLoopManager fallback")
            batch_reward = reward_loop_manager.compute_rm_score(test_output_gen_batch)
            test_output_gen_batch = test_output_gen_batch.union(batch_reward)
        print("validation generation end")

        output_ids = test_output_gen_batch.batch["responses"]
        output_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        sample_outputs.extend(output_texts)

        test_batch = test_batch.union(test_output_gen_batch)
        test_batch.meta_info["validate"] = True

        batch_size = len(test_batch)
        sample_identifiers = _extract_sample_identifiers(
            test_batch.non_tensor_batch.get("extra_info"), batch_size
        )
        for key, values in sample_identifiers.items():
            trajectory_extra_infos_dict[key].extend(values)
        for key in _TRAJECTORY_TELEMETRY_KEYS:
            values = test_batch.non_tensor_batch.get(key)
            if values is None:
                trajectory_extra_infos_dict[key].extend([None] * batch_size)
            elif isinstance(values, np.ndarray):
                trajectory_extra_infos_dict[key].extend(values.tolist())
            elif isinstance(values, list):
                trajectory_extra_infos_dict[key].extend(values)
            else:
                trajectory_extra_infos_dict[key].extend([values] * batch_size)

        input_ids = test_batch.batch["prompts"]
        input_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        sample_inputs.extend(input_texts)
        sample_uids.extend(test_batch.non_tensor_batch["uid"])

        reward_tensor, reward_extra_info = extract_reward(test_batch)
        scores = reward_tensor.sum(-1).cpu().tolist()
        sample_scores.extend(scores)

        reward_extra_infos_dict["reward"].extend(scores)
        for key, values in reward_extra_info.items():
            if key not in reward_extra_infos_dict:
                reward_extra_infos_dict[key] = []
            if isinstance(values, np.ndarray):
                reward_extra_infos_dict[key].extend(values.tolist())
            else:
                reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

        if "__num_turns__" in test_batch.non_tensor_batch:
            sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])
        data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

    val_data_dir = config.trainer.get("validation_data_dir", None)
    if val_data_dir:
        _dump_generations(
            inputs=sample_inputs,
            outputs=sample_outputs,
            gts=sample_gts,
            scores=sample_scores,
            reward_extra_infos_dict=reward_extra_infos_dict,
            dump_path=val_data_dir,
            global_steps=global_steps,
            trajectory_extra_infos_dict=trajectory_extra_infos_dict,
        )

    for key_info, lst in reward_extra_infos_dict.items():
        assert len(lst) == 0 or len(lst) == len(sample_scores), (
            f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
        )

    data_sources = np.concatenate(data_source_lst, axis=0)
    return _val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)


@ray.remote(num_cpus=1)
class StandaloneAgentEvalRunner:
    def run(self, config):
        print(f"StandaloneAgentEvalRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        _ensure_standalone_rollout_config(config)
        OmegaConf.resolve(config)
        pprint(OmegaConf.to_container(config, resolve=True))

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        val_batch_size = config.data.val_batch_size if config.data.val_batch_size is not None else len(val_dataset)
        val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            num_workers=config.data["dataloader_num_workers"],
            shuffle=config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )
        assert len(val_dataloader) >= 1, "Validation dataloader is empty!"
        print(f"Size of val dataloader: {len(val_dataloader)}")

        # Keep the same rule/custom reward path as RayPPOTrainer validation.
        # This does not initialize actor/ref/critic; it only creates CPU reward loop
        # workers, or a reward-model server if the config explicitly enables one.
        reward_loop_manager = RewardLoopManager(config=config, rm_resource_pool=None)

        # This is the key difference from main_ppo: no actor/ref/critic worker_group.
        # worker_group=None makes AgentLoopManager launch standalone rollout servers.
        async_rollout_manager = AgentLoopManager.create(
            config=config,
            worker_group=None,
            rollout_resource_pool=None,
            reward_loop_worker_handles=reward_loop_manager.reward_loop_workers,
        )
        metrics = _validate(
            config,
            tokenizer,
            val_dataloader,
            async_rollout_manager,
            reward_loop_manager=reward_loop_manager,
            global_steps=0,
        )
        pprint(metrics)
        return metrics


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    _init_ray(config)

    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = StandaloneAgentEvalRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = StandaloneAgentEvalRunner.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


if __name__ == "__main__":
    main()
