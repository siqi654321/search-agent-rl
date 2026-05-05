#!/usr/bin/env python3
import argparse
import atexit
import math
import os
import subprocess
from pathlib import Path

import ray
import yaml
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy

from verl.service_actors import RetrievalServiceActor, SummarySGLangActor


DEFAULT_WORKING_DIR = Path(os.environ.get("WORKING_DIR", os.getcwd())).resolve()
DEFAULT_CONFIG_PATH = str(DEFAULT_WORKING_DIR / "src" / "config")
DEFAULT_TOOL_CONFIG_TEMPLATE = str(DEFAULT_WORKING_DIR / "src" / "config" / "tool_config" / "search_tool_config.yaml")
DEFAULT_TRAIN_DATA = str(DEFAULT_WORKING_DIR / "data" / "asearcher_searchr1" / "train.parquet")
DEFAULT_VAL_DATA = str(DEFAULT_WORKING_DIR / "data" / "asearcher_searchr1" / "test.parquet")
DEFAULT_SEARCH_SCRIPT = str(DEFAULT_WORKING_DIR / "retrieval" / "retrieval_server_sglang_summarize.py")
DEFAULT_INDEX_PATH = str(DEFAULT_WORKING_DIR / "data" / "e5_Flat.index")
DEFAULT_CORPUS_PATH = str(DEFAULT_WORKING_DIR / "data" / "wiki-18.jsonl")
DEFAULT_RETRIEVER_MODEL = str(DEFAULT_WORKING_DIR / "models" / "e5-base-v2")
DEFAULT_SUMMARY_MODEL_PATH = str(DEFAULT_WORKING_DIR / "models" / "Qwen3-1.7B")
DEFAULT_ACTOR_MODEL_PATH = str(DEFAULT_WORKING_DIR / "models" / "Qwen3-8B")
DEFAULT_TRAINING_SCRIPT = str(DEFAULT_WORKING_DIR / "multinode" / "train_verl_sync.sh")


SINGLE_NODE_8GPU_LAYOUT = {
    "summary": "0",
    "retrieval": "1,2,3",
    "train": "4,5,6,7",
}
DEFAULT_LOCAL_RETRIEVAL_URL = "http://127.0.0.1:1249/retrieve_summarize_compat"


def _parse_args():
    parser = argparse.ArgumentParser(description="Launch sync verl PPO with Ray-managed retrieval and sglang services")
    parser.add_argument("--address", default=os.environ.get("RAY_ADDRESS", "auto"))
    parser.add_argument("--working-dir", default=os.environ.get("WORKING_DIR", os.getcwd()))
    parser.add_argument("--config-path", default=os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH))
    parser.add_argument("--tool-config-template", default=os.environ.get("TOOL_CONFIG_TEMPLATE", DEFAULT_TOOL_CONFIG_TEMPLATE))
    parser.add_argument("--train-data", default=os.environ.get("TRAIN_DATA", DEFAULT_TRAIN_DATA))
    parser.add_argument("--val-data", default=os.environ.get("VAL_DATA", DEFAULT_VAL_DATA))
    parser.add_argument("--search-script", default=os.environ.get("RETRIEVAL_SCRIPT", DEFAULT_SEARCH_SCRIPT))
    parser.add_argument("--index-path", default=os.environ.get("INDEX_PATH", DEFAULT_INDEX_PATH))
    parser.add_argument("--corpus-path", default=os.environ.get("CORPUS_PATH", DEFAULT_CORPUS_PATH))
    parser.add_argument("--retriever-name", default=os.environ.get("RETRIEVER_NAME", "e5"))
    parser.add_argument("--retriever-model", default=os.environ.get("RETRIEVER_MODEL", DEFAULT_RETRIEVER_MODEL))
    parser.add_argument("--summary-model-path", default=os.environ.get("SUMMARY_MODEL_PATH", DEFAULT_SUMMARY_MODEL_PATH))
    parser.add_argument("--summary-model-name", default=os.environ.get("SUMMARY_MODEL_NAME", "default"))
    parser.add_argument("--summary-tp", type=int, default=int(os.environ.get("SUMMARY_TP", "1")))
    parser.add_argument("--summary-mem-fraction", type=float, default=float(os.environ.get("SUMMARY_MEM_FRACTION", "0.5")))
    parser.add_argument("--summary-port", type=int, default=int(os.environ.get("SUMMARY_PORT", "30000")))
    parser.add_argument("--retrieval-port", type=int, default=int(os.environ.get("RETRIEVAL_PORT", "1249")))
    parser.add_argument("--summary-startup-timeout-s", type=int, default=int(os.environ.get("SUMMARY_STARTUP_TIMEOUT_S", "300")))
    parser.add_argument("--retrieval-startup-timeout-s", type=int, default=int(os.environ.get("RETRIEVAL_STARTUP_TIMEOUT_S", "900")))
    parser.add_argument("--retrieval-num-gpus", type=float, default=float(os.environ.get("RETRIEVAL_NUM_GPUS", "1")))
    parser.add_argument("--training-script", default=os.environ.get("TRAINING_SCRIPT", DEFAULT_TRAINING_SCRIPT))
    parser.add_argument("--experiment-name", default=os.environ.get("EXPERIMENT_NAME", "qwen3-8b-asearcher-tis-datarand-flash-attn-nokl"))
    parser.add_argument("--project-name", default=os.environ.get("PROJECT_NAME", "search_r1_like_async_rl"))
    parser.add_argument("--actor-model-path", default=os.environ.get("ACTOR_MODEL_PATH", DEFAULT_ACTOR_MODEL_PATH))
    parser.add_argument("--resume-from-path", default=os.environ.get("RESUME_FROM_PATH", ""))
    parser.add_argument("--summary-visible-devices", default=os.environ.get("SUMMARY_VISIBLE_DEVICES", SINGLE_NODE_8GPU_LAYOUT["summary"]))
    parser.add_argument("--retrieval-visible-devices", default=os.environ.get("RETRIEVAL_VISIBLE_DEVICES", SINGLE_NODE_8GPU_LAYOUT["retrieval"]))
    parser.add_argument("--train-gpus", default=os.environ.get("TRAIN_CUDA_VISIBLE_DEVICES", SINGLE_NODE_8GPU_LAYOUT["train"]))
    parser.add_argument("--summary-extra-arg", action="append", default=[])
    parser.add_argument("--retrieval-extra-arg", action="append", default=[])
    parser.add_argument("--train-nnodes", type=int, default=int(os.environ.get("TRAIN_NNODES", "1")))
    parser.add_argument("--train-gpus-per-node", type=int, default=int(os.environ.get("TRAIN_GPUS_PER_NODE", "4")))
    parser.add_argument("--service-mode", choices=["single", "per-node"], default=os.environ.get("SERVICE_MODE", "single"))
    parser.add_argument("--node-gpus", type=int, default=int(os.environ.get("NODE_GPUS", "8")))
    return parser.parse_args()


def _load_tool_template(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _patch_tool_config(template: dict, retrieval_url: str) -> dict:
    cfg = template
    tools = cfg.get("tools", [])
    for tool in tools:
        if tool.get("class_name") == "verl.tools.search_tool.SearchTool":
            tool.setdefault("config", {})["retrieval_service_url"] = retrieval_url
    return cfg


def _write_runtime_tool_config(cfg: dict, working_dir: str) -> str:
    runtime_dir = Path(working_dir) / ".ray_runtime" / "tool_config"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    path = runtime_dir / "search_tool_runtime.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return str(path.resolve())


def _parse_visible_devices(devices: str) -> list[int]:
    return [int(x.strip()) for x in devices.split(",") if x.strip()]


def _make_pg(gpu_count: int, name: str):
    pg = placement_group([{"CPU": 1, "GPU": gpu_count}], strategy="STRICT_PACK", name=name)
    ray.get(pg.ready())
    return pg


def _alive_gpu_nodes():
    nodes = []
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        gpu_count = int(node.get("Resources", {}).get("GPU", 0))
        if gpu_count <= 0:
            continue
        nodes.append({"node_id": node["NodeID"], "ip": node["NodeManagerAddress"], "gpu_count": gpu_count})
    return sorted(nodes, key=lambda x: x["ip"])


def _validate_per_node_resource_plan(args, nodes):
    sidecar_gpus = args.summary_tp + args.retrieval_num_gpus
    if not math.isclose(sidecar_gpus, round(sidecar_gpus), rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"per-node mode requires integer sidecar gpu usage, got summary_tp + retrieval_num_gpus = {sidecar_gpus}")
    sidecar_gpus = int(round(sidecar_gpus))
    required_per_node = args.train_gpus_per_node + sidecar_gpus
    for node in nodes[: args.train_nnodes]:
        if node["gpu_count"] < required_per_node:
            raise ValueError(
                f"Node {node['ip']} has {node['gpu_count']} GPUs, but per-node plan requires "
                f"train_gpus_per_node({args.train_gpus_per_node}) + sidecar_gpus({sidecar_gpus}) = {required_per_node}"
            )
    return sidecar_gpus


def _launch_single_node_services(args):
    summary_devices = _parse_visible_devices(args.summary_visible_devices)
    retrieval_devices = _parse_visible_devices(args.retrieval_visible_devices)
    train_devices = _parse_visible_devices(args.train_gpus)
    overlap = (set(summary_devices) & set(retrieval_devices)) | (set(summary_devices) & set(train_devices)) | (set(retrieval_devices) & set(train_devices))
    if overlap:
        raise ValueError(f"GPU partitions overlap: {sorted(overlap)}")

    cluster_gpus = int(ray.cluster_resources().get("GPU", 0))
    required_max_gpu = max(summary_devices + retrieval_devices + train_devices)
    if cluster_gpus <= required_max_gpu:
        raise ValueError(f"Cluster reports {cluster_gpus} GPUs, but requested GPU id {required_max_gpu}")

    summary_pg = _make_pg(len(summary_devices), "summary-sglang-pg")
    retrieval_pg = _make_pg(len(retrieval_devices), "retrieval-service-pg")

    summary_actor = SummarySGLangActor.options(
        num_gpus=len(summary_devices),
        name="summary-sglang",
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=summary_pg),
    ).remote(
        model_path=args.summary_model_path,
        tensor_parallel_size=args.summary_tp,
        mem_fraction_static=args.summary_mem_fraction,
        port=args.summary_port,
        extra_args=args.summary_extra_arg,
        startup_timeout_s=args.summary_startup_timeout_s,
    )
    summary_url = ray.get(summary_actor.start.remote())

    retrieval_actor = RetrievalServiceActor.options(
        num_gpus=len(retrieval_devices),
        name="retrieval-service",
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=retrieval_pg),
    ).remote(
        script_path=args.search_script,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retriever_name=args.retriever_name,
        retriever_model=args.retriever_model,
        sglang_base_url=summary_url,
        port=args.retrieval_port,
        extra_args=["--sglang_model", args.summary_model_name, *args.retrieval_extra_arg],
        startup_timeout_s=args.retrieval_startup_timeout_s,
    )
    retrieval_url = ray.get(retrieval_actor.start.remote())
    return [summary_actor, retrieval_actor], retrieval_url


def _launch_per_node_services(args):
    nodes = _alive_gpu_nodes()
    if len(nodes) < args.train_nnodes:
        raise ValueError(f"Alive GPU nodes {len(nodes)} < requested TRAIN_NNODES {args.train_nnodes}")

    sidecar_gpus = _validate_per_node_resource_plan(args, nodes)
    selected_nodes = nodes[: args.train_nnodes]
    actors = []
    launched = []
    for idx, node in enumerate(selected_nodes):
        strategy = NodeAffinitySchedulingStrategy(node_id=node["node_id"], soft=False)
        summary_actor = SummarySGLangActor.options(
            num_gpus=args.summary_tp,
            name=f"summary-sglang-node-{idx}",
            scheduling_strategy=strategy,
        ).remote(
            model_path=args.summary_model_path,
            tensor_parallel_size=args.summary_tp,
            mem_fraction_static=args.summary_mem_fraction,
            port=args.summary_port,
            extra_args=args.summary_extra_arg,
            startup_timeout_s=args.summary_startup_timeout_s,
        )
        summary_url = ray.get(summary_actor.start.remote())

        retrieval_actor = RetrievalServiceActor.options(
            num_gpus=args.retrieval_num_gpus,
            name=f"retrieval-service-node-{idx}",
            scheduling_strategy=strategy,
        ).remote(
            script_path=args.search_script,
            index_path=args.index_path,
            corpus_path=args.corpus_path,
            retriever_name=args.retriever_name,
            retriever_model=args.retriever_model,
            sglang_base_url=summary_url,
            port=args.retrieval_port,
            extra_args=["--sglang_model", args.summary_model_name, *args.retrieval_extra_arg],
            startup_timeout_s=args.retrieval_startup_timeout_s,
        )
        retrieval_url = ray.get(retrieval_actor.start.remote())
        actors.extend([summary_actor, retrieval_actor])
        launched.append({
            "node_id": node["node_id"],
            "ip": node["ip"],
            "gpu_count": node["gpu_count"],
            "summary_url": summary_url,
            "retrieval_url": retrieval_url,
        })
    return actors, launched, sidecar_gpus


def main():
    args = _parse_args()
    ray.init(address=args.address, ignore_reinit_error=True)

    if args.service_mode == "single":
        service_actors, retrieval_url = _launch_single_node_services(args)
        tool_cfg = _patch_tool_config(_load_tool_template(args.tool_config_template), f"{retrieval_url}/retrieve_summarize_compat")
        tool_cfg_path = _write_runtime_tool_config(tool_cfg, args.working_dir)
        print(f"[launcher] service_mode=single retrieval_url={retrieval_url}")
        env = os.environ.copy()
        env.update({
            "TOOL_CONFIG": tool_cfg_path,
            "RETRIEVAL_SERVICE_URL": f"{retrieval_url}/retrieve_summarize_compat",
        })
    else:
        service_actors, launched, sidecar_gpus = _launch_per_node_services(args)
        tool_cfg_path = args.tool_config_template
        print(f"[launcher] service_mode=per-node retrieval_url={DEFAULT_LOCAL_RETRIEVAL_URL}")
        print(
            f"[launcher] per-node resource plan: train_gpus_per_node={args.train_gpus_per_node}, "
            f"summary_gpus={args.summary_tp}, retrieval_gpus={args.retrieval_num_gpus}, sidecar_total={sidecar_gpus}"
        )
        for item in launched:
            print(
                f"[launcher] node={item['ip']} gpu_count={item['gpu_count']} "
                f"summary_url={item['summary_url']} retrieval_url={item['retrieval_url']}"
            )
        env = os.environ.copy()
        env.update({
            "TOOL_CONFIG": tool_cfg_path,
            "RETRIEVAL_SERVICE_URL": DEFAULT_LOCAL_RETRIEVAL_URL,
        })

    def _cleanup():
        for actor in service_actors:
            try:
                ray.get(actor.stop.remote())
            except Exception:
                pass

    atexit.register(_cleanup)

    env.update({
        "CONFIG_PATH": args.config_path,
        "TRAIN_DATA": args.train_data,
        "VAL_DATA": args.val_data,
        "EXPERIMENT_NAME": args.experiment_name,
        "PROJECT_NAME": args.project_name,
        "ACTOR_MODEL_PATH": args.actor_model_path,
        "RESUME_FROM_PATH": args.resume_from_path,
        "TRAIN_CUDA_VISIBLE_DEVICES": args.train_gpus,
        "TRAIN_NNODES": str(args.train_nnodes),
        "TRAIN_GPUS_PER_NODE": str(args.train_gpus_per_node),
        "RAY_ADDRESS": args.address,
    })

    print(f"[launcher] tool_config={tool_cfg_path}")
    print(f"[launcher] TRAIN_NNODES={args.train_nnodes} TRAIN_GPUS_PER_NODE={args.train_gpus_per_node}")
    subprocess.check_call(["bash", args.training_script], cwd=args.working_dir, env=env)


if __name__ == "__main__":
    main()
