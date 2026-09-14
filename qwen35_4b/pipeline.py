"""Small fixed-recipe driver. No historical launchers or stage-time code copies."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
ROOT = Path(os.environ.get("VERL_ROOT", REPO_ROOT / "verl")).expanduser().resolve()
BASE = "6dc2d2301e046006001dfdfbfb5f4666a1cf4ad1"
FILES = HERE / 'files'
# The package carries complete Python sources at the same relative paths.
# Hashes keep direct copies and later training runs on the verified recipe.
INSTALLED_FILES = {
    "verl/experimental/agent_loop/agent_loop.py": "9409cf536d0619c5d2a0499aeb197b0683ac61d86954f0cbc1cfb3a95525d2f6",
    "verl/experimental/agent_loop/tool_agent_loop.py": "a09757c275fa0c1e584cf62b1546aa0201255b4b983f145527dec8b79c922208",
    "verl/experimental/agent_loop/tool_parser.py": "bfe8fcbd18226b049cd411531688727c2ca03df27b369e0058d4ad572df6447a",
    "verl/models/transformers/monkey_patch.py": "6e34255834905fe30d1e2997cd9936227454d25eff24a8c8c4173c833be0de8e",
    "verl/models/transformers/qwen3_5.py": "91651d51e44caf825c9ceda0d88fc659e9072b7ee9e30b0d1d7f9b881073b792",
    "verl/tools/schemas.py": "e294118f48c18ac7c25c56c8dac3b883d93bdafa782d47f19db93cbc2a26ae8f",
    "verl/tools/search_tool.py": "ceb1d976c7d4dc875ddeaae8873851d2db6e5ed24c472fb738d28915a5b09ea7",
    "verl/tools/corpus_browser_tool.py": "e764ea423954861b240628bf55d2a5971ee207c428c20ec0f62f807cd9f1270f",
    "verl/tools/visit_tool.py": "955b9b847d393b96f909c189fa1c9f209407e5354353aa8eb455dfbe90919c3e",
    "verl/tools/utils/search_r1_like_utils.py": "ba6e47db6d379d375f47bd57f3d1215414aca18b3c939d421ea54df73854e074",
    "verl/utils/ulysses.py": "d7a5fbeec625f396eeadd95a6589434abec7563c82550a646d1515858c56c8be",
    "verl/utils/fsdp_utils.py": "0c4b64c635d3246dd6ac11645f79f85cb942662c21a1e8c5b4c4cb461821297e",
    "verl/utils/model.py": "2d9c0dfbfd9b776b766aa92952bed54433356223dca91b1c1e6d1914c9f3c856",
    "verl/workers/fsdp_workers.py": "788f6802add668cbca830506ebfaca1cb1376359a3e2132a16ecca30a1203f0a",
    "verl/workers/actor/dp_actor.py": "c909360a7e390bfa5470a0d88d388746d5911d024852ee25bf8f9e444a72ec53",
    "verl/workers/engine/fsdp/transformer_impl.py": "50e5bc5aef13c1bc354eb97a2de8fd8a13c384183d0ff7b28735998f4f4a331c",
    "verl/workers/rollout/sglang_rollout/sglang_rollout.py": "0c1702dae5da446d6c01fade6e5a581722122eaac383febd8c34a33fbf74e973",
    "verl/utils/dataset/prompt_completion_sft_dataset.py": "1d9c93ab116751c1e172c48c9a4a2d20fc96cd8557b8172caa3829f089cde121"
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def dry():
    return os.environ.get('DRY_RUN') == '1'


def path(key):
    return Path(os.environ[key]).expanduser().resolve()


def need(p):
    if not Path(p).exists():
        raise ValueError(f'Missing input: {p}; edit config.env')


@contextmanager
def run_lock():
    """Do not switch this run's retrieval environment during another stage."""
    root = path('RUN_ROOT')
    root.mkdir(parents=True, exist_ok=True)
    with (root/'pipeline.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another stage is running under RUN_ROOT; wait for it to finish') from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)


def check_base():
    if git('rev-parse', '--show-toplevel').stdout.strip() != str(ROOT):
        raise ValueError(f'VERL_ROOT must point to a VERL checkout: {ROOT}')
    if git('rev-parse', 'HEAD').stdout.strip() != BASE:
        raise ValueError(f'Wrong VERL base; need {BASE}')
    payload_hashes()


def payload_hashes():
    sources = {str(p.relative_to(FILES)) for p in FILES.rglob('*.py')}
    if sources != set(INSTALLED_FILES):
        raise ValueError('files/ must contain exactly the 18 declared Python sources')
    actual = {}
    for name, expected in INSTALLED_FILES.items():
        source = FILES/name
        if source.resolve() != source or not source.is_file() or digest(source) != expected:
            raise ValueError(f'Copy source is missing, redirected or modified: files/{name}')
        actual[name] = expected
    return actual


def check_edit_scope():
    changed = set(git('diff', '--name-only', 'HEAD', '--').stdout.splitlines())
    if changed - INSTALLED_FILES.keys():
        raise ValueError('Additional tracked edits are not part of this fixed recipe: ' + ', '.join(sorted(changed - INSTALLED_FILES.keys())))


def check_targets():
    for name, expected in INSTALLED_FILES.items():
        target = ROOT/name
        if target.resolve() != target or not target.is_file() or digest(target) != expected:
            raise ValueError(f'Installed file is missing, redirected or modified: {name}')
    check_edit_scope()


def install():
    check_base()
    check_edit_scope()
    if git('diff', '--cached', '--quiet', '--').returncode:
        raise ValueError('Staged changes exist. Use a NEW clean checkout; index is not modified')
    pending = []
    # Preflight every destination before copying anything. Exact already-copied
    # files are safe to reuse, including installations made by the older bundle.
    for name, expected in INSTALLED_FILES.items():
        target = ROOT/name
        if target.resolve() != target:
            raise ValueError(f'Refusing symlink target: {name}')
        if any(parent.exists() and not parent.is_dir() for parent in target.parents):
            raise ValueError(f'Copy target has a non-directory parent: {name}')
        if target.exists() and not target.is_file():
            raise ValueError(f'Copy target is not a regular file: {name}')
        if target.is_file() and digest(target) == expected:
            continue
        baseline = subprocess.run(['git', '-C', str(ROOT), 'show', BASE+':'+name], capture_output=True)
        if baseline.returncode == 0:
            if not target.is_file() or digest(target) != hashlib.sha256(baseline.stdout).hexdigest():
                raise ValueError(f'Copy target is missing or modified: {name}; nothing overwritten')
            if target.stat().st_nlink != 1:
                raise ValueError(f'Refusing hard-linked copy target: {name}')
        elif target.exists():
            raise ValueError(f'Untracked copy target already exists: {name}; nothing overwritten')
        pending.append((FILES/name, target))
    if not pending:
        check_targets()
        print('[install] all 18 same-name Python files already present; no files copied')
        return
    print(f'[install] copying {len(pending)} same-name Python files; trainer and checkpoint managers unchanged')
    for source, target in pending:
        print('[copy]', shlex.join(['cp', str(source), str(target)]), flush=True)
        if not dry():
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['cp', str(source), str(target)], check=True)
    if not dry():
        check_targets()


def check_installed():
    check_base()
    check_targets()


def hf(p):
    need(p / 'config.json')
    weights = list(p.glob('*.safetensors'))
    if not weights or any(f.stat().st_size == 0 for f in weights):
        raise ValueError(f'No HF safetensors weights: {p}')
    index = p/'model.safetensors.index.json'
    if index.exists():
        for name in set(json.loads(index.read_text())['weight_map'].values()):
            need(p/name)


def data_check(mode):
    import pyarrow.parquet as pq
    key, expected = ('SFT_DATA', 1807) if mode == 'sft' else ('RL_DATA', 338) if mode == 'rl' else ('EVAL_DATA', 830)
    p = path(key)
    table = pq.read_table(p)
    frozen = os.environ.get('FROZEN_DATA') == '1'
    historical_rows = expected
    if (frozen or mode == 'eval') and table.num_rows != expected:
        raise ValueError(f'{key}: expected frozen {expected} rows, got {table.num_rows}; do not substitute another pool')
    expected = table.num_rows
    if mode != 'eval' and expected < (16 if mode == 'sft' else 128):
        raise ValueError(f'{key}: {expected} rows cannot fill the configured training batch')
    if mode != 'eval' and not frozen:
        report = json.loads(path('SFT_REPORT' if mode == 'sft' else 'RL_REPORT').read_text())
        if (report.get('qwen35_4b_provenance') or {}).get('output_sha256') != digest(p):
            raise ValueError(f'{key}: missing/mismatched build provenance; frozen historical inputs require FROZEN_DATA=1')
    columns = set(table.column_names)
    if mode == 'sft':
        required = {'context_text', 'chosen_text', 'loss_spans_json', 'loss_tokens'}
        report = json.loads(path('SFT_REPORT').read_text())
        policy = report.get('loss_policy', {})
        if not report.get('final_gate', {}).get('passed') or report.get('rows', {}).get('output') != expected:
            raise ValueError('SFT report must match this passed causal pool and its actual row count')
        for name, value in {'context': 0, 'assistant_reasoning': 0, 'selected_tool_call': 1,
                            'selected_final_answer': 1, 'assistant_turn_end': 1}.items():
            if policy.get(name) != value:
                raise ValueError(f'Wrong causal loss policy: {name}')
    else:
        required = {'prompt', 'reward_model', 'data_source'}
        if 'prompt' in columns:
            prompts = table['prompt'].to_pylist()
            if len({json.dumps(v, sort_keys=True, ensure_ascii=False) for v in prompts}) != expected:
                raise ValueError('Duplicate prepared prompts')
            if any(not isinstance(v, list) or len(v) < 2 or v[0].get('role') != 'system'
                   or v[-1].get('role') != 'user' or not all(m.get('content') for m in v) for v in prompts):
                raise ValueError('Expected prepared native-tool system+user prompts, not raw question rows')
        if 'reward_model' in columns:
            for value in table['reward_model'].to_pylist():
                ground_truth = (value or {}).get('ground_truth')
                target = ground_truth.get('target') if isinstance(ground_truth, dict) else ground_truth
                if not target:
                    raise ValueError('Prepared row has no reward target')
        if 'extra_info' in columns and mode == 'rl':
            for value in table['extra_info'].to_pylist():
                extra = json.loads(value) if isinstance(value, str) else value or {}
                if (extra.get('benchmark_overfit') or {}).get('benchmark_contaminated'):
                    raise ValueError('Benchmark overfit data is NOT the safe RL pool')
    if not required <= columns:
        raise ValueError(f'{p} lacks prepared columns: {required - columns}')
    return {key: {'rows': expected, 'historical_rows': historical_rows, 'sha256': digest(p)}}


def hydra(values):
    return [key + '=' + json.dumps(value, separators=(',', ':')) for key, value in values.items()]


def ray_env():
    return {f'++ray_kwargs.ray_init.runtime_env.env_vars.{key}': value
            for key, value in os.environ.items()
            if key.startswith(('VERL_TOOL_AGENT_', 'TOOL_AGENT_', 'BROWSECOMP_')) or key in
            {'E5_BROWSER_BASE_URL', 'SEARCH_REWARD_MODE', 'SEARCH_STRICT_EVIDENCE_COEF_ENABLED',
             'SGLANG_DISABLE_CUDNN_CHECK', 'PYTHONPATH', 'CANDIDATE_WORKSPACE_ENABLE', 'SEARCH_QUERY_EXPANSION_ENABLE'}}


def agent_config(model, data, mode):
    training = mode == 'rl'
    p = 'actor_rollout_ref.'
    cfg = {'algorithm.adv_estimator': 'grpo', 'algorithm.use_kl_in_reward': False,
           'data.shuffle': training, 'data.seed': 2026, 'data.train_files': str(data),
           'data.val_files': str(data), 'data.train_batch_size': 128 if training else 256,
           'data.val_batch_size': 256 if training else int(os.environ.get('EVAL_BATCH', '32')),
           'data.max_prompt_length': 8192, 'data.max_response_length': 110000,
           'data.filter_overlong_prompts': True, 'data.truncation': 'error', 'data.return_raw_chat': True,
           p+'model.path': str(model), p+'model.use_remove_padding': True,
           p+'rollout.name': 'sglang', p+'rollout.mode': 'async', p+'rollout.tensor_model_parallel_size': 1,
           p+'rollout.gpu_memory_utilization': 0.40 if training else 0.75,
           p+'rollout.max_model_len': 122880, p+'rollout.temperature': 1.0 if training else 0.6,
           p+'rollout.top_p': 1.0 if training else 0.95, p+'rollout.top_k': -1,
           p+'rollout.n': 8 if training else 1, p+'rollout.agent.default_agent_loop': 'tool_agent',
           p+'rollout.agent.num_workers': 16 if training else int(os.environ.get('EVAL_WORKERS', '16')),
           p+'rollout.multi_turn.enable': True, p+'rollout.multi_turn.max_tool_response_length': 4096 if training else 16384,
           p+'rollout.multi_turn.tool_response_truncate_side': 'middle',
           p+'rollout.multi_turn.max_assistant_turns': 50, p+'rollout.multi_turn.max_user_turns': 50,
           p+'rollout.multi_turn.max_parallel_calls': 3, p+'rollout.multi_turn.format': 'qwen3_5',
           p+'rollout.multi_turn.tool_config_path': str(HERE/'tools.yaml'),
           'reward.custom_reward_function.path': str(HERE/'reward.py'),
           'reward.custom_reward_function.name': 'compute_score', 'trainer.nnodes': 1,
           'trainer.project_name': 'qwen35_4b', 'trainer.logger': ['console'], 'trainer.critic_warmup': 0,
           'trainer.save_freq': 5 if training else -1, 'trainer.test_freq': -1,
           'trainer.val_before_train': not training}
    if training:
        cfg.update({
            '++data.gen_batch_size': 128, '++algorithm.filter_groups.enable': False,
            p+'model.trust_remote_code': False, p+'model.enable_gradient_checkpointing': True,
            p+'model.enable_activation_offload': False, p+'model.use_fused_kernels': True,
            p+'model.fused_kernel_options.impl_backend': 'torch',
            p+'actor.optim.lr': 5e-7, p+'actor.optim.lr_warmup_steps_ratio': 0.0,
            p+'actor.ppo_mini_batch_size': 32, p+'actor.ppo_micro_batch_size_per_gpu': 1,
            p+'actor.use_kl_loss': True, p+'actor.kl_loss_coef': 5e-4,
            p+'actor.clip_ratio_low': 0.2, p+'actor.clip_ratio_high': 0.28,
            p+'actor.kl_loss_type': 'low_var_kl', p+'actor.entropy_coeff': 0,
            p+'actor.ulysses_sequence_parallel_size': 4, p+'actor.fsdp_config.ulysses_sequence_parallel_size': 4,
            p+'actor.fsdp_config.param_offload': True, p+'actor.fsdp_config.optimizer_offload': True,
            p+'actor.fsdp_config.use_torch_compile': False, p+'ref.log_prob_micro_batch_size_per_gpu': 1,
            p+'ref.ulysses_sequence_parallel_size': 4, p+'ref.fsdp_config.ulysses_sequence_parallel_size': 4,
            p+'ref.fsdp_config.param_offload': True, p+'ref.fsdp_config.use_torch_compile': False,
            p+'rollout.free_cache_engine': True, p+'rollout.enable_chunked_prefill': True,
            p+'rollout.enable_prefix_caching': True, p+'rollout.log_prob_micro_batch_size_per_gpu': 1,
            p+'rollout.calculate_log_probs': True, p+'rollout.val_kwargs.temperature': 0.7,
            p+'rollout.val_kwargs.top_p': 0.8, p+'rollout.val_kwargs.top_k': 20, p+'rollout.val_kwargs.n': 1,
            'algorithm.rollout_correction.bypass_mode': False, 'algorithm.rollout_correction.rollout_is': 'token',
            'algorithm.rollout_correction.rollout_is_threshold': 2.0,
            'algorithm.rollout_correction.rollout_is_batch_normalize': False,
            '++'+p+'rollout.engine_kwargs.sglang.random_seed': 2026,
            '+'+p+'actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap': ['Qwen3_5DecoderLayer', 'Qwen3_5VisionBlock'],
            '+'+p+'ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap': ['Qwen3_5DecoderLayer', 'Qwen3_5VisionBlock'],
            p+'actor.checkpoint.save_contents': ['model', 'optimizer', 'extra', 'hf_model'],
            'trainer.total_training_steps': 20, 'trainer.total_epochs': 20, 'trainer.resume_mode': 'disable',
            'trainer.max_actor_ckpt_to_keep': 4, 'trainer.experiment_name': 'safe_core_grpo',
        })
    else:
        cfg.update({p+'rollout.val_kwargs.temperature': 0.6, p+'rollout.val_kwargs.top_p': 0.95,
                    p+'rollout.val_kwargs.top_k': -1, p+'rollout.val_kwargs.n': 1,
                    'trainer.val_only': True, 'trainer.total_epochs': 1})
    return cfg


def sft_config():
    return {'data.train_files': str(path('SFT_DATA')), 'data.val_files': None,
            'data.custom_cls.path': str(ROOT/'verl/utils/dataset/prompt_completion_sft_dataset.py'),
            'data.custom_cls.name': 'PromptCompletionSFTDataset', '+data.context_key': 'context_text',
            '+data.completion_key': 'chosen_text', '+data.loss_spans_key': 'loss_spans_json',
            'data.train_batch_size': 16, 'data.micro_batch_size_per_gpu': 1,
            'data.max_length': 127000, 'data.max_token_len_per_gpu': 32768,
            'data.truncation': 'error', 'data.pad_mode': 'no_padding', 'data.use_dynamic_bsz': True,
            'optim.lr': 2e-7, 'model.path': str(path('BASE_MODEL')), 'model.use_remove_padding': True,
            'model.use_fused_kernels': True, 'model.fused_kernel_options.impl_backend': 'torch',
            'model.enable_activation_offload': False, 'engine': 'fsdp', 'engine.use_torch_compile': False,
            'engine.ulysses_sequence_parallel_size': 4,
            '+engine.wrap_policy.transformer_layer_cls_to_wrap': ['Qwen3_5DecoderLayer','Qwen3_5VisionBlock'],
            'trainer.project_name': 'qwen35_4b', 'trainer.experiment_name': 'causal_sft',
            'trainer.total_training_steps': 112, 'trainer.save_freq': 28, 'trainer.resume_mode': 'disable',
            'trainer.logger': ['console'], 'checkpoint.save_contents': ['model','optimizer','extra','hf_model'],
            'trainer.max_ckpt_to_keep': 5}


def gpu_count(devices):
    ids = devices.split(',')
    if any(not x.isdigit() for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('GPU IDs must be distinct comma-separated integers')
    return len(ids)


def launch(command, cfg, devices, output, inputs):
    cfg = {**cfg, 'trainer.n_gpus_per_node': gpu_count(devices), 'trainer.default_local_dir': str(output)}
    cmd = [*command, *hydra(cfg)]
    print('[launch]', 'CUDA_VISIBLE_DEVICES='+devices, shlex.join(cmd), flush=True)
    if dry():
        return
    output.mkdir(parents=True, exist_ok=False)
    metadata = {'base_commit': BASE, 'installation': 'same_name_python_copy',
                'runtime_sha256': payload_hashes(), 'config': cfg,
                'inputs': inputs, 'gpus': devices,
                'code_sha256': {p.name: digest(p) for p in HERE.iterdir() if p.is_file() and p.suffix in {'.py','.sh','.yaml'}}}
    (output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    subprocess.run(cmd, cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': devices}, check=True)


def service_health(port):
    try:
        with urlopen(f'http://127.0.0.1:{port}/health', timeout=5) as r:
            return json.load(r)
    except (URLError, TimeoutError):
        with socket.socket() as s:
            s.settimeout(1)
            if s.connect_ex(('127.0.0.1', port)) == 0:
                raise ValueError(f'Port {port} is busy but not a healthy retrieval service')
        return None


def start_service(cmd, devices, state, log, ready, *, timeout=1800, metadata=None):
    """Shared startup for Wiki, teacher and dual retrieval; never adopt a foreign PID."""
    cmd = [str(x) for x in cmd]
    print('[service]', 'CUDA_VISIBLE_DEVICES='+devices, shlex.join(cmd), flush=True)
    if dry():
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as handle:
        proc = subprocess.Popen(cmd, env={**os.environ, 'CUDA_VISIBLE_DEVICES': devices},
                                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                start_new_session=True)
    write_json(state, {**(metadata or {}), 'pid': proc.pid, 'command': cmd, 'owner': 'qwen35_4b'})
    for _ in range(timeout):
        if proc.poll() is not None:
            raise ValueError(f'Service exited: {log}')
        try:
            if ready():
                return
        except (OSError, ValueError):
            pass  # A process may open its socket before its health endpoint is ready.
        time.sleep(1)
    raise ValueError(f'Service startup timed out: {log}; owned process retained for inspection')


def service_matches(payload, mode, role):
    runtime = (payload or {}).get('runtime') or {}
    allowed = str(path('SAFE_DOCIDS')) if mode == 'safe' else ''
    expected_index = path('SEARCH_INDEX' if role == 'primary' else 'REASONING_INDEX')
    expected_encoder = path('SEARCH_ENCODER' if role == 'primary' else 'REASONING_ENCODER')
    allowed_count = len({s.strip() for s in path('SAFE_DOCIDS').read_text().splitlines() if s.strip()}) if mode == 'safe' else None
    return bool(payload and payload.get('success') and runtime.get('corpus_source_path') == str(path('CORPUS'))
                and (runtime.get('allowed_docids_path') or '') == allowed
                and runtime.get('allowed_docids_count') == allowed_count
                and runtime.get('corpus_rows') == 100195
                and runtime.get('query_encoder_model') == str(expected_encoder)
                and runtime.get('index_path') == str(expected_index)
                and runtime.get('server_source_sha256') == digest(HERE/'retrieval.py')
                and (role != 'primary' or (runtime.get('search_global_enable')
                     and runtime.get('dual_retrieval_enable')
                     and runtime.get('dual_secondary_rank_url') == f'http://127.0.0.1:{os.environ["SECONDARY_PORT"]}/rank')))


def stop_owned(state):
    if not state.is_file():
        return False
    saved = json.loads(state.read_text())
    if int(saved.get('pid', 0)) <= 1 or len(saved.get('command', [])) < 2:
        return False
    proc = Path('/proc')/str(int(saved['pid']))/'cmdline'
    if not proc.exists() or proc.read_bytes().split(b'\0')[:-1] != [s.encode() for s in saved['command']]:
        return False
    command = saved['command']
    if not (command[1] in {str(HERE/'retrieval.py'), str(HERE/'wiki_retrieval.py')}
            or command[1:3] == ['-m', 'sglang.launch_server'] and saved.get('owner') == 'qwen35_4b'):
        return False
    print('[retrieval] stopping only this package\'s recorded process', saved['pid'], flush=True)
    os.kill(saved['pid'], signal.SIGTERM)
    return True


def retrieval(mode):
    if mode not in {'safe','full'}:
        raise ValueError('retrieval mode must be safe or full')
    if dry():
        print(f'[retrieval] {mode}, dual search, GPUs={os.environ["RETRIEVAL_GPUS"]}; no service started')
        return
    for key in ('CORPUS','SEARCH_INDEX','SEARCH_ENCODER','REASONING_INDEX','REASONING_ENCODER'):
        need(path(key))
    if mode == 'safe':
        need(path('SAFE_DOCIDS'))
        if not path('SAFE_DOCIDS').read_text().strip():
            raise ValueError('Safe allow-list is empty')
    logs = path('RUN_ROOT')/'services'
    logs.mkdir(parents=True, exist_ok=True)
    primary, secondary = int(os.environ['PRIMARY_PORT']), int(os.environ['SECONDARY_PORT'])
    if primary == secondary or not all(1024 <= port <= 65535 for port in (primary,secondary)):
        raise ValueError('Use distinct unprivileged retrieval ports')
    for role, port in [('secondary',secondary),('primary',primary)]:
        state = logs/f'{port}.json'
        health = service_health(port)
        if service_matches(health, mode, role):
            print(f'[retrieval] reusing verified {mode} {role} port={port}')
            continue
        if health:
            if not stop_owned(state):
                raise ValueError(f'Port {port} belongs to another/stale service. Stop it manually or change ports; no process killed')
            for _ in range(30):
                if service_health(port) is None:
                    break
                time.sleep(1)
            else:
                raise ValueError(f'Owned service on {port} did not stop')
        cmd = [sys.executable, str(HERE/'retrieval.py'), '--retrieval_method', 'dense',
               '--corpus_repo_id', str(path('CORPUS')), '--faiss_gpu', '--host', '127.0.0.1',
               '--port', str(port), '--topk', '50', '--search_global_rank_cutoff', '50']
        if mode == 'safe':
            cmd += ['--allowed_docids_path', str(path('SAFE_DOCIDS'))]
        if role == 'secondary':
            cmd += ['--index_path', str(path('REASONING_INDEX')), '--query_encoder_model', str(path('REASONING_ENCODER')),
                    '--query_encoder_backend','transformer_eos','--query_encoder_max_length','8192',
                    '--query_encoder_batch_size','8','--search_global_rerank_mode','rrf',
                    '--query_instruction', "Instruct: Given a user's reasoning followed by a web search query, retrieve relevant passages that answer the query while incorporating the user's reasoning\nQuery:"]
        else:
            cmd += ['--index_path', str(path('SEARCH_INDEX')), '--query_encoder_model', str(path('SEARCH_ENCODER')),
                    '--search_request_concurrency','16','--search_global_enable','--search_lazy_frontier_hydration',
                    '--search_global_output_topk','10','--search_global_rerank_mode','rrf_lexical',
                    '--search_global_rich_topk','5','--search_global_rich_budget','2100','--search_global_rich_mode','chunks',
                    '--search_global_compact_mode','query','--search_global_compact_snippet_chars','240',
                    '--search_global_max_output_chars','15000','--search_card_chunk_chars','2100','--search_card_overlap_chars','240',
                    '--search_result_link_label','OpenURL','--search_result_open_tool_name','open',
                    '--browser_line_chars','160','--browser_default_num_lines','40','--browser_max_num_lines','64',
                    '--dual_retrieval_enable','--dual_secondary_rank_url',f'http://127.0.0.1:{secondary}/rank',
                    '--dual_candidate_topk','50','--dual_rrf_k','60','--dual_primary_weight','1.0',
                    '--dual_secondary_weight','2.0','--dual_primary_rerank_mode','rrf','--dual_secondary_timeout','600']
        start_service(cmd, os.environ['RETRIEVAL_GPUS'], state, logs/f'{port}-{mode}.log',
                      lambda: service_matches(service_health(port), mode, role),
                      timeout=900, metadata={'mode': mode})
    request = Request(f'http://127.0.0.1:{primary}/search', headers={'Content-Type':'application/json'},
                      data=json.dumps({'queries':['Albert Einstein physicist biography'], 'topk':50,
                                       'reasoning':'Find biographical sources; inspect the source before answering.',
                                       'frontier_enable':True,'frontier_candidate_topk':50,
                                       'frontier_base_rank_cutoff':10,'frontier_output_topk':10}).encode())
    with urlopen(request, timeout=650) as r:
        result = json.load(r)
    fusion = result.get('fusion') or {}
    if not result.get('success') or not fusion.get('applied') or fusion.get('fallback_error'):
        raise ValueError(f'Dual retrieval probe failed: {fusion}')
    print('[retrieval]', mode, 'dual probe passed', flush=True)


def score(file):
    from reward import strict_terminal_reward_details
    values, ids, valid, exact = [], set(), 0, 0
    with Path(file).open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get('query_id')
            if key is None:
                key = row.get('input')
            if key is None or str(key) == '' or str(key) in ids:
                raise ValueError('Missing/duplicate question identity in n1 eval')
            ids.add(str(key))
            target = row.get('gts')
            target = target.get('target') if isinstance(target,dict) else target
            if not target:
                raise ValueError('Missing ground-truth target')
            # Historical offline score used the first reference if a list.
            if isinstance(target,list):
                target = target[0]
            details = strict_terminal_reward_details(row.get('output') or '', str(target))
            values.append(details['base_f1']); valid += details['protocol_valid']; exact += details['base_f1'] >= 1-1e-12
    if len(values) != 830:
        raise ValueError(f'Expected 830 rows, got {len(values)}')
    report = {'rows':830,'strict_f1_x100':100*sum(values)/830,'strict_exact':exact,
              'protocol_valid':valid,'jsonl_sha256':digest(file)}
    print(json.dumps(report,indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['install','verify','sft','rl','eval','retrieval','score'])
    parser.add_argument('extra', nargs='*')
    args = parser.parse_args()
    if args.stage in {'install','verify','sft','rl'} and args.extra:
        parser.error(f'{args.stage} does not accept arguments')
    if args.stage == 'retrieval' and (len(args.extra) > 1 or args.extra and args.extra[0] not in {'safe','full'}):
        parser.error('retrieval accepts safe or full')
    if args.stage == 'install':
        return install()
    if args.stage == 'verify':
        check_installed()
        print('[verify] all 18 installed Python files match files/ exactly')
        return
    if args.stage == 'score':
        if len(args.extra) != 1:
            raise ValueError('score needs one JSONL path')
        return score(args.extra[0])
    if not dry():
        check_installed()
        with run_lock():
            return execute(args)
    return execute(args)


def execute(args):
    if args.stage == 'retrieval':
        return retrieval(args.extra[0] if args.extra else 'safe')
    data = {} if dry() else data_check(args.stage)
    if args.stage in {'sft','rl'}:
        devices = os.environ['TRAIN_GPUS']
        if gpu_count(devices) != 8:
            raise ValueError('This fixed recipe uses 8 training GPUs; a 4-GPU experiment is a different recipe')
        output = path('RUN_ROOT')/args.stage
        if output.exists():
            raise ValueError(f'Run directory exists: {output}; choose a new RUN_ROOT, no checkpoint overwritten')
        if args.stage == 'sft':
            if not dry():
                hf(path('BASE_MODEL'))
                from inspect import signature
                from fla.ops.cp.context import build_cp_context
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule
                if 'cp_context' not in signature(chunk_gated_delta_rule).parameters:
                    raise ValueError('FLA must support cp_context for SP4')
            return launch(['torchrun','--standalone','--nnodes=1','--nproc_per_node=8','-m','verl.trainer.sft_trainer'],
                          sft_config(), devices, output, data)
        if not dry():
            hf(path('SFT_MODEL'))
        retrieval('safe')
        cfg = {**agent_config(path('SFT_MODEL'),path('RL_DATA'),'rl'), **ray_env(),
               'trainer.validation_data_dir':str(output/'validation')}
        return launch([sys.executable,'-m','verl.trainer.main_ppo','--config-path',str(ROOT/'verl/trainer/config'),
                       '--config-name','ppo_trainer'], cfg, devices, output, data)
    steps = args.extra or ['20','15','10','5']
    if not all(s in {'5','10','15','20','sft','scaffold'} for s in steps) or len(set(steps)) != len(steps):
        raise ValueError('eval accepts distinct 20 15 10 5, or sft, or scaffold')
    models = {s: path('BASE_MODEL') if s=='scaffold' else path('SFT_MODEL') if s=='sft'
              else path('RUN_ROOT')/'rl'/f'global_step_{s}'/'actor/huggingface' for s in steps}
    tag = os.environ['EVAL_TAG']
    if not tag.replace('_','').replace('-','').isalnum():
        raise ValueError('EVAL_TAG must be alphanumeric with underscore/hyphen')
    for step, model in models.items():
        if not dry():
            hf(model)
        if (path('RUN_ROOT')/f'eval_{step}_{tag}').exists():
            raise ValueError('Eval output exists; choose a new EVAL_TAG')
    if int(os.environ['EVAL_BATCH']) < int(os.environ['EVAL_WORKERS']):
        raise ValueError('EVAL_BATCH must be >= EVAL_WORKERS')
    retrieval('full')
    for step, model in models.items():
        output = path('RUN_ROOT')/f'eval_{step}_{tag}'
        cfg = {**agent_config(model,path('EVAL_DATA'),'eval'),**ray_env(),
               'trainer.validation_data_dir':str(output),'trainer.experiment_name':f'eval_{step}'}
        if step == 'scaffold':
            for key,value in {'FIRST_TOOL_CALL_BOUNDARY':'1','FIRST_TOOL_CALL_MAX_TOKENS':'8192',
                              'FINAL_ANSWER_BOUNDARY':'1','GENERIC_ANSWER_RESCUE':'1','DUPLICATE_STREAK_LIMIT':'2'}.items():
                cfg['++ray_kwargs.ray_init.runtime_env.env_vars.VERL_TOOL_AGENT_'+key] = value
                cfg['++ray_kwargs.ray_init.runtime_env.env_vars.TOOL_AGENT_'+key] = value
        launch([sys.executable,str(HERE/'evaluate.py'),'--config-path',str(ROOT/'verl/trainer/config'),
                '--config-name','ppo_trainer'],cfg,os.environ['EVAL_GPUS'],output,data)
        if not dry():
            (output/'score.json').write_text(json.dumps(score(output/'0.jsonl'),indent=2)+'\n')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, subprocess.CalledProcessError) as error:
        raise SystemExit(f'ERROR: {error}')
