# Search Agent RL

Qwen3-8B trained with pure RL, reaching `36+` BrowseComp Plus token-F1 in `250` steps.

This repository builds a search-centric RL workflow on top of the `verl` submodule. It includes:

- retrieval service scripts in `retrieval/`
- training patches and configs in `src/`
- fully async training patches in `src/async/`
- OPD training patches in `src/opd/`
- mtools training patches in `src/mtools/`
- Qwen3.5 main-agent evaluation patches and tool configs in `main_agent_eval/`
- data conversion scripts in `data/`
- the training entry script `train.sh`
- the fully async training entry script `train_async.sh`
- the OPD training entry script `train_opd.sh`
- the mtools training entry script `train_mtools.sh`
- the BrowseComp+ evaluation entry script `browsecomp_plus_eval.sh`
- the mtools BrowseComp+ evaluation entry script `browsecomp_plus_eval_mtools.sh`
- the Qwen3.5 main-agent evaluation entry script `eval_verl_qwen35.sh`

## 1. Environment Setup

### 1.1 Clone the repository and initialize submodules

```bash
git clone https://github.com/siqi654321/search-agent-rl.git
cd search-agent-rl
git submodule update --init --recursive
```

### 1.2 Python environment

Python `3.10` or `3.11` is recommended. Install `verl` and the retrieval / serving dependencies first.

One common setup flow is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ./verl
pip install sglang fastapi uvicorn faiss-gpu datasets transformers sentencepiece
```

If `faiss-gpu` is not available on your platform, replace it with an appropriate installable variant.

### 1.3 Model and directory conventions

The scripts prefer environment variables when provided. Otherwise they default to repository-relative paths such as:

- `models/Qwen3-1.7B`
- `models/Qwen3-8B`
- `models/Qwen3.5-35B-A3B`
- `models/Qwen3-Embedding-8B`
- `models/e5-base-v2`

Recommended directory layout:

```text
search-agent-rl/
├── data/
├── models/
│   ├── Qwen3-1.7B/
│   ├── Qwen3-8B/
│   ├── Qwen3.5-35B-A3B/
│   ├── Qwen3-Embedding-8B/
│   └── e5-base-v2/
├── main_agent_eval/
├── retrieval/
├── src/
├── eval_verl_qwen35.sh
├── browsecomp_plus_eval_mtools.sh
├── train_opd.sh
├── train_mtools.sh
├── train_async.sh
├── train.sh
└── browsecomp_plus_eval.sh
```

## 2. Data Preparation

### 2.1 ASearcher training data

`train.sh` uses the following by default:

- `data/asearcher_searchr1/train.parquet`
- `data/asearcher_searchr1/test.parquet`

Place your training and validation parquet files there, or override them before running:

```bash
export TRAIN_DATA=/path/to/train.parquet
export VAL_DATA=/path/to/test.parquet
```

### 2.2 Retrieval index and corpus

`train.sh` reads these files by default:

- `data/e5_Flat.index`
- `data/wiki-18.jsonl`

Override them if needed:

```bash
export INDEX_PATH=/path/to/e5_Flat.index
export CORPUS_PATH=/path/to/wiki-18.jsonl
export RETRIEVER_MODEL=/path/to/e5-base-v2
```

### 2.3 BrowseComp+ data conversion

Script: `data/convert_browsecomp_plus_to_search_r1_like.sh`

By default, it converts the Hugging Face dataset `Tevatron/browsecomp-plus` into:

- `data/browsecompplus_searchr1/`

Run it with:

```bash
bash data/convert_browsecomp_plus_to_search_r1_like.sh
```

To override the source dataset or output directory:

```bash
HF_REPO_ID=Tevatron/browsecomp-plus OUTPUT_DIR=./data/browsecompplus_searchr1 \
bash data/convert_browsecomp_plus_to_search_r1_like.sh
```

### 2.4 BrowseComp+ retrieval resources

`browsecomp_plus_eval.sh` uses the following by default:

- `data/browsecomp-plus-indexes/qwen3-embedding-8b`
- `Tevatron/browsecomp-plus-corpus`

If you already have a local index or a local corpus, override them with:

```bash
export INDEX_PATH=/path/to/browsecomp-index
export CORPUS_REPO_ID=/path/to/local-corpus-or-hf-repo-id
export QUERY_ENCODER_MODEL=/path/to/Qwen3-Embedding-8B
```

## 3. Training

### 3.1 Standard training

Training entry point: `train.sh`

The script does the following:

1. starts the local retrieval service `retrieval/retrieval_server_sglang_summarize.py`
2. starts an SGLang server for `Qwen3-1.7B`
3. syncs the patched files from `src/` into `verl/`
4. launches GRPO training through `verl.trainer.main_ppo`

Run training with:

```bash
bash train.sh
```

Common override example:

```bash
SGLANG_MODEL_PATH=./models/Qwen3-1.7B \
ACTOR_MODEL_PATH=./models/Qwen3-8B \
INDEX_PATH=./data/e5_Flat.index \
CORPUS_PATH=./data/wiki-18.jsonl \
TRAIN_DATA=./data/asearcher_searchr1/train.parquet \
VAL_DATA=./data/asearcher_searchr1/test.parquet \
bash train.sh
```

Logs are written to the repository root by default:

- `retrieve.out`
- `qwen3_1.7b.out`

### 3.2 Fully async training

Fully async training entry point: `train_async.sh`

The script does the following:

1. starts the local retrieval service `retrieval/retrieval_server_sglang_summarize.py`
2. starts an SGLang server for `Qwen3-1.7B`
3. syncs the base patches from `src/` into `verl/verl/`
4. syncs the fully async patches from `src/async/` into `verl/verl/`
5. launches fully async GRPO training through `verl.experimental.fully_async_policy.fully_async_main`

Run fully async training with:

```bash
bash train_async.sh
```

Common override example:

```bash
SGLANG_MODEL_PATH=./models/Qwen3-1.7B \
MODEL_PATH=./models/Qwen3-8B \
INDEX_PATH=./data/e5_Flat.index \
CORPUS_PATH=./data/wiki-18.jsonl \
TRAIN_DATA=./data/asearcher_searchr1/train.parquet \
VAL_DATA=./data/asearcher_searchr1/test.parquet \
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
TRAIN_GPUS_PER_NODE=4 \
ROLLOUT_GPUS_PER_NODE=2 \
bash train_async.sh
```

Useful fully async overrides include:

- `ROLLOUT_MODE=async`
- `ROLLOUT_NAME=sglang`
- `TRAIN_BATCH_SIZE`
- `GEN_BATCH_SIZE`
- `TOTAL_ROLLOUT_STEPS`
- `STALENESS_THRESHOLD`
- `MAX_CONCURRENT_SAMPLES`
- `CKPTS_DIR`

The script also keeps the process alive with `sleep infinity` after launching training, which is useful in some job schedulers.

### 3.3 OPD training

OPD training entry point: `train_opd.sh`

The script does the following:

1. starts the local retrieval service `retrieval/retrieval_server_sglang_summarize.py`
2. starts an SGLang server for `Qwen3-1.7B`
3. syncs the base patches from `src/` into `verl/verl/`
4. syncs the OPD patches from `src/opd/` into `verl/verl/`
5. launches GRPO training with on-policy distillation enabled through `verl.trainer.main_ppo`

Run OPD training with:

```bash
bash train_opd.sh
```

Common override example:

```bash
SGLANG_MODEL_PATH=./models/Qwen3-1.7B \
ACTOR_MODEL_PATH=./models/Qwen3-4B \
TEACHER_MODEL_PATH=./models/Qwen3-8B \
INDEX_PATH=./data/e5_Flat.index \
CORPUS_PATH=./data/wiki-18.jsonl \
TRAIN_DATA=./data/asearcher_searchr1/train.parquet \
VAL_DATA=./data/asearcher_searchr1/test.parquet \
EXPERIMENT_NAME=qwen3-4b-asearcher-opd-from-8b \
bash train_opd.sh
```

Useful OPD overrides include:

- `TEACHER_MODEL_PATH`
- `OPD_ENABLE`
- `OPD_KL_COEF`
- `OPD_PURE_DISTILL`
- `TRAIN_VISIBLE_DEVICES`
- `RETRIEVAL_VISIBLE_DEVICES`
- `SGLANG_VISIBLE_DEVICES`
- `CKPTS_DIR`
- `PROJECT_NAME`
- `EXPERIMENT_NAME`

The script keeps the OPD-specific patch sync in one place:

- `src/opd/fsdp_workers.py` -> `verl/verl/workers/fsdp_workers.py`
- `src/opd/main_ppo.py` -> `verl/verl/trainer/main_ppo.py`
- `src/opd/ray_trainer.py` -> `verl/verl/trainer/ppo/ray_trainer.py`
- `src/opd/utils.py` -> `verl/verl/trainer/ppo/utils.py`

### 3.4 Mtools training

Mtools training entry point: `train_mtools.sh`

The script does the following:

1. starts one local SGLang summary server for tool-side summarization
2. starts `src/mtools/retrieval_server_train.py` to provide `/search` and `/browse`
3. syncs the shared reward / tool parser patches from `src/` into `verl/verl/`
4. syncs the mtools-specific tool patches from `src/mtools/` into `verl/verl/`
5. converts the base parquet data into the mtools prompt format with `src/mtools/prepare_mtool_data.py`
6. launches GRPO training through `verl.trainer.main_ppo`

Run mtools training with:

```bash
bash train_mtools.sh
```

Common override example:

```bash
SUMMARY_MODEL_PATH=./models/Qwen3-1.7B \
ACTOR_MODEL_PATH=./models/Qwen3-4B \
INDEX_PATH=./data/e5_Flat.index \
CORPUS_PATH=./data/wiki-18.jsonl \
RAW_TRAIN_DATA=./data/asearcher_searchr1/train.parquet \
RAW_VAL_DATA=./data/asearcher_searchr1/test.parquet \
TOOL_CONFIG=./src/mtools/mtool_config.yaml \
EXPERIMENT_NAME=qwen3-4b-mtools-search-visit \
bash train_mtools.sh
```

Useful mtools training overrides include:

- `SUMMARY_MODEL_PATH`
- `ACTOR_MODEL_PATH`
- `INDEX_PATH`
- `CORPUS_PATH`
- `RETRIEVER_MODEL`
- `RAW_TRAIN_DATA`
- `RAW_VAL_DATA`
- `TRAIN_DATA`
- `VAL_DATA`
- `TOOL_CONFIG`
- `TRAIN_VISIBLE_DEVICES`
- `RETRIEVER_VISIBLE_DEVICES`
- `SUMMARY_VISIBLE_DEVICES`
- `PROJECT_NAME`
- `EXPERIMENT_NAME`

The script keeps the mtools-specific patch sync in one place:

- `src/mtools/search_r1_like_utils.py` -> `verl/verl/tools/utils/search_r1_like_utils.py`
- `src/mtools/search_tool.py` -> `verl/verl/tools/search_tool.py`
- `src/mtools/visit_tool.py` -> `verl/verl/tools/visit_tool.py`

### 3.5 Multinode sync training

Multinode entry scripts:

- environment sync on every node: `multinode/setup_env.sh`
- Ray job submit on the head node: `multinode/launch.sh`
- training entry inside the Ray job: `multinode/train_verl_sync.sh`

The multinode workflow is:

1. start a multi-node Ray cluster manually
2. run `multinode/setup_env.sh` on every node to sync the required patches into `verl/`
3. run `multinode/launch.sh` on the Ray head node
4. the launcher submits a Ray job that executes `src/multinode/ray_launch_sync_training.py`
5. the Ray job starts the summary service and retrieval service, then runs `multinode/train_verl_sync.sh`

Run the environment sync on every node first:

```bash
bash multinode/setup_env.sh
```

Then submit the training job on the Ray head node:

```bash
bash multinode/launch.sh
```

Common override example:

```bash
RAY_ADDRESS=auto \
TRAIN_NNODES=8 \
TRAIN_GPUS_PER_NODE=4 \
SUMMARY_TP=1 \
RETRIEVAL_NUM_GPUS=1 \
SUMMARY_MODEL_PATH=./models/Qwen3-1.7B \
ACTOR_MODEL_PATH=./models/Qwen3-8B \
RETRIEVER_MODEL=./models/e5-base-v2 \
INDEX_PATH=./data/e5_Flat.index \
CORPUS_PATH=./data/wiki-18.jsonl \
EXPERIMENT_NAME=qwen3-8b-search-multinode \
bash multinode/launch.sh
```

Useful multinode overrides include:

- `RAY_ADDRESS`
- `SERVICE_MODE`
- `CONFIG_PATH`
- `TOOL_CONFIG_TEMPLATE`
- `TRAINING_SCRIPT`
- `TRAIN_DATA`
- `VAL_DATA`
- `RETRIEVAL_SCRIPT`
- `INDEX_PATH`
- `CORPUS_PATH`
- `RETRIEVER_MODEL`
- `SUMMARY_MODEL_PATH`
- `ACTOR_MODEL_PATH`
- `PROJECT_NAME`
- `EXPERIMENT_NAME`
- `SUMMARY_TP`
- `SUMMARY_MEM_FRACTION`
- `SUMMARY_PORT`
- `SUMMARY_STARTUP_TIMEOUT_S`
- `RETRIEVAL_PORT`
- `RETRIEVAL_STARTUP_TIMEOUT_S`
- `RETRIEVAL_NUM_GPUS`
- `TRAIN_CUDA_VISIBLE_DEVICES`
- `TRAIN_NNODES`
- `TRAIN_GPUS_PER_NODE`
- `SUMMARY_VISIBLE_DEVICES`
- `RETRIEVAL_VISIBLE_DEVICES`
- `NODE_GPUS`
- `CKPTS_DIR`
- `RESUME_FROM_PATH`

## 4. Evaluation

### 4.1 Standard BrowseComp+ evaluation

BrowseComp+ evaluation entry point: `browsecomp_plus_eval.sh`

The script will:

1. start `retrieval/retrieval_server_browsecomp_plus.py`
2. start three `Qwen3-8B` SGLang servers
3. run `verl.trainer.main_ppo` in `val_only` mode for evaluation

Run evaluation with:

```bash
bash browsecomp_plus_eval.sh
```

Common override example:

```bash
BASE_MODEL_PATH=./models/Qwen3-8B \
CHECKPOINT_PATH=./checkpoints/qwen3-8b-asearcher-datarand-flash-attn/global_step_250 \
INDEX_PATH=./data/browsecomp-plus-indexes/qwen3-embedding-8b \
QUERY_ENCODER_MODEL=./models/Qwen3-Embedding-8B \
bash browsecomp_plus_eval.sh
```

### 4.2 Mtools BrowseComp+ evaluation

Mtools evaluation entry point: `browsecomp_plus_eval_mtools.sh`

The script will:

1. start three `Qwen3-1.7B` SGLang summary servers
2. start `src/mtools/retrieval_server_browsecomp_plus.py`
3. convert the evaluation parquet into the mtools prompt format with `src/mtools/prepare_mtool_data.py`
4. run `verl.trainer.main_ppo` in `val_only` mode with the mtools search / visit tools enabled

Run mtools evaluation with:

```bash
bash browsecomp_plus_eval_mtools.sh
```

Common override example:

```bash
BASE_MODEL_PATH=./models/Qwen3-4B \
CHECKPOINT_PATH=./checkpoints/qwen3-4b-mtools/global_step_250 \
INDEX_PATH=./data/browsecomp-plus-indexes/qwen3-embedding-8b \
QUERY_ENCODER_MODEL=./models/Qwen3-Embedding-8B \
RAW_VAL_DATA=./data/browsecompplus_searchr1/test.parquet \
bash browsecomp_plus_eval_mtools.sh
```

Useful mtools evaluation overrides include:

- `BASE_MODEL_PATH`
- `CHECKPOINT_PATH`
- `INDEX_PATH`
- `CORPUS_REPO_ID`
- `QUERY_ENCODER_MODEL`
- `RAW_VAL_DATA`
- `VAL_DATA`
- `TOOL_CONFIG`
- `VALIDATION_DATA_DIR`
- `SUMMARY_BASE_URLS`
- `RETRIEVER_VISIBLE_DEVICES`
- `EVAL_VISIBLE_DEVICES`

### 4.3 Qwen3.5 main-agent BrowseComp+ evaluation

Qwen3.5 main-agent evaluation entry point: `eval_verl_qwen35.sh`

This workflow uses the repository-local files in `main_agent_eval/` and does not
depend on external source trees. The script:

1. starts `main_agent_eval/retrieval_server.py` with global search ranking and the stateful browser endpoints
2. uses `main_agent_eval/browsecomp_tool_config.yaml` to expose the native `search`, `open`, and `find` tools
3. copies the main-agent patches from `main_agent_eval/` into the corresponding `verl/verl/` package locations
4. converts the evaluation parquet to the native Qwen3.5 tool-prompt format
5. runs `verl.trainer.main_agent_eval` in validation-only mode

The verification-checkpoint, search-frontier, answer-rescue, and soft-scaffold
controls used by the reference run are already enabled as overridable defaults in
the script. A normal repository-local run is therefore:

```bash
bash eval_verl_qwen35.sh 2>&1 | tee eval_qwen35_open_find.out
```

The default resource locations are:

- model: `models/Qwen3.5-35B-A3B`
- query encoder: `models/Qwen3-Embedding-8B`
- index: `data/browsecomp-plus-indexes/qwen3-embedding-8b`
- corpus: `Tevatron/browsecomp-plus-corpus`
- input data: `data/browsecompplus_searchr1/test.parquet`
- generated evaluation data: `data/main_agent_eval_browsecomp_test.parquet`
- tool config: `main_agent_eval/browsecomp_tool_config.yaml`
- retrieval server: `main_agent_eval/retrieval_server.py`

Override resource locations or runtime capacity without editing the script:

```bash
BASE_MODEL_PATH=/path/to/Qwen3.5-35B-A3B \
QUERY_ENCODER_MODEL=/path/to/Qwen3-Embedding-8B \
INDEX_PATH=/path/to/browsecomp-plus-index \
CORPUS_REPO_ID=/path/to/browsecomp-plus-corpus \
RAW_VAL_DATA=/path/to/test.parquet \
RETRIEVER_VISIBLE_DEVICES=0,1,2,3 \
EVAL_VISIBLE_DEVICES=2,3,4,5,6,7 \
VALIDATION_DATA_DIR=./validation_qwen35_open_find \
bash eval_verl_qwen35.sh 2>&1 | tee eval_qwen35_open_find.out
```

Set `CHECKPOINT_PATH` to evaluate a checkpoint. Leave it unset to evaluate the
base model directly. Additional useful overrides include:

- `TOPK`
- `SEARCH_OUTPUT_TOPK`
- `SEARCH_RICH_TOPK`
- `SEARCH_RICH_BUDGET_CHARS`
- `SEARCH_COMPACT_SNIPPET_CHARS`
- `SEARCH_MAX_OUTPUT_CHARS`
- `MAX_TOOL_RESPONSE_LENGTH`
- `MAX_PARALLEL_CALLS`
- `AGENT_NUM_WORKERS`
- `TOOL_CONFIG`
- `RETRIEVAL_SERVER_SCRIPT`
- `PROJECT_NAME`
- `EXPERIMENT_NAME`

## 5. Notes

- `verl/` is a submodule, so always run `git submodule update --init --recursive` after the first clone.
- If your models, indexes, or corpora live elsewhere, prefer overriding with environment variables instead of writing absolute paths back into the scripts.
- For debugging, you can temporarily remove `nohup` and run the commands in the foreground.

## 6. Acknowledgements

Special thanks to the following write-ups for inspiration and practical reference:

- `https://zhuanlan.zhihu.com/p/1987092986388038648`
- `https://zhuanlan.zhihu.com/p/2007446730245961005`

## 7. Citation

If you fork this repository, use it in experiments, or build on top of it in your own work, please cite this repository.

```bibtex
@misc{search_agent_rl_2026,
  title        = {Search Agent RL},
  year         = {2026},
  howpublished = {\url{https://github.com/siqi654321/search-agent-rl}},
  note         = {GitHub repository}
}
```
