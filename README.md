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
- the Qwen3.5 multi-trajectory aggregation entry script `aggregate_verl_qwen35.sh`
- the Qwen3.5-4B data entry script `prepare_qwen35_4b_data.sh`
- the Qwen3.5-4B training entry script `train_qwen35_4b.sh`
- the Qwen3.5-4B evaluation entry script `eval_qwen35_4b.sh`

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
│   ├── AgentIR-4B/
│   ├── Qwen3.5-4B/
│   └── e5-base-v2/
├── main_agent_eval/
├── qwen35_4b/
├── retrieval/
├── src/
├── prepare_qwen35_4b_data.sh
├── train_qwen35_4b.sh
├── eval_qwen35_4b.sh
├── eval_verl_qwen35.sh
├── aggregate_verl_qwen35.sh
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

### 2.5 Qwen3.5-4B synthetic training data

Qwen3.5-4B data entry point: `prepare_qwen35_4b_data.sh`

The implementation is under `qwen35_4b/`. It builds two training datasets:

- a causal SFT dataset from protected `n=4` base-model trajectories, supervising selected tool calls, corrected actions, final answers, and turn boundaries rather than entire trajectories
- a GRPO question pool built from contamination-filtered corpus documents, multi-hop chains, Qwen3.5-35B-A3B question generation, semantic filtering, `n=8` rollout audits, and evidence/length-based core selection

The default outputs are repository-relative:

- `data/qwen35_4b/sft.parquet`
- `data/qwen35_4b/qa/clean/questions.parquet`
- `data/qwen35_4b/rl.parquet`
- data-stage receipts under `data/qwen35_4b/state/`
- service and data logs under `outputs/qwen35_4b/`

Export the resource locations once before starting the workflow. The values
below are placeholders; do not write machine-specific locations into tracked
files:

```bash
export BASE_MODEL=/path/to/Qwen3.5-4B
export RAW_SFT_DATA=/path/to/asearcher/train.parquet
export WIKI_CORPUS=/path/to/wiki-18.jsonl
export WIKI_INDEX=/path/to/e5_Flat.index
export WIKI_ENCODER=/path/to/e5-base-v2

export TEACHER_MODEL=/path/to/Qwen3.5-35B-A3B
export CORPUS=/path/to/browsecomp-plus-corpus
export SEARCH_INDEX=/path/to/browsecomp-plus-index
export SEARCH_ENCODER=/path/to/Qwen3-Embedding-8B
export REASONING_INDEX=/path/to/agentir-browsecomp-plus-index
export REASONING_ENCODER=/path/to/AgentIR-4B

export BENCHMARK_JSONL=/path/to/benchmark.jsonl
export QREL_GOLDS=/path/to/qrel-golds.txt
export QREL_EVIDENCE=/path/to/qrel-evidence.txt
```

First build the causal SFT data:

```bash
bash prepare_qwen35_4b_data.sh sft 2>&1 | tee qwen35_4b_sft_data.out
```

After completing SFT training as described in section 3.6, build the synthetic
QA pool, generate trajectories, and select the GRPO pool:

```bash
bash prepare_qwen35_4b_data.sh qa 2>&1 | tee qwen35_4b_qa.out
bash prepare_qwen35_4b_data.sh rollout 2>&1 | tee qwen35_4b_rollout.out
bash prepare_qwen35_4b_data.sh select 2>&1 | tee qwen35_4b_select.out
```

Rollout can be resumed one complete batch at a time, for example:

```bash
bash prepare_qwen35_4b_data.sh rollout 2
```

Completed stages are reused only when their input, code, recipe, and output
receipts still match. An interrupted rollout batch is retained separately and
rerun as a whole; trajectories from partial attempts are never appended to a
new batch.

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

### 3.6 Qwen3.5-4B causal SFT and GRPO

Qwen3.5-4B training entry point: `train_qwen35_4b.sh`

Both training stages require eight GPUs. The entry script verifies that the
`verl/` submodule is at the pinned compatible revision and installs the 18
complete runtime files from `qwen35_4b/files/` into matching locations inside
the submodule. It refuses conflicting local modifications and does not reset
the repository.

After generating `data/qwen35_4b/sft.parquet`, run causal SFT:

```bash
TRAIN_GPUS=0,1,2,3,4,5,6,7 \
bash train_qwen35_4b.sh sft 2>&1 | tee qwen35_4b_sft.out
```

The fixed SFT recipe trains for 112 steps, saves every 28 steps, and uses
sequence parallelism 4. The subsequent rollout and GRPO stages use the step-28
Hugging Face checkpoint by default:

```text
outputs/qwen35_4b/sft/global_step_28/huggingface/
```

Once the QA rollout and selection stages from section 2.5 have produced
`data/qwen35_4b/rl.parquet`, start GRPO:

```bash
TRAIN_GPUS=0,1,2,3,4,5,6,7 \
RETRIEVAL_GPUS=0,1,2,3 \
bash train_qwen35_4b.sh rl 2>&1 | tee qwen35_4b_rl.out
```

The GRPO recipe initializes from SFT step 28, samples eight trajectories per
question, trains for 20 steps, and retains checkpoints 5, 10, 15, and 20 under
`outputs/qwen35_4b/rl/`. Dual retrieval combines the normal query route and the
reasoning-aware route with configurable corpus, index, encoder, GPU, and port
locations from `qwen35_4b/config.env` or environment variables.

The end-to-end order is therefore:

```text
prepare sft -> train sft -> prepare qa -> prepare rollout -> prepare select -> train rl -> prepare eval -> eval
```

Use `DRY_RUN=1` with either entry script to validate paths and print the fixed
commands without starting services or GPU jobs. Use a new `DATA_ROOT` after
changing data-generation code, and a new `RUN_ROOT` when retraining so existing
datasets and checkpoints are not overwritten.

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

1. starts a primary BrowseComp+ retriever and a reasoning-aware secondary retriever from `main_agent_eval/retrieval_server.py`
2. fuses both rankings with weighted reciprocal-rank fusion
3. uses `main_agent_eval/browsecomp_tool_config.yaml` to expose the native `search`, `open`, and `find` tools
4. copies the main-agent patches from `main_agent_eval/` into the corresponding `verl/verl/` package locations
5. converts the evaluation parquet to the native Qwen3.5 tool-prompt format
6. runs one seeded trajectory through `verl.trainer.main_agent_eval` in validation-only mode

The reference setup enables search-frontier and generic-answer rescue while
disabling scaffold-v2, verification checkpoint, context reset, evidence-state
rebuild, frontier-open-first, candidate workspace, and PI navigation. These are
overridable environment defaults in the script.

The default resource locations are:

- evaluation model: `models/Qwen3.5-35B-A3B`
- primary query encoder: `models/Qwen3-Embedding-8B`
- primary index: `data/browsecomp-plus-indexes/qwen3-embedding-8b`
- secondary query encoder: `models/AgentIR-4B`
- secondary index: `data/agentir/AgentIR_browsecomp-plus`
- corpus: `Tevatron/browsecomp-plus-corpus`
- input data: `data/browsecompplus_searchr1/test.parquet`
- generated evaluation data: `data/main_agent_eval_browsecomp_test.parquet`
- tool config: `main_agent_eval/browsecomp_tool_config.yaml`
- retrieval server: `main_agent_eval/retrieval_server.py`

Run one trajectory with a specific seed:

```bash
EVAL_SEED=2026 \
VALIDATION_DATA_DIR="$PWD/validation_main_agent_qwen35_dual_seed2026" \
bash eval_verl_qwen35.sh 2>&1 | tee eval_qwen35_dual_seed2026.out
```

To produce the four inputs for aggregation on one machine, run the seeds
sequentially. The retrieval services are reused after the first run:

```bash
for seed in 2026 2027 2028 2029; do
  EVAL_SEED="$seed" \
  VALIDATION_DATA_DIR="$PWD/validation_main_agent_qwen35_dual_seed${seed}" \
  bash eval_verl_qwen35.sh 2>&1 | tee "eval_qwen35_dual_seed${seed}.out"
done
```

Run seeds in parallel only when each process has isolated evaluator GPUs and
retrieval ports, or when all processes intentionally share already-running
retrieval services.

Override resource locations or runtime capacity without editing the script:

```bash
BASE_MODEL_PATH=/path/to/Qwen3.5-35B-A3B \
QUERY_ENCODER_MODEL=/path/to/Qwen3-Embedding-8B \
INDEX_PATH=/path/to/browsecomp-plus-index \
AGENTIR_QUERY_ENCODER_MODEL=/path/to/AgentIR-4B \
AGENTIR_INDEX_PATH=/path/to/agentir-browsecomp-plus-index \
CORPUS_REPO_ID=/path/to/browsecomp-plus-corpus \
RAW_VAL_DATA=/path/to/test.parquet \
RETRIEVER_VISIBLE_DEVICES=0,1,2,3 \
AGENTIR_RETRIEVER_VISIBLE_DEVICES=0,1,2,3 \
EVAL_VISIBLE_DEVICES=2,3,4,5,6,7 \
N_GPUS_PER_NODE=6 \
EVAL_SEED=2029 \
VALIDATION_DATA_DIR=./validation_main_agent_qwen35_dual_seed2029 \
bash eval_verl_qwen35.sh 2>&1 | tee eval_qwen35_dual_seed2029.out
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
- `DUAL_SEARCH_RATE_LIMIT`
- `DUAL_RETRIEVAL_CANDIDATE_TOPK`
- `DUAL_RETRIEVAL_RRF_K`
- `DUAL_RETRIEVAL_PRIMARY_WEIGHT`
- `DUAL_RETRIEVAL_SECONDARY_WEIGHT`
- `DUAL_REASONING_MAX_CHARS`
- `TOOL_CONFIG`
- `RETRIEVAL_SERVER_SCRIPT`
- `SEARCH_TOOL_SOURCE`
- `TOOL_AGENT_LOOP_SOURCE`
- `PROJECT_NAME`
- `EXPERIMENT_NAME`

### 4.4 Qwen3.5 multi-trajectory aggregation

Aggregation entry point: `aggregate_verl_qwen35.sh`

After the four seeded runs have produced `0.jsonl`, the aggregation workflow:

1. aligns the trajectories and groups identical normalized answers
2. creates target-blind candidate records containing only the question, proposed answers, run status, and trajectory-visible evidence
3. starts or reuses an OpenAI-compatible SGLang server for the aggregation model
4. asks the model to choose an existing candidate without synthesizing a new answer
5. loads targets only in the final evaluation stage and reports every single trajectory, the four-trajectory oracle, and the aggregate score

The aggregation implementation is
`main_agent_eval/evaluate_multitrajectory.py`. It uses the BrowseComp evaluation
helpers already installed under the target VERL checkout's `scripts/` directory:

- `evaluate_browsecomp_answer_verifier.py`
- `evaluate_browsecomp_candidate_selector.py`
- `evaluate_browsecomp_multitrajectory.py`

With the default seed output directories and aggregation model location, run:

```bash
bash aggregate_verl_qwen35.sh 2>&1 | tee aggregate_qwen35_k4.out
```

The default artifacts are written under `multitrajectory/`:

- `browsecomp_qwen35_k4.input.jsonl`
- `browsecomp_qwen35_k4.predictions.jsonl`
- `browsecomp_qwen35_k4.report.json`
- `browsecomp_qwen35_k4.server.out`

Override the model, GPUs, endpoint, or trajectory locations without editing the
scripts:

```bash
AGG_MODEL=/path/to/Qwen3.5-35B-A3B \
AGG_VISIBLE_DEVICES=0,1 \
AGG_TENSOR_PARALLEL_SIZE=2 \
AGG_PORT=30010 \
TRAJECTORY_2026=/path/to/seed2026/0.jsonl \
TRAJECTORY_2027=/path/to/seed2027/0.jsonl \
TRAJECTORY_2028=/path/to/seed2028/0.jsonl \
TRAJECTORY_2029=/path/to/seed2029/0.jsonl \
bash aggregate_verl_qwen35.sh 2>&1 | tee aggregate_qwen35_k4.out
```

To use an aggregation server that is already running, set
`START_AGGREGATOR_SERVER=0` and provide `AGG_ENDPOINT` and `AGG_MODEL`. The
Python runner resumes an existing predictions file and only submits unfinished
rows; choose a different `RUN_NAME` or output paths for a clean rerun.

### 4.5 Qwen3.5-4B checkpoint evaluation

Qwen3.5-4B evaluation entry point: `eval_qwen35_4b.sh`

First convert the 830-question BrowseComp+ parquet to the same native
`search`/`open`/`find` prompt format used by the training recipe:

```bash
RAW_EVAL_DATA=/path/to/browsecomp-plus/test.parquet \
bash prepare_qwen35_4b_data.sh eval 2>&1 | tee qwen35_4b_eval_data.out
```

This writes `data/qwen35_4b/eval.parquet` by default. Then evaluate the four
retained GRPO checkpoints in descending order:

```bash
EVAL_GPUS=4,5,6,7 \
EVAL_WORKERS=16 \
EVAL_BATCH=32 \
bash eval_qwen35_4b.sh 20 15 10 5 2>&1 | tee qwen35_4b_eval.out
```

Calling the script without checkpoint arguments uses the same `20 15 10 5`
order. Each checkpoint uses full-corpus dual retrieval, one trajectory per
question, temperature `0.6`, and top-p `0.95`. Results are written to:

```text
outputs/qwen35_4b/eval_20_strict_v1/0.jsonl
outputs/qwen35_4b/eval_20_strict_v1/score.json
outputs/qwen35_4b/eval_15_strict_v1/...
outputs/qwen35_4b/eval_10_strict_v1/...
outputs/qwen35_4b/eval_5_strict_v1/...
```

Use a new `EVAL_TAG` for a clean rerun because existing result directories are
not overwritten:

```bash
EVAL_TAG=rerun \
bash eval_qwen35_4b.sh 20 15 10 5 2>&1 | tee qwen35_4b_eval_rerun.out
```

The SFT step-28 checkpoint and the untrained base model can be evaluated as
separate controls:

```bash
bash eval_qwen35_4b.sh sft
bash eval_qwen35_4b.sh scaffold
```

The complete 4B workflow is therefore:

```bash
bash prepare_qwen35_4b_data.sh sft
bash train_qwen35_4b.sh sft
bash prepare_qwen35_4b_data.sh qa
bash prepare_qwen35_4b_data.sh rollout
bash prepare_qwen35_4b_data.sh select
bash train_qwen35_4b.sh rl
bash prepare_qwen35_4b_data.sh eval
bash eval_qwen35_4b.sh 20 15 10 5
```

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
