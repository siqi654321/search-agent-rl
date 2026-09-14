#!/usr/bin/env bash
# One entry for the fixed recipe. No stage-time code copying or implicit deletion.
set -euo pipefail
STAGE="${1:-all}"
if (( $# )); then shift; fi
case "$STAGE" in
  help|-h|--help)
    printf '%s\n' \
      'Usage: bash run.sh [all | STAGE [ARGS]]' \
      '  install / verify    Copy / check the 18 same-name VERL Python files' \
      '  sft-data            Protected n4 rollout and causal SFT dataset' \
      '  sft                 Train 112 steps; RL uses checkpoint 28' \
      '  qa                  Safe corpus, chains, atomic QA and semantic cleaning' \
      '  rollout [BATCH]     Strict n8 over all questions, or one numbered batch' \
      '  select              Analyze batches and merge the length-evidence core' \
      '  rl                  GRPO20; save 5/10/15/20 and retain all four' \
      '  eval-data           Prepare the 830-question evaluation dataset' \
      '  eval [STEPS...]     Evaluate 20 15 10 5 by default; or sft / scaffold' \
      '  retrieval safe|full Start/check dual retrieval for training/evaluation' \
      '  score FILE          Recompute strict F1 for one 830-row JSONL' \
      'Set DRY_RUN=1 to print commands without starting services or training.'
    exit 0 ;;
  all|install|verify|sft-data|sft|qa|select|rl|eval-data)
    if (( $# )); then echo "ERROR: $STAGE does not accept arguments" >&2; exit 2; fi ;;
  rollout|eval|retrieval|score) ;;
  *) echo "ERROR: unknown stage '$STAGE'; run bash run.sh --help" >&2; exit 2 ;;
esac

PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$PACKAGE_DIR/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-$REPO_ROOT/verl}"
if [[ ! -d "$VERL_ROOT" ]]; then
  echo "ERROR: VERL_ROOT does not exist: $VERL_ROOT" >&2
  exit 1
fi
cd "$REPO_ROOT"
set -a
source "$PACKAGE_DIR/config.env"
set +a
export PACKAGE_DIR REPO_ROOT VERL_ROOT
export PYTHONPATH="$VERL_ROOT:$PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 SGLANG_DISABLE_CUDNN_CHECK=1
# Reset historical behavior extensions explicitly. Keep Frontier and resource
# finalization, exactly as in the recorded strict recipe.
for key in FIRST_TOOL_CALL_BOUNDARY FINAL_ANSWER_BOUNDARY FINAL_ANSWER_PREFILL \
  GENERIC_ANSWER_RESCUE GENERIC_ANSWER_RESCUE_PREFILL GENERIC_ANSWER_RESCUE_EXTRA_ATTEMPTS \
  SCAFFOLD_V2 VERIFICATION_CHECKPOINT CONTEXT_RESET EVIDENCE_STATE_REBUILD \
  FRONTIER_OPEN_FIRST CANDIDATE_WORKSPACE PI_NAVIGATION DISTILL_TRACE; do
  export "VERL_TOOL_AGENT_${key}=0" "TOOL_AGENT_${key}=0"
done
export CANDIDATE_WORKSPACE_ENABLE=0
export SEARCH_QUERY_EXPANSION_ENABLE=0
export VERL_TOOL_AGENT_DUPLICATE_STREAK_LIMIT=999 TOOL_AGENT_DUPLICATE_STREAK_LIMIT=999
export VERL_TOOL_AGENT_FINAL_RESERVE_TOKENS=8192 TOOL_AGENT_FINAL_RESERVE_TOKENS=8192
export VERL_TOOL_AGENT_FINAL_ATTEMPT_TOKENS=2048 TOOL_AGENT_FINAL_ATTEMPT_TOKENS=2048
export VERL_TOOL_AGENT_FINAL_MAX_ATTEMPTS=2 TOOL_AGENT_FINAL_MAX_ATTEMPTS=2
export VERL_TOOL_AGENT_FINALIZATION_STRUCTURE_CLOSURE=1 VERL_TOOL_AGENT_SEARCH_FRONTIER=1
export VERL_TOOL_AGENT_SEARCH_FRONTIER_BASE_TOPK=10 VERL_TOOL_AGENT_SEARCH_FRONTIER_OUTPUT_TOPK=10
export VERL_TOOL_AGENT_SEARCH_FRONTIER_CANDIDATE_TOPK=50 VERL_TOOL_AGENT_SEARCH_FRONTIER_OVERLAP_THRESHOLD=0.90
export VERL_TOOL_AGENT_SEARCH_FRONTIER_COMPACT_SNIPPET_CHARS=800 VERL_TOOL_AGENT_SEARCH_FRONTIER_EXCLUDE_LIMIT=1000
export SEARCH_REWARD_MODE=strict_terminal_f1 SEARCH_STRICT_EVIDENCE_COEF_ENABLED=0
export BROWSECOMP_SEARCH_SERVICE_URL="http://127.0.0.1:$PRIMARY_PORT/search"
export BROWSECOMP_BROWSER_SERVICE_URL="http://127.0.0.1:$PRIMARY_PORT"
export E5_BROWSER_BASE_URL="$BROWSECOMP_BROWSER_SERVICE_URL"
export BROWSECOMP_SEARCH_TOPK=50 BROWSECOMP_SEARCH_NUM_WORKERS=16 BROWSECOMP_SEARCH_RATE_LIMIT=16
export BROWSECOMP_DUAL_RETRIEVAL_ENABLE=1 BROWSECOMP_DUAL_REASONING_MAX_CHARS=8000

case "$STAGE" in
  all)
    for stage in install sft-data sft qa rollout select rl eval-data; do
      bash "$PACKAGE_DIR/run.sh" "$stage"
    done
    exec bash "$PACKAGE_DIR/run.sh" eval 20 15 10 5 ;;
  sft-data) exec python3 "$PACKAGE_DIR/data.py" sft ;;
  eval-data) exec python3 "$PACKAGE_DIR/data.py" eval ;;
  qa|rollout|select) exec python3 "$PACKAGE_DIR/data.py" "$STAGE" "$@" ;;
  *) exec python3 "$PACKAGE_DIR/pipeline.py" "$STAGE" "$@" ;;
esac
