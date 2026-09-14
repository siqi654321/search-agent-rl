#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PACKAGE_DIR="${QWEN35_4B_DIR:-$ROOT_DIR/qwen35_4b}"
STAGE="${1:-}"
if (( $# )); then shift; fi

case "$STAGE" in
    sft)
        INTERNAL_STAGE=sft-data
        ;;
    qa|rollout|select)
        INTERNAL_STAGE="$STAGE"
        ;;
    eval)
        INTERNAL_STAGE=eval-data
        ;;
    help|-h|--help|"")
        cat <<'EOF'
Usage: bash prepare_qwen35_4b_data.sh STAGE [ARGS]

Stages:
  sft                 Build causal SFT data from protected base-model rollouts
  qa                  Build the safe corpus, chains, synthetic QA, and semantic filter
  rollout [BATCH]     Generate n=8 trajectories for all QA or one numbered batch
  select              Audit completed rollout batches and build the GRPO pool
  eval                Convert the benchmark data to the native evaluation prompt

Run these around training in this order:
  data sft -> train sft -> data qa -> data rollout -> data select -> train rl -> data eval -> eval
EOF
        exit 0
        ;;
    *)
        echo "Unknown data stage: $STAGE" >&2
        exit 2
        ;;
esac

bash "$PACKAGE_DIR/install.sh"
exec bash "$PACKAGE_DIR/run.sh" "$INTERNAL_STAGE" "$@"
