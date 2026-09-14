#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PACKAGE_DIR="${QWEN35_4B_DIR:-$ROOT_DIR/qwen35_4b}"
STAGE="${1:-}"
if (( $# )); then shift; fi

case "$STAGE" in
    sft|rl)
        ;;
    help|-h|--help|"")
        cat <<'EOF'
Usage: bash train_qwen35_4b.sh STAGE

Stages:
  sft    Train the 112-step causal SFT recipe and save every 28 steps
  rl     Initialize from SFT step 28 and train the 20-step GRPO recipe
EOF
        exit 0
        ;;
    *)
        echo "Unknown training stage: $STAGE" >&2
        exit 2
        ;;
esac

if (( $# )); then
    echo "$STAGE does not accept positional arguments" >&2
    exit 2
fi

bash "$PACKAGE_DIR/install.sh"
exec bash "$PACKAGE_DIR/run.sh" "$STAGE"
