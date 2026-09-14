#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PACKAGE_DIR="${QWEN35_4B_DIR:-$ROOT_DIR/qwen35_4b}"

case "${1:-}" in
    help|-h|--help)
        cat <<'EOF'
Usage: bash eval_qwen35_4b.sh [20 15 10 5 | sft | scaffold]

With no arguments, evaluates RL checkpoints 20, 15, 10, and 5 in that order.
Use "sft" for the SFT step-28 model or "scaffold" for the base 4B model.
EOF
        exit 0
        ;;
esac

bash "$PACKAGE_DIR/install.sh"
if (( $# )); then
    exec bash "$PACKAGE_DIR/run.sh" eval "$@"
fi
exec bash "$PACKAGE_DIR/run.sh" eval 20 15 10 5
