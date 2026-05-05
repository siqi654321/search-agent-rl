#!/usr/bin/env bash
set -xeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "$0")/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-$ROOT_DIR/verl}"
VERL_SRC_DIR="${VERL_SRC_DIR:-$VERL_ROOT/verl}"

cp "$ROOT_DIR/src/tool_parser.py" "$VERL_SRC_DIR/experimental/agent_loop/tool_parser.py"
cp "$ROOT_DIR/src/multinode/search_tool.py" "$VERL_SRC_DIR/tools/search_tool.py"
cp "$ROOT_DIR/src/search_r1_like_utils.py" "$VERL_SRC_DIR/tools/utils/search_r1_like_utils.py"
cp "$ROOT_DIR/src/search_r1_like_qa_em.py" "$VERL_SRC_DIR/utils/reward_score/search_r1_like_qa_em.py"
cp "$ROOT_DIR/src/__init__.py" "$VERL_SRC_DIR/utils/reward_score/__init__.py"
cp "$ROOT_DIR/src/multinode/service_actor.py" "$VERL_SRC_DIR/service_actors.py"
