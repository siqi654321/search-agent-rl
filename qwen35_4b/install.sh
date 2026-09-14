#!/usr/bin/env bash
# Validate all inputs/targets, then cp complete same-name Python files from files/.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run.sh" install "$@"
