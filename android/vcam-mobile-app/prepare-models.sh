#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PYTHON=${PYTHON:-"$ROOT/.venv/bin/python"}
exec "$PYTHON" "$SCRIPT_DIR/model-tools/prepare_models.py" "$@"
