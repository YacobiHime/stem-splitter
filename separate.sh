#!/usr/bin/env bash
# Usage: ./separate.sh [--force] [--explore] [--optional] [--clean] FILE_OR_DIR...
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$DIR/.venv/bin/python" -m pipeline "$@"
