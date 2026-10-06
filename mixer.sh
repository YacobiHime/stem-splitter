#!/usr/bin/env bash
# Usage: ./mixer.sh [--port 8000] [--host 0.0.0.0]   ブラウザで聴くミキサー（LAN 内のみ）
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$DIR/.venv/bin/python" -m mixer.server "$@"
