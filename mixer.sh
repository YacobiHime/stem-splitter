#!/usr/bin/env bash
# Usage: ./mixer.sh [--port 8000] [--host 0.0.0.0] [--tailscale-auth]   ブラウザで聴くミキサー（LAN / Tailscale）
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# keep the log from growing without bound while the server runs for weeks
LOG="$DIR/mixer.log"
if [ -f "$LOG" ] && [ "$(stat -c%s "$LOG" 2>/dev/null || echo 0)" -gt 5242880 ]; then
  tail -c 1048576 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi
export PYTHONPATH="$DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$DIR/.venv/bin/python" -m mixer.server "$@"
