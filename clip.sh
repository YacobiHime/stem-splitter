#!/usr/bin/env bash
# Cut a test clip: ./clip.sh SONG [START_SEC=60] [DURATION_SEC=30]
# -> inbox/clips/<song>_clip<START>s.wav (44.1kHz stereo float32)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src="$1"; start="${2:-60}"; dur="${3:-30}"
name="$(basename "${src%.*}")"
mkdir -p "$DIR/inbox/clips"
out="$DIR/inbox/clips/${name}_clip${start}s.wav"
ffmpeg -nostdin -hide_banner -loglevel error -y -ss "$start" -t "$dur" -i "$src" \
  -vn -af aresample=resampler=soxr -ar 44100 -ac 2 -c:a pcm_f32le "$out"
echo "$out"
