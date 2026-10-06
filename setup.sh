#!/usr/bin/env bash
# Usage: ./setup.sh [--full]
#   基本:   .venv（Python 3.12 + CUDA 版 torch）、requirements.txt、vendor/BTC-ISMIR19（コード認識）
#   --full: ＋ YourMT3+（vendor/YourMT3 と重み 561MB）と MuScriptor（高精度の MIDI 化）
# 何度実行してもよい（入っているものはそのまま）。分離モデルの重みは初回の分離のときに自動でダウンロードされる。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

FULL=0
for a in "$@"; do
  case "$a" in
    --full) FULL=1 ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
BTC_COMMIT=2682317
YMT3_COMMIT=5e66c1e
YMT3_CKPT=amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt
YMT3_SHA256=ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25

step() { printf '\n==> %s\n' "$*"; }
need() { command -v "$1" >/dev/null || { echo "$1 が見つかりません。$2" >&2; exit 1; }; }

need uv "https://docs.astral.sh/uv/ の手順で入れてください（curl -LsSf https://astral.sh/uv/install.sh | sh）"
need git "sudo apt install git"
need ffmpeg "sudo apt install ffmpeg"
command -v nvidia-smi >/dev/null || echo "注意: nvidia-smi が見つかりません。GPU が無いと分離にとても時間がかかります。" >&2

step "Python 環境（.venv）"
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
export VIRTUAL_ENV="$DIR/.venv"

step "torch（$TORCH_INDEX）"
uv pip install torch==2.11.0 torchaudio==2.11.0 --index-url "$TORCH_INDEX"

step "依存パッケージ（requirements.txt）"
uv pip install -r requirements.txt
uv pip install --no-deps basic-pitch==0.4.0

clone() {  # clone URL DIR COMMIT
  if [ ! -d "$2/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone -q "$1" "$2"
  fi
  git -C "$2" -c advice.detachedHead=false checkout -q "$3"
}

step "コード認識（vendor/BTC-ISMIR19）"
mkdir -p vendor
clone https://github.com/jayg996/BTC-ISMIR19 vendor/BTC-ISMIR19 "$BTC_COMMIT"

if [ "$FULL" = 1 ]; then
  step "高精度の MIDI 化（requirements-full.txt）"
  uv pip install -r requirements-full.txt

  step "YourMT3+（vendor/YourMT3）"
  clone https://huggingface.co/spaces/mimbres/YourMT3 vendor/YourMT3 "$YMT3_COMMIT"
  ckpt="vendor/YourMT3/$YMT3_CKPT"
  if ! echo "$YMT3_SHA256  $ckpt" | sha256sum -c --status 2>/dev/null; then
    echo "重みをダウンロード（561MB）"
    mkdir -p "$(dirname "$ckpt")"
    curl -fL --progress-bar -o "$ckpt.part" "https://huggingface.co/spaces/mimbres/YourMT3/resolve/main/$YMT3_CKPT"
    echo "$YMT3_SHA256  $ckpt.part" | sha256sum -c --quiet
    mv "$ckpt.part" "$ckpt"
  fi
fi

step "確認"
.venv/bin/python - <<'EOF'
import torch
from mixer import transcribe, yourmt3
print("CUDA:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "なし（CPU で動く）")
print("YourMT3:", "あり" if yourmt3.available() else "なし（自動はボーカルを Basic Pitch で作る）")
print("MuScriptor:", "あり" if transcribe.muscriptor_available() else "なし（自動はストリングスを Basic Pitch で作る）")
EOF

cat <<'EOF'

セットアップ完了。
  ./clip.sh inbox/曲.flac 60 30      # 30秒のテストクリップを作る
  ./separate.sh inbox/clips/...wav   # 分離（初回はモデルのダウンロードがあり時間がかかる）
  ./mixer.sh                         # ブラウザ用ミキサー（表示される URL を開く）
EOF
if [ "$FULL" = 1 ]; then
  cat <<'EOF'
MuScriptor の重みは、https://huggingface.co/MuScriptor/muscriptor-medium でライセンス（CC BY-NC 4.0）に
同意してから `uvx hf auth login` でログインすると、初回の MIDI 化のときに自動でダウンロードされる。
EOF
fi
