#!/usr/bin/env bash
# output/<曲名>/ の m4a に、inbox/ か music/ の <曲名>.m4a のカバー画像を埋め込む
# DRY=1（既定）なら確認だけ。DRY=0 で実際に書き換える
set -u
cd "$(dirname "$0")"
DRY="${DRY:-1}"
for dir in output/*/; do
  name="$(basename "$dir")"
  src=""
  for cand in "inbox/$name.m4a" "music/$name.m4a"; do
    if [ -f "$cand" ]; then src="$cand"; break; fi
  done
  if [ -z "$src" ]; then echo "元の曲が見つからない: $name"; continue; fi
  mapfile -d '' files < <(find "$dir" -name "*.m4a" -print0)
  for f in "${files[@]}"; do
    if ffprobe -v error -select_streams v -show_entries stream=index -of csv=p=0 "$f" | grep -q .; then
      echo "スキップ（画像あり）: $f"; continue
    fi
    if [ "$DRY" = 1 ]; then echo "付ける予定: $f ← $src"; continue; fi
    tmp="${f%.m4a}.cover-tmp.m4a"
    if ffmpeg -v error -y -i "$f" -i "$src" -map 0:a -map 1:v:0 -c copy -disposition:v:0 attached_pic "$tmp"; then
      mv "$tmp" "$f"; echo "付けた: $f"
    else
      rm -f "$tmp"; echo "失敗: $f"
    fi
  done
done
