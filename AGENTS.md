# AGENTS.md

キーボード耳コピ用のステム分離（`pipeline/`）と、LAN 内のブラウザで聴くミキサー（`mixer/`）。
使い方・出力形式・設定の詳細は [README.md](README.md)、モデルの出典・ライセンス・sha256 は [models/MODELS.md](models/MODELS.md)。

## 絶対に守ること

- **音源・分離結果・モデル重みをコミットしない／外部に送らない。** 個人の練習専用（著作物）。`music/` `inbox/` `output/` `models/**`（MODELS.md と SHA256SUMS.txt を除く）は `.gitignore` 済み。
- `.mixer_token` を表示・ログ出力・コミットしない（ミキサーのアクセストークン）。
- 新しいモデルを入れるときは、オリジナル配布元・ライセンス表記・sha256 を確認して `models/MODELS.md` と `models/SHA256SUMS.txt` に追記する。
- GPU は RTX 4060 Ti 16GB の1枚。重いモデルは同時に載せない（YourMT3 / MuScriptor はサブプロセスで動かし、終われば VRAM を返す）。

## 環境

- Python 3.12、uv の仮想環境 `.venv/`（torch + CUDA 12.8、pymss）。コマンドは `.venv/bin/python` で実行する。パッケージ追加は `VIRTUAL_ENV=.venv uv pip install ...`。
- 外部リポジトリは `vendor/`（BTC-ISMIR19 = コード認識、YourMT3 = MIDI 化）。git 管理外。
- MuScriptor は `.venv/bin/muscriptor`（CLI）。無ければ `transcribe.muscriptor_available()` が False になり、自動モードは別のモデルに戻る。
- ffmpeg が必要。

## よく使うコマンド

```bash
./clip.sh inbox/song.flac 60 30           # 30秒のテストクリップ（inbox/clips/）。動作確認はまずこれで
./separate.sh inbox/clips/xxx.wav         # 分離（終わったステップはスキップ。--force で全部やり直し）
.venv/bin/python check.py output/<曲名>    # 出力の検証（長さ・無音・クリップ・mix の合計が原曲に一致するか）
./mixer.sh                                # ミキサー起動（ポート 8000）。ログは mixer.log
```

テストスイートは無い。変更後は、テストクリップで分離 → `check.py` → 必要ならミキサーで実際に聴く／表示を確認する。

## 構成

| パス | 役割 |
|---|---|
| `pipeline/run.py` | カスケード分離の本体。`config.yaml` のステップを順に実行し、`output/<曲名>/` に出力と `manifest.json` を書く |
| `pipeline/audio.py` | 読み書き・変換（ffmpeg / soundfile） |
| `config.yaml` | モデル・ステップ・出力の設定。モデルの差し替えはここ（コード変更不要） |
| `check.py` | 出力の検証 |
| `mixer/server.py` | HTTP サーバー（トークン認証、PCM 配信、書き出し、API） |
| `mixer/jobs.py` / `profiles.py` | ブラウザから追加した曲のバックグラウンド分離 / 分離の設定 → パイプライン設定への変換 |
| `mixer/transcribe.py` / `yourmt3.py` | パートの MIDI 化（Basic Pitch・Transkun・YourMT3+・MuScriptor）と楽器ごとの後処理 |
| `mixer/analysis.py` / `meter.py` / `chords.py` | 拍・小節・拍子（beat_this）、コード（BTC）、キー |
| `mixer/static/index.html` | 画面（1ファイル、ビルド無し） |

## 守るべき性質

- **`mix/` のトラックは全部足すと原曲に一致する**（差 −67dB 以下）。分離の設定やカスケードを変えたら `check.py` で確認する。新しいパートは「残り」から引き算で取り出し、最後の残りを「その他」にする形を崩さない。
- パイプラインは再開可能：ステップの出力が入力より新しく、同じモデルで作られていればスキップする。この判定を壊さない。
- 解析結果・MIDI はキャッシュされる。**出力が変わる変更をしたら** `mixer/analysis.py` の `VERSION` / `mixer/transcribe.py` の `VERSION` を上げる（作り直しのトリガー）。
- 精度に関わる変更は数値で比較する（例: MIDI はノート F1、分離は聴き比べ＋mix の残差）。README に実測値を残している箇所は、変えたら更新する。

## コードの書き方

- コードのコメント・docstring は英語、README・MODELS.md・画面の文言は日本語。周りのコードに合わせる。
- 依存を増やす前に、既存のもの（numpy / soundfile / librosa / torch）で済まないか考える。
- 画面は `index.html` 1ファイルに収める（フレームワーク・ビルド無し）。

## コミット

- メッセージは日本語で `feat: 内容。` の形（バグ修正は `fix:`、ほかも同様の接頭辞）。本文も日本語。
- 作業ブランチは `web-mixer`、メインは `main`。
