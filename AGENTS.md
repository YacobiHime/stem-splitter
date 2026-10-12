# AGENTS.md

キーボード耳コピ用のステム分離（`pipeline/`）と、LAN / Tailscale 経由で聴くブラウザ用ミキサー（`mixer/`）。
使い方・出力形式・設定の詳細は [README.md](README.md)、モデルの出典・ライセンス・sha256 は [models/MODELS.md](models/MODELS.md)。

## 絶対に守ること

- **音源・分離結果・モデル重みをコミットしない／外部に送らない。** 個人の練習専用（著作物）。`music/` `inbox/` `output/` `models/**`（MODELS.md と SHA256SUMS.txt を除く）は `.gitignore` 済み。
- `.mixer_token` を表示・ログ出力・コミットしない（ミキサーのアクセストークン）。
- 認証用 URL の `?k=...` もトークンとして扱う。コマンド出力・テスト結果・回答・アクセスログに値を残さない。Tailscale の IP / ホスト名だけなら共有できる。
- ミキサーをインターネットへ公開しない。Tailscale は端末を同じ tailnet に参加させる用途に限り、ルーターのポート開放や Tailscale Funnel は使わない。
- 新しいモデルを入れるときは、オリジナル配布元・ライセンス表記・sha256 を確認して `models/MODELS.md` と `models/SHA256SUMS.txt` に追記する。
- GPU は RTX 4060 Ti 16GB の1枚。重いモデルは同時に載せない（YourMT3 / MuScriptor はサブプロセスで動かし、終われば VRAM を返す）。

## 環境

- 環境は `./setup.sh`（`--full` で YourMT3+ と MuScriptor も）で作る。Python 3.12、uv の仮想環境 `.venv/`（torch + CUDA 12.8、pymss）。コマンドは `.venv/bin/python` で実行する。
- パッケージを足したら、バージョンを固定して `requirements.txt`（高精度の MIDI 化だけで使うものは `requirements-full.txt`）にも書く。`VIRTUAL_ENV=.venv uv pip install ...`。
- 外部リポジトリは `vendor/`（BTC-ISMIR19 = コード認識、YourMT3 = MIDI 化）。git 管理外。
- MuScriptor は `.venv/bin/muscriptor`（CLI）。無ければ `transcribe.muscriptor_available()` が False になり、自動モードは別のモデルに戻る。
- ffmpeg が必要。

## よく使うコマンド

```bash
./clip.sh inbox/song.flac 60 30           # 30秒のテストクリップ（inbox/clips/）。動作確認はまずこれで
./separate.sh inbox/clips/xxx.wav         # 分離（終わったステップはスキップ。--force で全部やり直し）
.venv/bin/python check.py output/<曲名>    # 出力の検証（長さ・無音・クリップ・mix の合計が原曲に一致するか）
./mixer.sh                                # ミキサーを前面起動（ポート 8000）。ログはターミナルへ出る
./mixer.sh --tailscale-auth               # 同じ Tailscale アカウントの端末はトークンなしで接続
```

テストスイートは無い。分離・出力に関わる変更後は、テストクリップで分離 → `check.py` → 必要ならミキサーで実際に聴く／表示を確認する。
ミキサーの認証に関わる変更後は、Tailscale の同一アカウントからの表示、未認証の LAN アクセス、別サイトからの POST / PUT、偽の Host ヘッダーを確認する。検証で実トークンを出力しない。

## 構成

| パス | 役割 |
|---|---|
| `pipeline/run.py` | カスケード分離の本体。`config.yaml` のステップを順に実行し、`output/<曲名>/` に出力と `manifest.json` を書く |
| `pipeline/audio.py` | 読み書き・変換（ffmpeg / soundfile） |
| `config.yaml` | モデル・ステップ・出力の設定。モデルの差し替えはここ（コード変更不要） |
| `check.py` | 出力の検証 |
| `mixer/server.py` / `mixer/tailscale_auth.py` | HTTP サーバー（トークン / Tailscale 認証、PCM 配信、書き出し、API） |
| `mixer/jobs.py` / `mixer/profiles.py` | ブラウザから追加した曲のバックグラウンド分離（`.mixer_jobs.json`）/ 分離の設定 → パイプライン設定への変換 |
| `mixer/playlists.py` | プレイリストの保存と操作（`.mixer_playlists.json`。API は `GET/POST /api/playlists`） |
| `mixer/transcribe.py` / `yourmt3.py` | パートの MIDI 化（Basic Pitch・Transkun・YourMT3+・MuScriptor）と楽器ごとの後処理 |
| `mixer/analysis.py` / `meter.py` / `chords.py` | 拍・小節・拍子（beat_this）、コード（BTC）、キー |
| `mixer/static/index.html` | 画面（1ファイル、ビルド無し） |

## 守るべき性質

- **`mix/` のトラックは全部足すと原曲に一致する**（差 −67dB 以下）。分離の設定やカスケードを変えたら `check.py` で確認する。新しいパートは「残り」から引き算で取り出し、最後の残りを「その他」にする形を崩さない。
- パイプラインは再開可能：ステップの出力が入力より新しく、同じモデルで作られていればスキップする。この判定を壊さない。
- 解析結果・MIDI はキャッシュされる。**出力が変わる変更をしたら** `mixer/analysis.py` の `VERSION` / `mixer/transcribe.py` の `VERSION` を上げる（作り直しのトリガー）。
- 精度に関わる変更は数値で比較する（例: MIDI はノート F1、分離は聴き比べ＋mix の残差）。README に実測値を残している箇所は、変えたら更新する。
- プレイリスト（`.mixer_playlists.json`）の `dirs` から、存在しない曲を自動で削除しない。描画・再生時にスキップするだけにする（曲を復元したら元の位置に戻る）。
- Tailscale 自動認証は、サーバー自身の Tailscale アドレスへの直接接続と、接続元端末の所有者がサーバーと同じことを Local API で確認する。別アカウント・タグ付き端末・LAN 接続・照会失敗は許可しない。ブラウザの変更操作には同一オリジンの確認を残す。

## コードの書き方

- コードのコメント・docstring は英語、README・MODELS.md・画面の文言は日本語。周りのコードに合わせる。
- 依存を増やす前に、既存のもの（numpy / soundfile / librosa / torch）で済まないか考える。
- 画面は `index.html` 1ファイルに収める（フレームワーク・ビルド無し）。

## 画面（index.html）を変えるとき

- `index.html` は `Cache-Control: no-cache` で配信されるので、UI だけの変更はブラウザの再読込で反映される。`server.py` を変えたときだけミキサーの再起動が必要。
- 構文チェックは `.venv/bin/python -m py_compile mixer/server.py` と、`<script>` 部分を取り出しての `node --check`。テストスイートは無いので、実際にブラウザで操作して確認する。
- ポップオーバーは `openPop(アンカー, HTML, 初期化)`（同じアンカーの再クリックで閉じる）。**開くたびに `innerHTML` が作り直される**ので、`pop` 自身に `addEventListener` すると開くたびに蓄積する。ハンドラは `pop.onchange = …` のようにプロパティ代入するか、作り直した要素に個別に付ける。状態は開くたびに `store` から読み直す。
- 選択肢は `.btns` ＋ `.btn.on`（ボタン群）か `.opt` カード（ラベル＋説明つき。ポップオーバー内では `.pop .opt` の上書きが効く）。説明文は `<p class="sub">`。文言は日本語、チェックポイントのファイル名は出さない。
- 端末ごとの状態は localStorage の `store` ラッパーに置く。使っているキー: `prefs`（マスター音量・メトロノーム・表示など全体設定）、`libSort`、`song`（最後に開いた曲）、`song:<曲dir>`（ミックスのフェーダー等）、`sepProfile`、`notesPrefs`（ノーツの見え方・表示ずれ）、`noteModel`、`playlist`（選択中のプレイリスト名）。**複数端末で共有すべきものはサーバー側に置く**（プレイリスト → `.mixer_playlists.json`、ジョブ → `.mixer_jobs.json`）。
- YourMT3+ / MuScriptor が導入済みかは `GET /api/models`（`yourmt3.available()` / `transcribe.muscriptor_available()`）で取れる。UI は未導入の選択肢を無効化し、サイレントにフォールバックさせない。
- 表示と音の同期は `frame()` の `visualPos()` で出力レイテンシを補正している。時刻を扱う変更をするときは、スケジューリング用の `currentPos()` と描画用の `visualPos()` を混ぜない。

## コミット

- メッセージは日本語で `feat: 内容。` の形（バグ修正は `fix:`、ほかも同様の接頭辞）。本文も日本語。
- 作業ブランチは `web-mixer`、メインは `main`。
