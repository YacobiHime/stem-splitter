# stem-sep — キーボード耳コピ用ステム分離

Ave Mujica / Roselia / MyGO!!!!! などの曲から Piano / Strings / Synth / Keys / Organ / Digital Piano / Chorus を抜き出し、
Audacity で並べて聴き比べるためのツール。**分離結果は個人の練習専用。アップロード・公開・Git へのコミットはしない。**

## 使い方

```bash
cd ~/music_split_server
./separate.sh inbox/song.flac            # 1曲
./separate.sh inbox/*.flac               # 複数曲（フォルダ指定 ./separate.sh inbox も可）
./separate.sh --force song.wav           # 既存の出力を無視して全ステップをやり直す
./separate.sh --explore song.wav         # Mega 53 の全53ステムを output/<曲名>/explore/ に保存
./separate.sh --optional song.wav        # 比較用モデルも実行（結果は work/alt_*.wav）
./separate.sh --clean song.wav           # 出力を作ったあと work/ を削除（容量節約）
.venv/bin/python check.py                # 全曲の出力を検証（output/<曲名> を指定してもよい）
./clip.sh inbox/song.flac 60 30          # 60秒目から30秒のテストクリップを inbox/clips/ に作成
```

| オプション | 内容 |
|---|---|
| `--force` | 既存の出力があってもスキップせず再処理する |
| `--explore` | `config.yaml` の `explore.model`（Mega 53 フル版）を upper に適用し、全ステムを `explore/` に保存する |
| `--optional` | `optional: true` のステップ（比較用モデル）も実行する |
| `--clean` | 正常に終わった曲の `work/` を削除する。出力ファイルは残るが、次に実行すると全ステップをやり直す |
| `--keep-work` | 互換用。何も変わらない（`work/` はもともと残す） |
| `--config FILE` / `--output DIR` | 設定ファイル / 出力先を変える |

途中で失敗・中断しても、もう一度同じコマンドを実行すれば続きから処理する。
ステップの出力が揃っていて、入力より新しく、同じモデルで作られていればスキップする。入力やモデルが変わったステップと、それに依存するステップだけを再実行する。

## 処理の流れ（カスケード）

```text
入力 ─ffmpeg(soxr)→ original (44.1kHz/stereo/float32)
  │
  ├─[vocals]  BS-Roformer-Resurrection ─┬─ vocals ─┬─[karaoke] frazer&becruily ─→ lead / backing(=chorus)
  │                                      │          └─[mega_back_vocal] Mega53 back-vocal ─→ chorus-alt
  │                                      └─ instrumental
  │                                             │
  │                    [six_stem] BS-Roformer-SW: drums, bass, guitar, piano, other, (vocals)
  │                                             │
  │                    [upper] instrumental − drums − bass   (または guitar + piano + other)
  │                                             │
  │        ┌─────────────────┬─────────────────┼──────────────┬──────────────┐
  │   [strings]          [mega_synth]      [mega_keys]   [mega_organ]  [mega_digital_piano]
  │   gilliaan v1         Mega53 単ステム版（フル版と出力はビット単位で一致）
  ▼
output/<曲名>/<曲名>_<楽器名>.m4a  ＋ manifest.json ＋ log.txt
```

## 出力

```text
output/<曲名>/
├── <曲名>_original.m4a  原曲（変換・ゲイン調整後）
├── <曲名>_upper.m4a     ボーカル・ドラム・ベース抜き（耳コピの基準トラック）
├── <曲名>_piano.m4a     BS-Roformer-SW の piano
├── <曲名>_strings.m4a   Bowed Strings 専用モデル
├── <曲名>_synth.m4a     Mega53 synth
├── <曲名>_keys.m4a      Mega53 keys
├── <曲名>_organ.m4a     Mega53 organ
├── <曲名>_digital-piano.m4a Mega53 digital-piano
├── <曲名>_chorus.m4a    karaoke モデルの backing
├── <曲名>_chorus-alt.m4a Mega53 back-vocal（vocals に適用）
├── <曲名>_lead.m4a      karaoke モデルの lead
├── manifest.json         使用モデル・sha256・各ステップの時間と VRAM・各ファイルのピーク/RMS
├── log.txt               実行ログ（各ステップの所要時間と VRAM ピーク）
├── explore/              --explore 時のみ（53ステム）
└── work/                 中間ファイル（vocals, instrumental, drums, bass, upper_subtract, upper_sum, alt_* …）
```

- 出力は **m4a（AAC 256kbps）**・44.1kHz・ステレオ・全ファイル同じ長さ。ステムごとの音量正規化はしない。
  形式は `config.yaml` の `output_format` で変えられる（`aac` / `alac`＝ロスレス m4a / `wav`＝32bit float）。
  `work/` と `explore/` は、モデルの入出力に使うので常に 32bit float WAV。
  AAC はファイル末尾に約13ms の無音の詰め物が付くが、先頭はそろっている（Audacity で並べてもずれない）。
- **何も検出されなかったパートは保存しない**（`drop_silent`）。判定は、そのファイルで一番大きい1秒間の RMS が、原曲の同じ値より 30dB 以上小さいとき。
  短いソロしか無いパートも、その1秒が大きければ残る。保存しなかったものは log.txt と manifest の `dropped_silent` に記録され、work/ には残る。
- どれかのファイルのピークが上限（AAC は -1dBFS、それ以外は 1.0）を超えた場合だけ、**全ファイルに同じゲイン**をかけて収める（`global_gain_on_clip`。パート間の音量関係は保たれる。かけたゲインは manifest の `global_gain` に記録される）。
- `wav` 形式のときは、内容が `work/` と同じファイルをハードリンクにしている（容量を食わない）。

## 設定（config.yaml）

モデルの差し替え・保存するステム・chunk size などはすべて `config.yaml` で変えられる（コードの変更は不要）。

- `steps[].model`：`pymss list` に出る名前、または `pymss register` で登録した名前
- `steps[].outputs`：`{モデルのステム名: work/ に保存する名前}`。ステム名はモデル YAML の `training.instruments` を見る
- `steps[].inference_params`：`chunk_size` / `overlap_size` / `batch_size` などの上書き
- `upper_method`：`subtract`（instrumental − drums − bass）か `sum`（guitar + piano + other）。
  `work/upper_subtract.wav` と `work/upper_sum.wav` は常に両方作るので、聴き比べてから決めればよい。
  変更すると、upper を入力にしているステップだけが自動で再実行される。
- `deliverables`：楽器名と、その元になる work ファイルの対応（並び順もここで決まる）
- `filename_pattern`：出力ファイル名の形。既定は `{song}_{part}`（例 `天球のmujica_strings.m4a`）。
  `{nn}_{song}_{part}` にすると先頭に通し番号が付き、ファイル一覧で番号順に並ぶ
- `explore.stems_per_pass`：explore を何ステムずつ処理するか（下の VRAM の項を参照）

## モデル

詳細（出典 URL・ライセンス・sha256）は [models/MODELS.md](models/MODELS.md) を参照。

| 役割 | モデル | 比較用（--optional） |
|---|---|---|
| ボーカル | BS-Roformer-Resurrection (pcunwa) | becruily_deux |
| Lead/Back | bs_roformer_karaoke_frazer_becruily | mel_band_roformer_bve_gonza |
| ドラム/ベース/ピアノ | BS-Roformer-SW | Mega53 piano |
| Strings | gilliaan_bowedstrings_bs_v1 | gilliaan_bowedstrings_bs_v2（register 済み）, Mega53 bowed_strings |
| Synth/Keys/Organ/Digital Piano | Mega53 単ステム版 | — |
| Chorus（比較） | Mega53 back-vocal | — |
| --explore | Mega53 フル版（53ステム） | — |

## Audacity での推奨確認手順

1. `output/<曲名>/` の `<曲名>_*.m4a` を全部 Audacity にドラッグ＆ドロップする（トラック名＝ファイル名なので、楽器名で見分けられる）。
2. **upper をソロ**にして、上物全体の流れを掴む。
3. **piano → strings → synth / keys / organ / digital-piano** の順にソロで聴き、パートごとに音を拾う。
4. 迷ったら、そのパートと **original** を交互に（または両方）鳴らして、原曲で本当に鳴っているかを確かめる。
5. コーラスのハモりを拾うときは chorus と chorus-alt を聴き比べ、聴き取りやすいほうを使う。
6. 分離にはにじみが残る（例：synth に strings が少し混ざる）。複数のステムに同じフレーズが出たら、upper と原曲で判断する。

## Windows との受け渡し（共有なし：scp / WinSCP）

このサーバーではファイル共有を設定していない（ネットワークに公開されるディレクトリは無い）。
`<server>` はこのマシンの IP アドレスかホスト名（`hostname -I` で確認できる）。

**PowerShell（Windows 10/11 標準の OpenSSH）**
```powershell
# 曲を送る
scp "C:\Music\song.flac" vllm-admin@<server>:~/music_split_server/inbox/
# 結果を受け取る（フォルダごと）
scp -r "vllm-admin@<server>:~/music_split_server/output/<曲名>" "C:\Users\<you>\Music\stems\"
```
曲名に空白や記号が含まれるときは、リモート側のパスを `"..."` で囲む。うまくいかないときは WinSCP を使うのが確実。

**WinSCP**：新しいサイト → SFTP、ホスト `<server>`、ユーザー `vllm-admin` で接続 →
左（Windows）と右（`/home/vllm-admin/music_split_server/`）の間でドラッグ＆ドロップする。`inbox/` に置いて、`output/<曲名>/` を持ち帰る。
`work/` と `explore/` は大きいので、必要なときだけ持ち帰る。

Samba 共有（output は読み取り専用、inbox は書き込み可、LAN 内のみ・ゲスト無効）に切り替えたくなったら、
sudo が必要なので別途設定する。

## 処理時間と VRAM（RTX 4060 Ti 16GB、4分30秒の曲）

| ステップ | 時間 | VRAM ピーク（reserved） |
|---|---|---|
| vocals / karaoke / six_stem | 各 約7.5秒 | 2.8〜3.7 GB |
| strings | 約7秒 | 2.3 GB |
| Mega53 単ステム ×5 | 各 約6秒 | 1.6 GB |
| **通常処理の合計** | **約70秒** | **最大 3.7 GB** |
| --explore（14ステムずつ4パス） | 約35秒 | 6.1 GB |

## トラブルシュート

| 症状 | 対処 |
|---|---|
| `CUDA out of memory` | 他に GPU を使っているプロセスが無いか `nvidia-smi` で確認する。該当ステップの `inference_params` に `chunk_size` を小さめ（例 `441000`）に設定する。explore なら `stems_per_pass` を下げる |
| モデルのダウンロードが遅い・失敗する | `config.yaml` の `download_source` を `huggingface` / `modelscope` / `hf-mirror` に切り替える。手動なら `.venv/bin/pymss download <名前> --model-dir models --source huggingface` |
| 途中で止まった | 同じコマンドをもう一度実行する（終わっているステップはスキップされる）。原因は `output/<曲名>/log.txt` の `FAILED` 行とトレースバックを見る |
| 結果がおかしい・モデルを変えた | `--force` で全部やり直す。モデルを変えたステップと、それに依存するステップは自動で再実行される |
| check.py に `SILENT` が出る | その曲にそのパートが無い（例：オルガン不使用）ことが多い。エラーではなく警告 |
| check.py に `CLIP` が出る | 通常は `global_gain_on_clip: true` で自動的に防がれる。false にしている場合に出る |
| 容量が足りない | `--clean` で work/ を消す（1曲あたり 2〜3GB）。`explore/` も大きい（4分半の曲で 約5GB） |
| ffmpeg が無い | `sudo apt install ffmpeg` |

## ディレクトリ

```text
~/music_split_server/
├── separate.sh        入口
├── clip.sh            テストクリップを切り出す
├── check.py           出力の検証
├── config.yaml        モデル・ステップ・出力の設定
├── pipeline/          実処理（Python）
├── models/            モデル（MODELS.md, SHA256SUMS.txt）
├── vendor/            ZFTurbo/Music-Source-Separation-Training（予備。今は未使用）
├── inbox/             入力曲を置く
├── output/            結果
└── .venv/             uv 仮想環境（Python 3.12, torch 2.11+cu128, pymss 2.1.7）
```
