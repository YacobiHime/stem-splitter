# 採用モデル一覧

取得日: 2026-10-05。取得経路は pymss カタログ（[baicai1145/pymss](https://huggingface.co/baicai1145/pymss) の HuggingFace ミラー。リポジトリの表記は Apache-2.0）。
ただしこのミラーの表記は、元の重みの権利を保証するものではない。そのため下の「ライセンス」欄には**オリジナル配布元の表記**を書いている。
全ファイルについて、オリジナル配布元の LFS sha256（GitHub のものはリリース asset の digest）と一致することを確認済み。
一覧は `SHA256SUMS.txt` にもある。

用途は個人の耳コピ練習のみ。分離結果も重みも再配布しない。

## 通常パイプライン

| 役割 | pymss 名 | オリジナル出典 | ライセンス表記 | sha256 |
|---|---|---|---|---|
| ① ボーカル分離 | `BS-Roformer-Resurrection.ckpt` | https://huggingface.co/pcunwa/BS-Roformer-Resurrection | 表記なし | `9dbfe5cb572e4ed32a15ec727d7bd06c8d7aba97509e6fda5bc008bb1e0b2dd5` |
| ② Lead/Backing | `bs_roformer_karaoke_frazer_becruily.ckpt` | https://huggingface.co/becruily/bs-roformer-karaoke | 表記なし | `eb90ee24c1154d83fbcfd27e96182f19e061557cc6e4746953125e08c29389f9` |
| ③ 6ステム | `BS-Roformer-SW.ckpt` | https://huggingface.co/enerjazzer/BS-ROFO-SW-Fixed （元の jarredou アカウントは削除済み。作者は未確認） | **unknown**（[出所に関する Issue](https://github.com/openmirlab/bs-roformer-infer/issues/3)） | `24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e` |
| ④ Strings | `gilliaan_bowedstrings_bs_v1.ckpt` | https://huggingface.co/gilliaan/Stem-Separation-Models （`BowedStrings/BandSplitRoformer/gilliaan_bsroformer_bowedstrings_v1.ckpt`） | 表記なし | `282fabc28fb106edcc5e5e8383ea36559602fe53d0c406c736e08f58d66710fc` |
| ⑤ synth | `bs_mega_53stem_synth_mvsep.ckpt` | https://huggingface.co/noblebarkrr/BS-Roformer-MVSep-Mega-53-stems （v1。MVSep Mega 53 を単ステムに分割したもの） | 表記なし（元モデルは MIT リポジトリのリリースで配布） | `04749df7e2fab770a0d26bf9aa4850c1a6cd89547fc8bfe9c9d223602c0afb82` |
| ⑤ keys | `bs_mega_53stem_keys_mvsep.ckpt` | 同上 | 同上 | `4bc2bde8c5b26d5e22720304deace45dfeaffb2d02dc07922a8f2be17d82c844` |
| ⑤ organ | `bs_mega_53stem_organ_mvsep.ckpt` | 同上 | 同上 | `14c1c2a26fbd5061be85a31a170b6989ba296871895905ea6a1a048d200895ba` |
| ⑤ digital-piano | `bs_mega_53stem_digital-piano_mvsep.ckpt` | 同上 | 同上 | `481c741ff36208c9b11860297ff5c9a502873662724d35a437bf64aaade890bd` |
| ⑥ back-vocal | `bs_mega_53stem_back-vocal_mvsep.ckpt` | 同上 | 同上 | `9e60a9057340574977b0f858084113d952ba4731a7bbe054bf49d1d5338ca9d9` |

## --explore 用

| 役割 | pymss 名 | オリジナル出典 | ライセンス表記 | sha256 |
|---|---|---|---|---|
| Mega 53（全53ステム） | `mvsep_mega_model_bs_roformer_53_stems_v1.ckpt` | https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.21 | 配布元リポジトリは MIT（重み単体の表記はなし） | `c62820893bbf86d4e734f966bd142d9157cfc8bb8e79e9d8f9ea553f3ff3519f` |

単ステム版とフル版の出力は、テストクリップでビット単位まで一致することを確認した（差分 0）。

## 比較用（--optional）

| 役割 | pymss 名 | オリジナル出典 | ライセンス表記 | sha256 |
|---|---|---|---|---|
| ① ボーカル（比較） | `becruily_deux.ckpt` | https://huggingface.co/becruily/mel-band-roformer-deux | **CC BY-NC 4.0** | `10255c02295bf3e3865d4ee50ff752d7b19b124ed5fd93b147babc4333eda3aa` |
| ② Lead/Back（比較） | `mel_band_roformer_bve_gonza.ckpt` | https://huggingface.co/Gonzaluigi/Mel-Band-Roformer-BVE-Gonzaluigi | 表記なし | `e003bbb97ebef78c59cd6b46c5fc3d0f2303cd49f6fb98eb8b8f0f8075899ae3` |
| ④ Strings v2（比較） | `gilliaan_bowedstrings_bs_v2`（`pymss register` で登録） | https://huggingface.co/gilliaan/Stem-Separation-Models （`BowedStrings/BandSplitRoformer/gilliaan_bsroformer_bowedstrings_v2.ckpt`）。ファイルは `custom/gilliaan/` | 表記なし | `e49feede4125404930ea4dd2cd634b8a5850cc9e24dc0565a01ccac466a5bbff` |
| ④ Strings（比較） | `bs_mega_53stem_bowed_strings_mvsep.ckpt` | noblebarkrr（上と同じ） | 表記なし | `598a644d78dd6a385140b2134733eef42d9525805a84d55f224c80a1ca39ff61` |
| ③ Piano（比較） | `bs_mega_53stem_piano_mvsep.ckpt` | noblebarkrr（上と同じ） | 表記なし | `8e58ad544386136028ec1f5764bb95a235235492144112d94782ff1333905fe9` |

## 再登録手順（カスタムモデル）

```bash
.venv/bin/pymss register gilliaan_bowedstrings_bs_v2 --type bs_roformer \
  --model "$PWD/models/custom/gilliaan/gilliaan_bsroformer_bowedstrings_v2.ckpt" \
  --config "$PWD/models/custom/gilliaan/gilliaan_bsroformer_bowedstrings_v2.yaml"
```

## ミキサーの解析用（拍・小節・コード）

`mixer/analysis.py` が使う。どちらも無ければ librosa だけの方法（精度は低い）に自動で戻る。

| 役割 | 入手方法 | オリジナル出典 | ライセンス表記 | sha256 |
|---|---|---|---|---|
| コード認識（170種） | `git clone https://github.com/jayg996/BTC-ISMIR19 vendor/BTC-ISMIR19`（commit `2682317`）。重みはリポジトリ同梱の `test/btc_model_large_voca.pt` | https://github.com/jayg996/BTC-ISMIR19 （Park et al., ISMIR 2019） | MIT | `1673d23f8f9a55ae7f9e8b80a51da616debb22675b8d8b67ea6ce0ef37b0ab51` |
| 拍・小節頭・拍子 | `VIRTUAL_ENV=.venv uv pip install git+https://github.com/CPJKU/beat_this.git`（1.1.0）。重みは初回に `~/.cache/torch/hub/checkpoints/beat_this-final0.ckpt` へ自動ダウンロード | https://github.com/CPJKU/beat_this （Foscarin et al., ISMIR 2024） | MIT（コードと重み） | `8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331` |

## ミキサーの「分離の設定」（カスタム分離など）で使うもの

初めて使うときに pymss カタログ（HuggingFace ミラー）から自動ダウンロードされる。オリジナル配布元とライセンス表記は未確認。

| 役割 | pymss 名 | sha256 |
|---|---|---|
| ドラムの分解（キック/スネア/タム/ハイハット/ライド/クラッシュ） | `MDX23C-DrumSep-aufr33-jarredou.ckpt` | `d2a4aa53eb584d21eead358a4e66d1882ad182911be018f052b5da73be9096d0` |
| アコギ（エレキ = ギター − アコギ） | `bs_mega_53stem_acoustic-guitar_mvsep.ckpt` | `fa386b2e7b1ea4f12b9b5c557444c0dc78648ef4ee299de2759d86457e182b3e` |
| 木管 | `bs_mega_53stem_woodwind_mvsep.ckpt` | `9a1335d2cd21cd4fdd6964e7ab96c9e9776b2643644f763e2f59bb6c53d39060` |
| 金管 | `bs_mega_53stem_brass_mvsep.ckpt` | `e7d7bf86a031f3253019ff9dc424485d8085946fc428ab66522c568ee17b8027` |
| 会話 / サウンドトラック / 効果音 | `model_bandit_plus_dnr_sdr_11.47.ckpt` | `c48284779f7d1258a6527d3aaa18a532d45c1f506e2dcc25d5ab179a8c5e2573` |

| コーラスを声質で2つ（男声/女声モデル） | `model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt` | `123c00786bdbc6bd462dddb35cd21fd6ae99ab8319f93f63a8abc1012e593d94` |
| コーラスを2人の歌い手に | `model_mel_band_roformer_ep_0_sdr_7.9319_fixed.ckpt` | `9ae8ae1a7d2fab0a7c405fee27ac8332702a31d689de58c9fb6f63b30410832b` |
| コーラスを4声（S/A/T/B、合唱用） | `model_scnet_ep_36_sdr_5.4596.ckpt` | `1e104ce6f6733542f8356ac2e59f2b7ece49ebfd2713d1ff635806de5ceb483f` |
| まとめたパートからピアノ（「さらに分離」） | `bs_mega_53stem_piano_mvsep.ckpt` | `8e58ad544386136028ec1f5764bb95a235235492144112d94782ff1333905fe9` |

ほかに、既存の Resurrection / karaoke / BS-Roformer-SW / gilliaan strings / Mega53 の synth・organ・keys を組み合わせて使う。

コーラスを分けるモデルについて（FIRE BIRD・春日影の30秒で試した結果）: どれも女性ボーカルの曲なので、男声/女声モデルは「声の高さ」ではなく声質で分けている
（2つの出力の中心の音の高さはほぼ同じ）。そのため画面では「声1 / 声2」と表示する。karaoke モデルをコーラスにかける方法も試したが、全部が片方に入って分けられなかった。

## ノーツ表示（MIDI 化）

`mixer/transcribe.py` が使う。

| 役割 | 入手方法 | オリジナル出典 | ライセンス表記 |
|---|---|---|---|
| MIDI 化（標準・どの楽器も） | `VIRTUAL_ENV=.venv uv pip install --no-deps basic-pitch==0.4.0` と `uv pip install onnxruntime-gpu pretty_midi resampy mir_eval audioread "setuptools<70"`。basic-pitch の依存指定（TensorFlow 2.15 以下）は Python 3.12 用が無いので `--no-deps` で入れ、ONNX 版のモデル（パッケージ同梱の `nmp.onnx`）だけを使う。`resampy` が `pkg_resources` を使うため setuptools は 70 未満 | https://github.com/spotify/basic-pitch （Bittner et al., ICASSP 2022） | Apache-2.0 |
| MIDI 化（ピアノ専用・任意） | `uv pip install piano_transcription_inference`。重みは初回に `~/piano_transcription_inference_data/` へ自動ダウンロード（約165MB） | https://github.com/qiuqiangkong/piano_transcription_inference （Kong et al., 2021） | Apache-2.0 |
| MIDI 化（ストリングスの自動選択先） | `uv pip install muscriptor`（0.3.0）。重み `MuScriptor/muscriptor-medium` は HuggingFace でライセンスに同意し `uvx hf auth login` したあと、初回に自動ダウンロード（`~/.cache/huggingface/`） | https://github.com/muscriptor/muscriptor （Kyutai・Mirelo 2026, arXiv 2607.08168） | コード MIT、重み **CC BY-NC 4.0**（個人利用のみ） |
| MIDI 化（高精度・ボーカルの自動選択先） | `GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/spaces/mimbres/YourMT3 vendor/YourMT3`（commit `5e66c1e`）のあと、重み `amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt`（561MB、「YPTF.MoE+Multi (noPS)」）を `https://huggingface.co/spaces/mimbres/YourMT3/resolve/main/<そのパス>` から取得。依存は `uv pip install "lightning>=2.2.1,<2.7" transformers==4.45.1 deprecated`（transformers の内部 API を使うので 4.45.1 に固定。wandb は不要でスタブにしている）。`mixer/yourmt3.py` が別プロセスで動かす | https://github.com/mimbres/YourMT3 （Chang et al., 2024 "YourMT3+"） | GitHub のコードは GPL-3.0、Space の表記は Apache-2.0（個人利用のみ・再配布しない）。重みの sha256 `ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25`（Space の LFS と一致） |
