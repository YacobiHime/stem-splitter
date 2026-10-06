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

ほかに、既存の Resurrection / karaoke / BS-Roformer-SW / gilliaan strings / Mega53 の synth・organ・keys を組み合わせて使う。
