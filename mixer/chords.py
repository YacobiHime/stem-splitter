"""Chord recognition with BTC (Bi-directional Transformer for Chord recognition, Park et al., ISMIR 2019).

Model code and pretrained weights: vendor/BTC-ISMIR19 (MIT license,
https://github.com/jayg996/BTC-ISMIR19). The large-vocabulary model knows 170 labels:
12 roots x {maj, min, dim, aug, min6, maj6, min7, minmaj7, maj7, 7, dim7, hdim7, sus2, sus4} + N + X.

frame_log_probs() gives (frames, 170) log-probabilities at ~10.8 frames/s; analysis.py
averages them over each beat and smooths the beat sequence.
"""

import sys
from pathlib import Path

import numpy as np

BTC_DIR = Path(__file__).resolve().parent.parent / "vendor" / "BTC-ISMIR19"
SR = 22050
N_BINS, BINS_PER_OCTAVE, HOP = 144, 24, 2048
INST_LEN = 10.0                  # the model computes the CQT in 10 s pieces of 108 frames
TIMESTEP = 108
FRAME_S = INST_LEN / TIMESTEP
QUALITIES = ["min", "maj", "dim", "aug", "min6", "maj6", "min7", "minmaj7", "maj7", "7", "dim7", "hdim7", "sus2", "sus4"]
N_LABELS = 170                   # 168 chords + X (unknown) + N (no chord)

_model = None


def label(idx):
    """Our label for a BTC class: '<root pc>:<quality>' ('2:min7'), 'N' or 'X'."""
    if idx == 169:
        return "N"
    if idx == 168:
        return "X"
    return f"{idx // 14}:{QUALITIES[idx % 14]}"


def available():
    return (BTC_DIR / "test" / "btc_model_large_voca.pt").exists()


def _load():
    global _model
    if _model is not None:
        return _model
    import torch

    if str(BTC_DIR) not in sys.path:
        sys.path.insert(0, str(BTC_DIR))
    if not hasattr(np, "float"):   # the 2019 code uses np.float (removed in numpy 1.24); keep vendor/ unmodified
        np.float = float
    from btc_model import BTC_model  # noqa: E402  (vendored module)

    cfg = dict(feature_size=N_BINS, timestep=TIMESTEP, num_chords=N_LABELS, input_dropout=0.2, layer_dropout=0.2,
                  attention_dropout=0.2, relu_dropout=0.2, num_layers=8, num_heads=4, hidden_size=128,
                  total_key_depth=128, total_value_depth=128, filter_size=128, loss="ce", probs_out=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # the checkpoint stores numpy mean/std next to the weights, so it needs the full unpickler (local, trusted file)
    ckpt = torch.load(BTC_DIR / "test" / "btc_model_large_voca.pt", map_location=device, weights_only=False)
    model = BTC_model(config=cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    _model = (model, float(ckpt["mean"]), float(ckpt["std"]), device)
    return _model


def features(y):
    """Log-magnitude CQT exactly as BTC was trained on (computed in 10 s pieces)."""
    import librosa

    step = int(SR * INST_LEN)
    parts = [librosa.cqt(y[s:s + step], sr=SR, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, hop_length=HOP)
             for s in range(0, max(len(y), 1), step) if len(y[s:s + step]) > HOP]
    return np.log(np.abs(np.concatenate(parts, axis=1)) + 1e-6).T          # (frames, 144)


def frame_log_probs(y):
    """(frames, 170) log-probabilities for mono audio y at 22050 Hz; frame k starts at k * FRAME_S."""
    import torch
    import torch.nn.functional as F

    model, mean, std, device = _load()
    x = (features(y) - mean) / std
    n = x.shape[0]
    pad = (-n) % TIMESTEP
    x = torch.tensor(np.pad(x, ((0, pad), (0, 0))), dtype=torch.float32, device=device).unsqueeze(0)
    out = []
    with torch.no_grad():
        for t in range(x.shape[1] // TIMESTEP):
            h, _ = model.self_attn_layers(x[:, t * TIMESTEP:(t + 1) * TIMESTEP])
            out.append(F.log_softmax(model.output_layer.output_projection(h), -1)[0])
    return torch.cat(out).cpu().numpy()[:n]
