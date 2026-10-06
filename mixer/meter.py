"""Beats, bar lines and time signatures (any meter, changes allowed) with beat_this.

beat_this (CPJKU, ISMIR 2024, MIT; pip package beat-this) gives framewise beat / downbeat
probabilities. From them:
  1. pulse grid   beat_this beats; where it switched to half time, the missing beats are filled
                  in, so every pulse has (about) the same length -> one cell per pulse on the page
  2. bars         a bar-pointer HMM over the pulses: the state is (pulses per bar M, position);
                  M is 2..7, 9 or 12, may change from one bar to the next (rarely), common meters are
                  preferred; a bar starts where beat_this's downbeat probability is high and/or the
                  harmony changes (beat_this alone marks only every other bar line in some sections,
                  and misses odd meters like 5/4). 8 is left out: 8 pulses are two bars of 4 in practice
  3. signature    M with a guessed note value: 6 / 9 / 12 fast pulses -> /8 (compound), etc.
"""

import numpy as np

FPS = 50                     # beat_this frame rate
METERS = (2, 3, 4, 5, 6, 7, 9, 12)
# log prior of each meter when a song (or a new section) starts / the meter changes
METER_PRIOR = {4: 0.0, 3: -0.7, 6: -0.9, 2: -1.6, 5: -1.8, 7: -1.8, 9: -2.2, 12: -2.6}
METER_CHANGE = 0.01          # probability that the next bar has a different number of pulses

_model = None


def available():
    try:
        import beat_this  # noqa: F401
        return True
    except ImportError:
        return False


def _frames(y, sr):
    global _model
    import torch
    from beat_this.inference import Audio2Frames

    if _model is None:
        _model = Audio2Frames(checkpoint_path="final0", device="cuda" if torch.cuda.is_available() else "cpu")
    beat, down = _model(y, sr)
    return beat.cpu(), down.cpu()


def pulse_period(ibi):
    """The shortest beat length that a good share of the song uses (beat_this switches between
    e.g. 97 and 194 BPM readings section by section; the faster one is the pulse)."""
    lg = np.log2(ibi)
    hist, edges = np.histogram(lg, bins=np.arange(lg.min() - 0.05, lg.max() + 0.1, 0.05))
    peaks = [i for i in range(len(hist)) if hist[i] > 0 and hist[i] == hist[max(0, i - 2):i + 3].max()]
    share = hist / hist.sum()
    # mass within +-10% of each peak
    def mass(i):
        c = (edges[i] + edges[i + 1]) / 2
        return float(np.mean(np.abs(lg - c) < 0.14))
    for i in sorted(peaks, key=lambda i: edges[i]):
        if mass(i) >= 0.15 and share[i] > 0:
            c = (edges[i] + edges[i + 1]) / 2
            return float(np.median(ibi[np.abs(lg - c) < 0.14]))
    return float(np.median(ibi))


def regular_pulses(beats, period):
    """Fill gaps of 2-4 pulses with evenly spaced beats and drop beats that come far too early."""
    out = [beats[0]]
    for b in beats[1:]:
        g = b - out[-1]
        if g < 0.55 * period:
            continue
        k = int(round(g / period))
        if 2 <= k <= 4:
            prev = out[-1]          # (not out[-1] inside the loop: the list grows while filling)
            out.extend(prev + g * j / k for j in range(1, k))
        out.append(b)
    return np.array(out)


def secondary_positions(m):
    """Positions inside a bar of m pulses where harmony also changes fairly often."""
    return {2: {1}, 3: set(), 4: {2}, 5: {2, 3}, 6: {3}, 7: {3, 4}, 8: {4}, 9: {3, 6}, 12: {6}}.get(m, set())


def signature(m, pulse_bpm):
    """Guess the written time signature of a bar of m pulses."""
    if m in (6, 9, 12) and pulse_bpm >= 130:
        return f"{m}/8"
    if m in (5, 7) and pulse_bpm >= 170:
        return f"{m}/8"
    return f"{m}/4"


def bar_pointer(act, change=None):
    """Viterbi over (meter, position) states. act[t] = beat_this downbeat probability at pulse t,
    change[t] = harmonic change 0..1 (an independent second cue).
    Returns position-in-bar and bar length for every pulse."""
    T = len(act)
    states = [(m, p) for m in METERS for p in range(m)]
    S = len(states)
    first = np.array([p == 0 for m, p in states])
    last = [i for i, (m, p) in enumerate(states) if p == m - 1]
    start_of = {m: states.index((m, 0)) for m in METERS}
    prior = np.array([METER_PRIOR[m] for m, p in states])
    a = np.clip(act, 0.05, 0.95)
    on, off = np.log(a), np.log(1 - a)
    if change is not None:
        # one-sided: a chord change argues for a bar line, but no change argues for nothing
        # (chords often last two bars or more)
        c = 0.5 + 0.4 * np.asarray(change)
        on, off = on + np.log(c), off + np.log(1 - c)
    emit = np.where(first[None, :], on[:, None], off[:, None])      # (T, S)
    # meter change at a bar line: from meter m to m'
    to_m = np.array([[np.log(1 - METER_CHANGE) if m2 == m1 else np.log(METER_CHANGE) + METER_PRIOR[m2] for m2 in METERS]
                     for m1 in METERS])
    nxt = np.arange(S) + 1                     # within a bar: (m, p) -> (m, p + 1)
    dp = prior - np.log(np.array([m for m, p in states], dtype=float)) + emit[0]       # any phase at the start
    back = np.zeros((T, S), dtype=np.int16)
    for t in range(1, T):
        new = np.full(S, -np.inf)
        bp = np.zeros(S, dtype=np.int16)
        inner = [i for i in range(S) if not first[i]]
        new[inner] = dp[np.array(inner) - 1]
        bp[inner] = np.array(inner) - 1
        lasts = dp[last]                                                   # (len(METERS),) score of finishing a bar
        for j, m2 in enumerate(METERS):
            cand = lasts + to_m[:, j]
            k = int(np.argmax(cand))
            new[start_of[m2]] = cand[k]
            bp[start_of[m2]] = last[k]
        dp = new + emit[t]
        back[t] = bp
    s = int(np.argmax(dp))
    path = [s]
    for t in range(T - 1, 0, -1):
        s = int(back[t][s])
        path.append(s)
    path = path[::-1]
    return np.array([states[i][1] for i in path]), np.array([states[i][0] for i in path])


def harmonic_change(harm, sr, pulses, period):
    """0..1 per pulse: how different the harmony (chroma) is in the 2 pulses after vs before it."""
    import librosa

    hop = 512
    ch = librosa.feature.chroma_stft(y=harm, sr=sr, n_fft=4096, hop_length=hop)
    ch = ch / (np.linalg.norm(ch, axis=0, keepdims=True) + 1e-9)
    w = max(1, int(round(2 * period * sr / hop)))
    fr = np.round(pulses * sr / hop).astype(int)
    out = np.zeros(len(pulses))
    for i, f in enumerate(fr):
        a, b = ch[:, max(0, f - w):f], ch[:, f:f + w]
        if a.shape[1] and b.shape[1]:
            out[i] = 1 - float(np.dot(a.mean(axis=1), b.mean(axis=1)) /
                               (np.linalg.norm(a.mean(axis=1)) * np.linalg.norm(b.mean(axis=1)) + 1e-9))
    # rank-normalize: the top ~25% of pulses (candidates for chord changes) get values near 1
    r = np.argsort(np.argsort(out)) / max(1, len(out) - 1)
    return np.clip((r - 0.5) * 2, 0, 1) ** 2


def split_doubled(pos, length):
    """6/8 and 12/8 are the same groove written two ways; when a song has both, write all of it in 6."""
    if (length == 6).any() and (length == 12).any():
        twelve = length == 12
        pos, length = pos.copy(), length.copy()
        pos[twelve] %= 6
        length[twelve] = 6
    return pos, length


def track(y, sr, harm=None):
    """Pulse times, bar starts (pulse indices), per-pulse position / bar length, pulse BPM.
    harm: the harmonic part of the song (no drums / lead) at the same rate, for the chord-change cue."""
    from beat_this.model.postprocessor import Postprocessor

    beat_logits, down_logits = _frames(y, sr)
    beats, _ = Postprocessor(type="minimal", fps=FPS)(beat_logits, down_logits)
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 8:
        return None
    period = pulse_period(np.diff(beats))
    pulses = regular_pulses(beats, period)
    # downbeat probability at each pulse (max over +-20% of a pulse)
    prob = 1 / (1 + np.exp(-down_logits.numpy()))
    w = max(1, int(round(0.2 * period * FPS)))
    fr = np.round(pulses * FPS).astype(int)
    act = np.array([prob[max(0, f - w):f + w + 1].max() if f < len(prob) else 0.0 for f in fr])
    pos, length = bar_pointer(act, None if harm is None else harmonic_change(harm, sr, pulses, period))
    pos, length = split_doubled(pos, length)
    bpm = 60.0 / float(np.median(np.diff(pulses)))
    return {"pulses": pulses, "pos": pos, "length": length, "bpm": bpm}
