"""Beat / chord / key estimation and waveform peaks for the mixer (librosa).

Everything is computed from the separated tracks in output/<song>/mix/:
  beats   mixer/meter.py: beat_this pulses (follows tempo drift; half-time sections filled in)
  bars    mixer/meter.py: bar-pointer HMM with any number of pulses per bar (2-7, 9, 12), meter
          changes allowed; signatures like 4/4, 3/4, 6/8, 5/4, 7/8
          Without beat_this: a constant-tempo librosa grid in 4/4.
  chords  BTC (mixer/chords.py, a pretrained Transformer with 170 chord labels incl. 7ths / sus / dim),
          run on the full mix and on the mix without drums and lead vocal, averaged; each beat is
          judged on its first part (an 8th-note anticipation at the end of a beat belongs to the
          next chord, as chord charts write it) and the beat sequence is smoothed with Viterbi,
          where chords change most easily on bar lines, then on the bar's inner strong beats
          (beat 3 of 4/4, the 2nd group of 6/8, 3+2 of 5/4, ...), rarely elsewhere.
          Without vendor/BTC-ISMIR19: chroma matched to 24 major/minor triads (the old method).
  key     Krumhansl-Schmuckler profiles on the whole-song chroma
The page lets the user correct the beat interpretation (x2, /2, 3/4, shift the bar lines).
"""

import numpy as np
import soundfile as sf

from . import chords as btc
from . import meter

VERSION = 5
SR = 22050
HOP = 512          # chroma frames (23 ms)
BEAT_HOP = 128     # beat tracking frames (5.8 ms), so the BPM is not quantized to 23 ms steps
NOTES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
# mixer track names differ by separation setting (mixer/profiles.py); group them by role
DRUM_PARTS = {"drums", "kick", "snare", "toms", "hihat", "cymbals", "drums-other"}
LEAD_PARTS = {"lead", "vocals", "speech"}            # melody / speech: left out of harmony
CHORUS_WEIGHT = {"chorus": 0.4}
PEAK_BINS = 2000

# Krumhansl-Kessler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def load_mono(path, sr=SR):
    import librosa

    a, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = a.mean(axis=1)
    return librosa.resample(y, orig_sr=file_sr, target_sr=sr, res_type="soxr_hq") if file_sr != sr else y


def chord_templates():
    labels, temps = [], []
    for root in range(12):
        for quality, third in (("", 4), ("m", 3)):
            t = np.zeros(12)
            t[[root, (root + third) % 12, (root + 7) % 12]] = [1.0, 0.8, 0.8]
            labels.append((root, quality))
            temps.append(t / np.linalg.norm(t))
    return labels, np.array(temps)


def key_name(root, minor):
    flat_major = {5, 10, 3, 8, 1, 6}          # F Bb Eb Ab Db Gb
    flat_minor = {2, 7, 0, 5, 10, 3}          # Dm Gm Cm Fm Bbm Ebm
    names = NOTES_FLAT if (root in (flat_minor if minor else flat_major)) else NOTES_SHARP
    return names[root] + ("m" if minor else "")


def estimate_key(chroma_total):
    best = None
    for root in range(12):
        for minor, prof in ((False, MAJOR_PROFILE), (True, MINOR_PROFILE)):
            r = np.corrcoef(chroma_total, np.roll(prof, root))[0, 1]
            if best is None or r > best[0]:
                best = (r, root, minor)
    return best[1], best[2]


def analyze(tracks):
    """tracks: {part: path to mix/<part>.flac}. Returns a JSON-able dict."""
    import librosa

    y = {part: load_mono(path) for part, path in tracks.items()}
    n = min(len(v) for v in y.values())
    y = {k: v[:n] for k, v in y.items()}
    zero = np.zeros(n, dtype=np.float32)
    inst = sum((v for k, v in y.items() if k not in LEAD_PARTS), zero)
    # harmony: everything but drums, lead vocal and bass (the bass is used separately for roots)
    harm = sum((CHORUS_WEIGHT.get(k, 1.0) * v for k, v in y.items() if k not in DRUM_PARTS | LEAD_PARTS | {"bass"}), zero)
    if not np.any(harm):
        harm = sum(y.values())
    if not np.any(inst):
        inst = sum(y.values())
    bass = y.get("bass", zero)

    harm_mix = sum((v for k, v in y.items() if k not in DRUM_PARTS | LEAD_PARTS), zero)
    if not np.any(harm_mix):
        harm_mix = sum(y.values())
    met = meter.track(sum(y.values()), SR, harm=harm_mix) if meter.available() else None
    if met is not None:
        beat_times, barpos, barlen = met["pulses"], met["pos"], met["length"]
        beat_frames = librosa.time_to_frames(beat_times, sr=SR, hop_length=HOP)
    else:
        beat_times, barpos, barlen = librosa_beats(inst)
        beat_frames = librosa.time_to_frames(beat_times, sr=SR, hop_length=HOP)
    if len(beat_times) < 8:
        return {"version": VERSION, "beats": [], "chords": [], "downbeats": [], "meters": [], "tempo": None, "key": None}

    # ---- chroma, beat-synchronous (each beat = median of frames until the next beat)
    chroma = librosa.feature.chroma_cqt(y=harm, sr=SR, hop_length=HOP, bins_per_octave=36)
    bchroma = librosa.feature.chroma_cqt(y=bass, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("E1"),
                                         n_octaves=4, bins_per_octave=36)
    m = min(chroma.shape[1], bchroma.shape[1])
    chroma, bchroma = chroma[:, :m], bchroma[:, :m]
    bounds = np.clip(np.concatenate([beat_frames, [m]]), 0, m)
    sync = librosa.util.sync(chroma, bounds, aggregate=np.median)[:, :len(beat_frames)]
    bsync = librosa.util.sync(bchroma, bounds, aggregate=np.median)[:, :len(beat_frames)]
    rms = librosa.util.sync(librosa.feature.rms(y=harm + 0.5 * bass, hop_length=HOP)[:, :m], bounds,
                            aggregate=np.mean)[0, :len(beat_frames)]

    bass_on = librosa.onset.onset_strength(y=bass, sr=SR, hop_length=HOP)
    bass_at = bass_on[np.clip(beat_frames, 0, len(bass_on) - 1)]
    bass_at = bass_at / (bass_at.max() + 1e-9)
    c = sync / (np.linalg.norm(sync, axis=0, keepdims=True) + 1e-9)
    novelty = np.r_[0.0, np.linalg.norm(np.diff(c, axis=1), axis=0)]
    novelty = novelty / (novelty.max() + 1e-9)

    def downbeat_of(names):
        """4/4 phase where chords, harmony and bass notes change most."""
        change = np.r_[1.0, (np.array(names[1:]) != np.array(names[:-1])).astype(float)]
        evidence = 1.0 * change + 0.5 * novelty + 0.5 * bass_at
        return int(np.argmax([evidence[p::4].mean() for p in range(4)]))

    if btc.available():
        model = "btc-large-voca"
        lp = 0.5 * (btc.frame_log_probs(sum(y.values())) +
                    btc.frame_log_probs(harm_mix))
        E = beat_emissions(lp, beat_times)
        if barpos is None:
            # the bar phase is the one under which the chord sequence fits the bar-line prior best
            best = None
            for phase in range(4):
                pos = (np.arange(len(beat_times)) - phase) % 4
                path, score = viterbi_beats(E, change_prior(pos, barlen))
                if best is None or score > best[0]:
                    best = (score, pos)
            barpos = best[1]
        path, _ = viterbi_beats(E, change_prior(barpos, barlen))
        names = [btc.label(i) for i in path]
    else:
        model = "chroma-templates"
        labels, temps = chord_templates()
        b = bsync / (bsync.max(axis=0, keepdims=True) + 1e-9)
        score = temps @ c + 0.25 * np.array([b[root] for root, _ in labels])
        quiet = rms < (np.percentile(rms, 95) * 10 ** (-35 / 20))
        score = np.vstack([score, np.where(quiet, 2.0, 0.35)[None, :]])     # "no chord" state
        prob = np.exp(8.0 * (score - score.max(axis=0, keepdims=True)))
        prob /= prob.sum(axis=0, keepdims=True)
        path = librosa.sequence.viterbi(prob, librosa.sequence.transition_loop(len(labels) + 1, 0.85))
        names = ["N" if s == len(labels) else f"{labels[s][0]}:{'min' if labels[s][1] else 'maj'}" for s in path]
        if barpos is None:
            barpos = (np.arange(len(beat_times)) - downbeat_of(names)) % 4

    # ---- key; the profiles cannot tell relative major/minor apart well (F vs Dm),
    # so pick the one whose tonic triad is played more often
    kroot, kminor = estimate_key((sync * rms[None, :]).sum(axis=1))
    rel_root = (kroot + 3) % 12 if kminor else (kroot - 3) % 12
    if tonic_count(names, rel_root, not kminor) > tonic_count(names, kroot, kminor):
        kroot, kminor = rel_root, not kminor

    if met is not None:
        tempo = met["bpm"]
    else:
        # average tempo = slope of a line through the beat times (robust to a few misplaced beats)
        idx = np.arange(len(beat_times))
        slope = np.polyfit(idx, beat_times, 1)[0]
        for _ in range(2):
            resid = beat_times - np.polyval(np.polyfit(idx, beat_times, 1), idx)
            keep = np.abs(resid - np.median(resid)) < 0.25 * slope + np.percentile(np.abs(resid), 50)
            slope = np.polyfit(idx[keep], beat_times[keep], 1)[0]
        tempo = 60.0 / slope
    starts = np.flatnonzero(barpos == 0)
    # time signature at each bar where it changes: [beat index, "6/8"]
    meters, last = [], None
    for i in starts:
        sig = meter.signature(int(barlen[i]), tempo)
        if sig != last:
            meters.append([int(i), sig])
            last = sig
    return {
        "version": VERSION,
        "tempo": round(float(tempo), 1),            # pulses per minute (one pulse = one cell on the page)
        "beats": [round(float(t), 4) for t in beat_times],
        "chords": names,
        "chord_model": model,
        "downbeats": [int(i) for i in starts],      # beat indices that start a bar
        "meters": meters,
        "beat_model": "beat_this" if met is not None else "librosa",
        "key": {"root": int(kroot), "minor": bool(kminor), "name": key_name(kroot, kminor)},
    }


def librosa_beats(inst):
    """Fallback without beat_this: constant-tempo beats; bar positions are decided later (4/4)."""
    import librosa

    # the strongest periodicity of the instrumental picks the tempo (librosa's own prior
    # prefers ~120 BPM and locked onto 2/3 of the real tempo on fast songs), then the tracker follows it
    onset = librosa.onset.onset_strength(y=inst, sr=SR, hop_length=BEAT_HOP, aggregate=np.median)
    tg = librosa.feature.tempogram(onset_envelope=onset, sr=SR, hop_length=BEAT_HOP, win_length=1536)
    bpms = librosa.tempo_frequencies(tg.shape[0], sr=SR, hop_length=BEAT_HOP)
    ac = tg.mean(axis=1)
    ok = (bpms >= 55) & (bpms <= 220)
    start_bpm = float(bpms[ok][np.argmax(ac[ok])])
    # half-time and full-time are often almost equally periodic; prefer the faster reading
    if start_bpm < 100:
        j = int(np.argmin(np.abs(bpms - 2 * start_bpm)))
        near = slice(max(j - 2, 0), j + 3)
        if ac[near].max() >= 0.8 * ac[ok].max():
            start_bpm = float(bpms[near][np.argmax(ac[near])])
    _, bf = librosa.beat.beat_track(onset_envelope=onset, sr=SR, hop_length=BEAT_HOP, trim=False,
                                    bpm=start_bpm, tightness=400)
    beat_times = librosa.frames_to_time(np.asarray(bf, dtype=int), sr=SR, hop_length=BEAT_HOP)
    return beat_times, None, np.full(len(beat_times), 4)


def change_prior(pos, length):
    """Probability that each beat keeps the previous chord, from its place in the bar."""
    stay = np.full(len(pos), 0.99)
    for i, (p, m) in enumerate(zip(pos, length)):
        if p == 0:
            stay[i] = 0.5
        elif p in meter.secondary_positions(int(m)):
            stay[i] = 0.85
    return stay


MINOR_QUALITIES = {"min", "min6", "min7", "minmaj7"}
MAJOR_QUALITIES = {"maj", "maj6", "maj7"}


def tonic_count(names, root, minor):
    want = MINOR_QUALITIES if minor else MAJOR_QUALITIES
    return sum(1 for n in names if ":" in n and int(n.split(":")[0]) == root and n.split(":")[1] in want)


def beat_emissions(lp, beat_times):
    """(beats, labels) evidence per beat: summed frame log-probs over the beat minus its last 8th
    (up to 0.2 s), so a chord pushed ahead of the bar line is credited to the bar it belongs to.
    Summing (not averaging) lets long beats outweigh the change prior, so slow songs keep fast changes."""
    fr = np.arange(len(lp)) * btc.FRAME_S + btc.FRAME_S / 2
    ibi = np.diff(np.r_[beat_times, beat_times[-1] + np.median(np.diff(beat_times))])
    E = np.zeros((len(beat_times), lp.shape[1]))
    for i, b in enumerate(beat_times):
        m = (fr >= b) & (fr < b + ibi[i] - min(0.5 * ibi[i], 0.2))
        if not m.any():
            m = np.abs(fr - b) == np.abs(fr - b).min()
        E[i] = lp[m].sum(axis=0)
    E[:, 168] = -np.inf          # "X" (unknown chord) is not useful on a chord chart
    return E


def viterbi_beats(E, stay):
    """(best label path, its log score); stay[t] = probability that beat t keeps the previous chord."""
    T, S = E.shape
    dp, back = E[0].copy(), np.zeros((T, S), dtype=np.int32)
    for t in range(1, T):
        best = int(np.argmax(dp))
        keep = dp + np.log(stay[t])
        move = dp[best] + np.log((1 - stay[t]) / (S - 1))
        back[t] = np.where(keep >= move, np.arange(S), best)
        dp = np.maximum(keep, move) + E[t]
    path = [int(np.argmax(dp))]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t][path[-1]]))
    return path[::-1], float(dp.max())


def peaks(read_block, frames, bins=PEAK_BINS):
    """Max |x| per bin as 0..255 ints; read_block(start, n) -> (n, ch) float32."""
    edges = np.linspace(0, frames, bins + 1).astype(np.int64)
    out = np.zeros(bins, dtype=np.float32)
    step = 44100 * 20
    pos = 0
    while pos < frames:
        n = min(step, frames - pos)
        a = np.abs(read_block(pos, n)).max(axis=1)
        lo = np.searchsorted(edges, pos, side="right") - 1
        hi = np.searchsorted(edges, pos + n, side="left")
        for i in range(max(lo, 0), min(hi, bins)):
            s, e = max(edges[i], pos) - pos, min(edges[i + 1], pos + n) - pos
            if e > s:
                out[i] = max(out[i], a[s:e].max())
        pos += n
    top = out.max() or 1.0
    return [int(v) for v in np.round(255 * out / top)], float(top)
