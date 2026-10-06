"""Separated tracks -> notes (MIDI) for the falling-notes view.

Models
  basic   Spotify Basic Pitch (ICASSP 2022, Apache-2.0; ONNX, polyphonic, any instrument). Default.
          pip: basic-pitch is installed with --no-deps (its TensorFlow pin has no Python 3.12 wheels;
          the ONNX model needs onnxruntime only).
  piano   ByteDance high-resolution piano transcription (Kong et al. 2021, Apache-2.0;
          pip piano_transcription_inference). Optional for piano parts: on our separated piano
          stems it was not better than Basic Pitch, but it gives exact onsets and velocities.
  mt3     YourMT3+ (Chang et al. 2024; mixer/yourmt3.py, run as a subprocess). Multi-instrument transformer
          trained on strings / winds (URMP, MusicNet) and singing among others. On real solo violin recordings
          (Bach Violin Dataset, 5 movements) note F1 at 50 ms: 0.71 vs Basic Pitch 0.58 (frame F1 0.75 vs 0.64);
          lead vocals look cleaner too. On separated bass and distorted guitar it misses most notes, and on dense
          section strings it drops some held notes — so for strings Basic Pitch's long notes that YourMT3 does
          not have are added back (ENSEMBLE_MIN_LEN).
  auto    per part: AUTO_MT3 kinds -> mt3, everything else -> basic.

Raw output is cleaned per kind of instrument (PROFILES): playable range, minimum length,
fragments of one held note joined, overtone ghosts (octave / twelfth above a louder note)
removed, and for the lead vocal one note at a time with vibrato wobble merged.
Notes are [start s, end s, midi pitch, velocity 1-127] in the time of the original song.
"""

import contextlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline import audio

VERSION = 5
MODELS = ("auto", "basic", "mt3", "piano")
AUTO_MT3 = {"strings", "lead"}          # kinds where YourMT3 did better than Basic Pitch
ENSEMBLE_MIN_LEN = 0.4                  # strings: Basic Pitch notes at least this long that YourMT3 missed are kept
SPECTRUM_MIN_DB = -22                   # YourMT3 notes this far below the strongest pitch at that moment are dropped
SPECTRUM_JOIN_GAP = 0.12                # same-pitch notes this close are one note when the pitch does not dip
SPECTRUM_DIP_DB = 6                     # ... by this much in between
MT3_PROFILE = {"strings": dict(voices=5)}   # a string section: violins I / II, violas, cellos, basses
SILENCE_DB = -45                        # notes where the track is this far below its loudest moment are noise
# kind -> cleanup settings
PROFILES = {
    # keyboards: no joining — a short gap between same-pitch notes is a new key press, not one held note
    # (and no melodia trick: it adds onset-less "continuation" notes; without it every note is a detected press)
    "keys":    dict(lo=21, hi=108, min_len=0.06, join_gap=-1, min_amp=0.18, voices=10, melodia=False),
    "pad":     dict(lo=24, hi=103, min_len=0.12, join_gap=0.10, min_amp=0.22, voices=8),
    "strings": dict(lo=28, hi=100, min_len=0.12, join_gap=0.10, min_amp=0.22, voices=8),
    "bass":    dict(lo=23, hi=67, min_len=0.08, join_gap=0.05, min_amp=0.20, voices=1),
    "guitar":  dict(lo=40, hi=96, min_len=0.06, join_gap=-1, min_amp=0.20, voices=6, melodia=False),
    "lead":    dict(lo=45, hi=88, min_len=0.09, join_gap=0.06, min_amp=0.20, voices=1, vibrato=True),
    "chorus":  dict(lo=45, hi=91, min_len=0.10, join_gap=0.08, min_amp=0.22, voices=3, vibrato=True),
    "other":   dict(lo=21, hi=108, min_len=0.08, join_gap=0.05, min_amp=0.22, voices=8),
}
KIND_OF = {
    "piano": "keys", "digital-piano": "keys", "keys": "keys", "keyboards": "keys", "piano-strings": "keys",
    "upper": "keys", "organ": "pad", "synth": "pad", "strings": "strings", "woodwind": "lead", "brass": "pad",
    "bass": "bass", "guitar": "guitar", "acoustic-guitar": "guitar", "electric-guitar": "guitar",
    "lead": "lead", "vocals": "lead", "chorus": "chorus",
}
# parts worth transcribing (drums, speech, effects... are not)
PITCHED = set(KIND_OF) | {"other", "instrumental", "music", "original"}
UNPITCHED = {"drums", "kick", "snare", "toms", "hihat", "cymbals", "drums-other", "drums-rest", "speech", "effects"}


def is_pitched(part):
    """Parts worth turning into notes, including pieces split off in the mixer ("chorus-v1", "keyboards-rest")."""
    return part not in UNPITCHED and (part in PITCHED or part.startswith(("chorus", "vocals")) or part.endswith("-rest")
                                      or part.endswith("-guitar"))

_bp_model = None
_piano_model = None


def kind_of(part):
    if part in KIND_OF:
        return KIND_OF[part]
    if part.startswith("chorus"):
        return "chorus"
    if part.startswith("vocals"):
        return "lead"
    return KIND_OF.get(part.rsplit("-rest", 1)[0], "other")


def basic_pitch_notes(y, sr, melodia=True, progress=lambda f: None):
    """Raw Basic Pitch note events [(start, end, pitch, amplitude)] for mono audio.
    melodia: Basic Pitch's "melodia trick" (also follows held notes that have no detected onset).
    progress(f) is called with 0..1 while the model runs."""
    global _bp_model
    warnings.filterwarnings("ignore")
    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch.constants import AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE, FFT_HOP
    from basic_pitch.inference import Model, predict

    class CountingModel(Model):
        """The loaded model, reporting progress: predict() runs once per audio window.
        (Basic Pitch only accepts Model instances, hence a subclass that skips loading.)"""

        def __init__(self, inner, total):
            self.inner, self.total, self.n = inner, total, 0

        def predict(self, x):
            out = self.inner.predict(x)
            self.n += 1
            progress(min(1.0, self.n / self.total))
            return out

    if _bp_model is None:
        _bp_model = Model(build_icassp_2022_model_path(FilenameSuffix.onnx))
    # same windowing as basic_pitch.inference.run_inference (30 overlapping frames)
    overlap = 30 * FFT_HOP
    hop = AUDIO_N_SAMPLES - overlap
    windows = max(1, math.ceil((len(y) * AUDIO_SAMPLE_RATE / sr + overlap // 2) / hop))
    with tempfile.TemporaryDirectory(prefix="bp-") as tmp:      # Basic Pitch reads a file
        wav = os.path.join(tmp, "in.wav")
        sf.write(wav, y, sr, subtype="FLOAT")
        _, _, events = predict(wav, CountingModel(_bp_model, windows), onset_threshold=0.5,
                               frame_threshold=0.3, minimum_note_length=58, melodia_trick=melodia)
    return [(float(s), float(e), int(p), float(a)) for s, e, p, a, _ in events]


class _SegmentProgress(io.StringIO):
    """The piano model prints "Segment i / N" while it runs; turn that into progress."""

    def __init__(self, progress):
        super().__init__()
        self.progress = progress

    def write(self, s):
        m = re.search(r"Segment (\d+) / (\d+)", s)
        if m:
            self.progress(int(m.group(1)) / max(1, int(m.group(2))))
        return len(s)


def piano_notes(y, sr, progress=lambda f: None):
    """Raw ByteDance piano model note events [(start, end, pitch, velocity/127)]."""
    global _piano_model
    import librosa
    import torch
    from piano_transcription_inference import PianoTranscription, sample_rate

    if _piano_model is None:
        _piano_model = PianoTranscription(device="cuda" if torch.cuda.is_available() else "cpu", checkpoint_path=None)
    y16 = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
    with tempfile.TemporaryDirectory(prefix="pt-") as tmp, contextlib.redirect_stdout(_SegmentProgress(progress)):
        out = _piano_model.transcribe(y16, os.path.join(tmp, "out.mid"))
    return [(float(e["onset_time"]), float(e["offset_time"]), int(e["midi_note"]), e["velocity"] / 127)
            for e in out["est_note_events"]]


def model_for(model, part):
    """The model actually used for a part ("auto" resolved)."""
    if model != "auto":
        return model
    from . import yourmt3
    return "mt3" if kind_of(part) in AUTO_MT3 and yourmt3.available() else "basic"


def yourmt3_notes(path, progress=lambda f: None):
    """Raw YourMT3 note events [(start, end, pitch, amplitude)], run in its own process (see mixer/yourmt3.py)."""
    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix="mt3-") as tmp:
        out = os.path.join(tmp, "notes.json")
        env = {**os.environ, "PYTHONPATH": str(root)}
        proc = subprocess.Popen([sys.executable, "-m", "mixer.yourmt3", os.fspath(path), out], cwd=root, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        tail = []
        for line in proc.stdout:
            if line.startswith("progress "):
                progress(float(line.split()[1]))
            else:
                tail = (tail + [line.rstrip()])[-20:]
        if proc.wait() != 0:
            raise RuntimeError("YourMT3 failed: " + " / ".join(t for t in tail if t)[-500:])
        return [(s, e, p, 1.0) for s, e, p, _ in json.load(open(out))]


def held_notes_missing(base, extra, min_len):
    """Notes of `extra` at least min_len long that no same-pitch note of `base` covers for half their length."""
    out = []
    for e in extra:
        if e[1] - e[0] < min_len:
            continue
        cov = sum(max(0.0, min(e[1], b[1]) - max(e[0], b[0])) for b in base if b[2] == e[2])
        if cov < 0.5 * (e[1] - e[0]):
            out.append(e)
    return out


def loudness(y, sr, hop=0.05):
    """RMS of the track in hop-second frames, and a function (start, end) -> loudest frame in that span."""
    h = int(sr * hop)
    n = max(1, len(y) // h)
    rms = np.sqrt((np.resize(y, n * h).reshape(n, h) ** 2).mean(axis=1))
    return rms, lambda s, e: rms[int(s / hop):max(int(s / hop) + 1, int(np.ceil(e / hop)))].max(initial=0.0)


def silence_gate(notes, y, sr, db=SILENCE_DB):
    """Drop notes during which the track stays db below its loudest 50 ms (separation leftovers in quiet parts)."""
    rms, peak = loudness(y, sr)
    thr = rms.max() * 10 ** (db / 20)
    return [e for e in notes if peak(e[0], e[1]) >= thr]


class Spectrum:
    """Energy per semitone (dB, CQT max over 3 bins per semitone, MIDI 21-108) of a track, to check notes
    against what is actually sounding."""
    HOP = 512

    def __init__(self, y, sr):
        import librosa
        c = np.abs(librosa.cqt(y, sr=sr, hop_length=self.HOP, fmin=librosa.midi_to_hz(21), n_bins=88 * 3,
                               bins_per_octave=36))
        self.db = 20 * np.log10(c.reshape(88, 3, -1).max(axis=1) + 1e-9)
        self.frame_max = self.db.max(axis=0)
        self.top = float(np.percentile(self.frame_max, 99))
        self.sr = sr

    def frames(self, s, e):
        a = int(s * self.sr / self.HOP)
        return a, min(self.db.shape[1], max(a + 1, int(e * self.sr / self.HOP)))

    def note(self, s, e, p):
        """(dB below the strongest semitone at that moment, dB below the track's loud level) over the first 0.5 s."""
        a, b = self.frames(s, min(e, s + 0.5))
        if not 21 <= p <= 108 or a >= b:
            return -99.0, -99.0
        row = self.db[int(p) - 21, a:b]
        return float(np.median(row - self.frame_max[a:b])), float(np.median(row) - self.top)


def check_with_spectrum(events, y, sr, min_db=SPECTRUM_MIN_DB, join_gap=SPECTRUM_JOIN_GAP, dip_db=SPECTRUM_DIP_DB):
    """For a model that hears notes the audio does not have (YourMT3 on dense separated stems):
    - drop notes whose pitch is min_db or more below the strongest pitch at that moment (nothing there),
    - join same-pitch notes less than join_gap apart when the pitch keeps sounding in between (no dip of
      dip_db): one held / tremolo note that was cut into a stutter of short notes,
    - amplitude = the pitch's own energy (0..1), so the voice limit keeps the strongest notes."""
    sp = Spectrum(y, sr)
    kept = []
    for s, e, p, _ in events:
        rel, level = sp.note(s, e, p)
        if rel >= min_db:
            kept.append([s, e, p, level])
    kept.sort(key=lambda n: (n[2], n[0]))
    out = []
    for n in kept:
        m = out[-1] if out else None
        if m and m[2] == n[2] and 0 <= n[0] - m[1] <= join_gap:
            a, b = sp.frames(m[1], n[0])
            gap = sp.db[int(n[2]) - 21, a:b + 1].min() - sp.top if b > a else min(m[3], n[3])
            if gap >= min(m[3], n[3]) - dip_db:
                m[1] = max(m[1], n[1])
                m[3] = max(m[3], n[3])
                continue
        out.append(n)
    return [(s, e, p, float(10 ** (max(level, -60.0) / 40))) for s, e, p, level in out]   # -60..0 dB -> 0.03..1


def clean(events, kind, **override):
    """Per-instrument cleanup of raw events -> [[start, end, pitch, velocity]] sorted by start.
    override: profile settings to change (YourMT3 output is already clean, so it gets a lighter pass)."""
    p = {**PROFILES[kind], **override}
    ev = sorted([list(e) for e in events if p["lo"] <= e[2] <= p["hi"]], key=lambda e: (e[2], e[0]))
    # 1. join fragments of one held note (same pitch, tiny gap)
    joined = []
    for e in ev:
        if joined and joined[-1][2] == e[2] and e[0] - joined[-1][1] <= p["join_gap"]:
            j = joined[-1]
            w = (j[1] - j[0], e[1] - e[0])
            j[3] = (j[3] * w[0] + e[3] * w[1]) / max(sum(w), 1e-9)
            j[1] = max(j[1], e[1])
        else:
            joined.append(e)
    # 2. vibrato: short notes a semitone apart that follow each other are one sung note
    if p.get("vibrato"):
        joined.sort(key=lambda e: e[0])
        merged = []
        for e in joined:
            m = merged[-1] if merged else None
            if m and abs(e[2] - m[2]) <= 1 and e[0] - m[1] <= 0.04 and min(e[1] - e[0], m[1] - m[0]) < 0.25:
                if e[1] - e[0] > m[1] - m[0]:
                    m[2] = e[2]
                m[1] = max(m[1], e[1])
                m[3] = max(m[3], e[3])
            else:
                merged.append(e)
        joined = merged
    # 3. too short / too weak
    keep = [e for e in joined if e[1] - e[0] >= p["min_len"] and e[3] >= p["min_amp"]]
    # 4. overtone ghosts: a weaker note an octave / twelfth / two octaves above a sounding note
    keep.sort(key=lambda e: e[0])
    starts = np.array([e[0] for e in keep])
    out = []
    for e in keep:
        ghost = False
        for q in keep[max(0, np.searchsorted(starts, e[0] - 2.0)):np.searchsorted(starts, e[1])]:
            if e[2] - q[2] in (12, 19, 24) and q[3] > 1.4 * e[3]:
                overlap = min(e[1], q[1]) - max(e[0], q[0])
                if overlap > 0.7 * (e[1] - e[0]):
                    ghost = True
                    break
        if not ghost:
            out.append(e)
    # 5. at most `voices` notes at once (keep the louder ones); lead vocal = 1
    if p["voices"] < 10:
        out.sort(key=lambda e: -e[3])
        chosen = []
        for e in out:
            busy = sum(1 for c in chosen if c[0] < e[1] - 0.03 and e[0] < c[1] - 0.03)
            if busy < p["voices"]:
                chosen.append(e)
        out = chosen
    out.sort(key=lambda e: (e[0], e[2]))
    vmax = max((e[3] for e in out), default=1.0) or 1.0
    return [[round(e[0], 4), round(e[1], 4), int(e[2]), int(np.clip(round(30 + 97 * e[3] / vmax), 1, 127))] for e in out]


def transcribe(path, part, model="basic", progress=lambda f: None):
    """Notes for one track file. model: "basic", "piano" or "mt3" ("auto": see model_for).
    progress(f): 0..1 over the whole job (decoding 0-5%, the model(s) 5-90%, cleaning 90-100%).
    Returns (notes, number of raw events, model used)."""
    model = model_for(model, part)
    progress(0.0)
    y = audio.decode(path, sample_rate=44100, channels=1)[:, 0]
    progress(0.05)
    kind = kind_of(part)
    melodia = PROFILES[kind].get("melodia", True)
    if model == "mt3":
        both = kind == "strings"
        step = lambda f: progress(0.05 + (0.65 if both else 0.85) * f)
        raw = yourmt3_notes(path, step)
        events = raw
        if both:
            bp = basic_pitch_notes(y, 44100, melodia, lambda f: progress(0.7 + 0.2 * f))
            events = raw + held_notes_missing(raw, clean(bp, kind), ENSEMBLE_MIN_LEN)
            raw = raw + bp
        # notes the spectrum does not back up go, stutters of one held note are joined; then the usual cleanup
        # without its own joining / length limits, keeping the voice limit (and vibrato merge for singing)
        notes = clean(check_with_spectrum(events, y, 44100), kind, min_len=0.03, join_gap=-1, min_amp=0,
                      **MT3_PROFILE.get(kind, {}))
    else:
        step = lambda f: progress(0.05 + 0.85 * f)
        raw = piano_notes(y, 44100, step) if model == "piano" else basic_pitch_notes(y, 44100, melodia, step)
        notes = clean(raw, kind)
    progress(0.95)
    notes = silence_gate(notes, y, 44100)
    progress(1.0)
    return notes, len(raw), model
