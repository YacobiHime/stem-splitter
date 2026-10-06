"""Separated tracks -> notes (MIDI) for the falling-notes view.

Models
  basic   Spotify Basic Pitch (ICASSP 2022, Apache-2.0; ONNX, polyphonic, any instrument). Default.
          pip: basic-pitch is installed with --no-deps (its TensorFlow pin has no Python 3.12 wheels;
          the ONNX model needs onnxruntime only).
  piano   ByteDance high-resolution piano transcription (Kong et al. 2021, Apache-2.0;
          pip piano_transcription_inference). Optional for piano parts: on our separated piano
          stems it was not better than Basic Pitch, but it gives exact onsets and velocities.

Raw output is cleaned per kind of instrument (PROFILES): playable range, minimum length,
fragments of one held note joined, overtone ghosts (octave / twelfth above a louder note)
removed, and for the lead vocal one note at a time with vibrato wobble merged.
Notes are [start s, end s, midi pitch, velocity 1-127] in the time of the original song.
"""

import contextlib
import io
import math
import os
import re
import tempfile
import warnings

import numpy as np
import soundfile as sf

from pipeline import audio

VERSION = 3
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


def clean(events, kind):
    """Per-instrument cleanup of raw events -> [[start, end, pitch, velocity]] sorted by start."""
    p = PROFILES[kind]
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
    """Notes for one track file. model: "basic" or "piano". progress(f): 0..1 over the whole job
    (decoding 0-5%, the model 5-85%, turning its output into notes and cleaning 85-100%)."""
    progress(0.0)
    y = audio.decode(path, sample_rate=44100, channels=1)[:, 0]
    progress(0.05)
    kind = kind_of(part)
    step = lambda f: progress(0.05 + 0.8 * f)
    raw = piano_notes(y, 44100, step) if model == "piano" else basic_pitch_notes(y, 44100, PROFILES[kind].get("melodia", True), step)
    progress(0.9)
    notes = clean(raw, kind)
    progress(1.0)
    return notes, len(raw)
