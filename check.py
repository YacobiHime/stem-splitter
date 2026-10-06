#!/usr/bin/env python3
"""Validate the final files in output/<song>/ (listed in manifest.json, else every top-level .wav/.m4a).

Checks: same length / sample rate / channels across all files (m4a is decoded
with ffmpeg, so encoder padding/priming would show up as a length mismatch),
expected codec, no silent file, no clipping (|x| > 1.0 after decoding).
Mixer tracks (mix/, lossless FLAC) must have exactly manifest format.frames frames
(AAC files are longer by the encoder padding), and their sum must match
work/original.wav (times the manifest global_gain) when work/ still exists.

Usage: .venv/bin/python check.py [output/<song> ...]   (default: every song in output/)
Exit code 0 = all songs OK (warnings allowed), 1 = errors found.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
SILENCE_DBFS = -60.0   # peak below this -> silent
MIX_RESIDUAL_DB = -60.0  # sum of mix/ tracks (16bit FLAC) minus original, relative to the original
EXPECTED_SR = 44100
EXPECTED_CH = 2


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,sample_rate,channels,bit_rate", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    return s["codec_name"], int(s["sample_rate"]), int(s["channels"]), s.get("bit_rate")


def load(path):
    """Return (audio(frames, ch), sample_rate, channels, codec label)."""
    if path.suffix == ".wav":
        info = sf.info(str(path))
        a, _ = sf.read(str(path), dtype="float32", always_2d=True)
        return a, info.samplerate, info.channels, info.subtype
    codec, sr, ch, br = probe(path)
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "f32le", "pipe:1"],
                         check=True, capture_output=True).stdout
    a = np.frombuffer(raw, dtype=np.float32).reshape(-1, ch)
    label = f"{codec}{'/' + str(round(int(br) / 1000)) + 'k' if br and codec == 'aac' else ''}"
    return a, sr, ch, label


def check_song(d):
    mf = d / "manifest.json"
    mix, m = [], {}
    if mf.exists():
        m = json.loads(mf.read_text())
        files = [d / name for name in m.get("files", {})]
        mix = [d / name for name in m.get("mix_files", {})]
        files += mix
    else:
        files = sorted(list(d.glob("*.wav")) + list(d.glob("*.m4a")))
    errors, warnings, rows = [], [], []
    missing = [f.name for f in files if not f.exists()]
    if missing:
        return [f"listed in manifest but missing: {missing}"], [], []
    if not files:
        return [f"no .wav / .m4a files in {d}"], [], []
    exts = {f.suffix for f in files if f not in mix}
    if len(exts) > 1:
        errors.append(f"mixed formats {sorted(exts)} (stale files from an older run?)")
    ref = None
    mix_frames = (m.get("format") or {}).get("frames")
    mix_sum = None
    for f in files:
        a, sr, ch, codec = load(f)
        frames = a.shape[0]
        peak = float(np.max(np.abs(a))) if a.size else 0.0
        peak_db = 20 * np.log10(peak) if peak > 0 else -np.inf
        rms = float(np.sqrt(np.mean(np.square(a, dtype=np.float64)))) if a.size else 0.0
        rms_db = 20 * np.log10(rms) if rms > 0 else -np.inf
        sig = (frames, sr, ch)
        status = []
        if f in mix:
            if mix_frames is not None and frames != mix_frames:
                status.append(f"frames={frames} != {mix_frames} (manifest)")
        else:
            ref = ref or (sig, f.name)
        if sr != EXPECTED_SR:
            status.append(f"sr={sr}")
        if ch != EXPECTED_CH:
            status.append(f"ch={ch}")
        if f.suffix == ".wav" and codec != "FLOAT":
            status.append(f"subtype={codec}")
        if f not in mix and sig != ref[0]:
            status.append(f"frames={frames} != {ref[0][0]} ({ref[1]})")
        if peak > 1.0:
            status.append(f"CLIP peak={peak:.4f}")
        name = f.relative_to(d).as_posix()
        for s in status:
            errors.append(f"{name}: {s}")
        if peak_db < SILENCE_DBFS:
            warnings.append(f"{name}: silent (peak {peak_db:.1f} dBFS)")
        if f in mix and not status:
            mix_sum = a.astype(np.float64) if mix_sum is None else mix_sum + a
        rows.append((name, frames, sr, ch, codec, peak_db, rms_db,
                     "ERR" if status else ("SILENT" if peak_db < SILENCE_DBFS else "ok")))
    work_orig = d / "work" / "original.wav"
    if mix_sum is not None and work_orig.exists():
        orig = sf.read(str(work_orig), dtype="float64", always_2d=True)[0] * m.get("global_gain", 1.0)
        n = min(len(orig), len(mix_sum))
        rms = lambda x: np.sqrt(np.mean(np.square(x)))
        rel = 20 * np.log10(max(rms(mix_sum[:n] - orig[:n]), 1e-12) / max(rms(orig[:n]), 1e-12))
        line = f"mix/ sum vs work/original.wav: residual {rel:.1f} dB"
        if rel > MIX_RESIDUAL_DB:
            errors.append(f"{line} (> {MIX_RESIDUAL_DB} dB: mix tracks overlap or are missing a part)")
        else:
            rows.append((line, None, None, None, None, None, None, "ok"))
    return errors, warnings, rows


def main(argv):
    dirs = [Path(p) for p in argv] or sorted(p for p in (ROOT / "output").iterdir() if p.is_dir())
    bad = False
    for d in dirs:
        errors, warnings, rows = check_song(d)
        print(f"\n== {d.name}")
        if rows:
            print(f"  {'file':<34}{'frames':>10}{'sr':>7}{'ch':>4}  {'codec':<9}{'peak dB':>9}{'rms dB':>9}  status")
            for r in rows:
                if r[1] is None:
                    print(f"  {r[0]:<70}  {r[7]}")
                    continue
                print(f"  {r[0]:<34}{r[1]:>10}{r[2]:>7}{r[3]:>4}  {r[4]:<9}{r[5]:>9.1f}{r[6]:>9.1f}  {r[7]}")
        for w in warnings:
            print(f"  WARN  {w}")
        for e in errors:
            print(f"  ERROR {e}")
        print("  RESULT:", "NG" if errors else ("OK (with warnings)" if warnings else "OK"))
        bad |= bool(errors)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
