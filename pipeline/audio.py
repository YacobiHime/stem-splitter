"""Audio I/O helpers. All files are 44.1kHz stereo float32 WAV."""

import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def convert_to_wav(src, dst, sample_rate):
    """Decode any ffmpeg-readable file to stereo float32 WAV at sample_rate."""
    dst = Path(dst)
    tmp = dst.with_name(dst.stem + ".tmp.wav")
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", os.fspath(src),
        "-vn", "-map", "0:a:0",
        "-af", "aresample=resampler=soxr:precision=28",
        "-ar", str(sample_rate), "-ac", "2",
        "-c:a", "pcm_f32le", os.fspath(tmp),
    ]
    subprocess.run(cmd, check=True)
    os.replace(tmp, dst)


def read(path):
    """Return (frames, channels) float32 array and sample rate."""
    audio, sr = sf.read(os.fspath(path), dtype="float32", always_2d=True)
    return audio, sr


def write(path, audio, sample_rate):
    """Atomically write (frames, channels) float32 WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.wav")
    sf.write(os.fspath(tmp), np.asarray(audio, dtype=np.float32), sample_rate, subtype="FLOAT", format="WAV")
    os.replace(tmp, path)


def write_flac(path, audio, sample_rate):
    """Atomically write (frames, channels) float32 as 16bit FLAC (clipped to [-1, 1])."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.flac")
    sf.write(os.fspath(tmp), np.clip(audio, -1.0, 1.0), sample_rate, subtype="PCM_16", format="FLAC")
    os.replace(tmp, path)


def write_m4a(path, audio, sample_rate, codec="aac", bitrate="256k", metadata=None):
    """Atomically encode (frames, channels) float32 to .m4a (AAC or ALAC) via ffmpeg."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.m4a")
    a = np.ascontiguousarray(audio, dtype=np.float32)
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "f32le", "-ar", str(sample_rate), "-ac", str(a.shape[1]), "-i", "pipe:0",
        "-c:a", codec,
    ]
    if codec == "aac":
        cmd += ["-b:a", bitrate]
    elif codec == "alac":
        cmd += ["-sample_fmt", "s32p"]  # ALAC has no float; ffmpeg stores this as 24bit lossless
    for k, v in (metadata or {}).items():
        cmd += ["-metadata", f"{k}={v}"]
    cmd += ["-movflags", "+faststart", "-f", "mp4", os.fspath(tmp)]
    subprocess.run(cmd, input=a.tobytes(), check=True)
    os.replace(tmp, path)


def decode(path, sample_rate=44100, channels=2):
    """Decode any file with ffmpeg to (frames, channels) float32."""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", os.fspath(path),
           "-f", "f32le", "-ar", str(sample_rate), "-ac", str(channels), "pipe:1"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, channels)


def fit_length(audio, frames, channels=2):
    """Trim or zero-pad to exactly `frames`, and force `channels` channels."""
    if audio.shape[1] != channels:
        audio = np.repeat(audio[:, :1], channels, axis=1) if audio.shape[1] == 1 else audio[:, :channels]
    if audio.shape[0] > frames:
        return audio[:frames]
    if audio.shape[0] < frames:
        pad = np.zeros((frames - audio.shape[0], channels), dtype=audio.dtype)
        return np.concatenate([audio, pad])
    return audio


def loudest_window_dbfs(audio, sample_rate, window_s=1.0):
    """RMS (dBFS) of the loudest window — robust to parts that only play briefly."""
    n = int(sample_rate * window_s)
    usable = audio.shape[0] // n * n
    if usable == 0:
        w = np.sqrt(np.mean(np.square(audio, dtype=np.float64))) if audio.size else 0.0
    else:
        x = audio[:usable].astype(np.float64).reshape(-1, n, audio.shape[1])
        w = float(np.sqrt(np.mean(x ** 2, axis=(1, 2))).max())
    return 20 * np.log10(w) if w > 0 else -np.inf


def stats(audio):
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0
    db = lambda v: round(20 * np.log10(v), 2) if v > 0 else None
    return {"peak": round(peak, 6), "peak_dbfs": db(peak), "rms_dbfs": db(rms)}
