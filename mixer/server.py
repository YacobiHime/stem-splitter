"""LAN-only web mixer for output/<song>/: play the stems together with a fader per part.

The browser streams 16bit PCM in chunks (GET /api/pcm) and mixes them with Web Audio,
so only a few seconds of each track are in memory — this keeps phones happy.
Tracks:
  mix   = output/<song>/mix/*.flac (do not overlap; all at 0 dB = the original)
  extra = the other final files (original, piano, strings, ...) to add on top, off by default

Tempo / key changes are rendered here with ffmpeg's rubberband filter, per track, into
output/<song>/.mixer_cache/var/<t..._p...>/ (the newest VARIANTS_KEPT per song are kept).
Beats / chords / key (mixer/analysis.py) and waveform peaks are cached in .mixer_cache/ too,
and computed in the background at startup.

Notes for the falling-notes view: POST /api/transcribe queues tracks, a worker thread runs
mixer/transcribe.py, results are cached in output/<song>/notes/<file>.json.

Songs can be added from the page (PUT /api/upload -> inbox/ -> mixer/jobs.py runs the pipeline).
Deleting a song moves output/<song> to output/.trash/ (nothing is removed for good).

Access needs a token (a cookie set by opening the URL printed at startup once),
so other people on the same network cannot listen in.
"""

import argparse
import hmac
import json
import mimetypes
import os
import secrets
import socket
import subprocess
import shutil
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

import numpy as np
import soundfile as sf

from pipeline import audio

from . import analysis
from .jobs import Jobs
from . import transcribe

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
TOKEN_FILE = ROOT / ".mixer_token"
COOKIE = "mixer_token"
CHANNELS = 2
MAX_REQUEST_FRAMES = 44100 * 30
MAX_GAIN = 2.0                 # +6 dB, the top of the faders
EXPORT_CEILING = 10 ** (-1 / 20)   # AAC overshoots slightly, so keep 1 dB of headroom (same as the pipeline)
MAX_UPLOAD = 1 << 30          # 1 GiB
CACHE_DIR = ".mixer_cache"   # output/<song>/.mixer_cache/<file>.s16 — decoded m4a, rebuilt when older than the source
VARIANTS_KEPT = 3             # tempo/key renderings kept per song (each is ~60 MB per track)
TEMPO_RANGE = (0.5, 1.5)
SEMI_RANGE = (-12, 12)
VOCAL_PARTS = {"lead", "chorus"}   # keep their formants when shifting the key (no chipmunk voice)
RENDER_POOL = ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 4) // 2)))


def variant_key(tempo, semi):
    """Normalized (tempo, semi, key) — key is None for the unmodified audio."""
    tempo = round(min(max(float(tempo), TEMPO_RANGE[0]), TEMPO_RANGE[1]), 2)
    semi = int(min(max(int(semi), SEMI_RANGE[0]), SEMI_RANGE[1]))
    key = None if (tempo == 1.0 and semi == 0) else f"t{tempo:.2f}_p{semi:+d}"
    return tempo, semi, key


def load_token():
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    token = secrets.token_urlsafe(16)
    TOKEN_FILE.write_text(token + "\n")
    TOKEN_FILE.chmod(0o600)
    return token


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))   # no packet is sent; this only picks the outgoing interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class Library:
    def __init__(self, out_root):
        self.out_root = Path(out_root).resolve()
        self.locks = {}
        self.locks_guard = threading.Lock()
        self.summaries = {}           # analysis.json path -> (mtime, summary)
        self.tr_queue, self.tr_status, self.tr_cond = [], {}, threading.Condition()
        threading.Thread(target=self.transcribe_worker, daemon=True).start()

    def manifest(self, song):
        d = self.out_root / song
        if "/" in song or song.startswith(".") or not (d / "manifest.json").is_file():
            return None, None
        return d, json.loads((d / "manifest.json").read_text())

    def songs(self):
        out = []
        for d in sorted(p for p in self.out_root.iterdir() if p.is_dir()):
            _, m = self.manifest(d.name)
            if not m or not m.get("mix_files"):
                continue
            out.append(self.describe(d, m))
        return out

    def cache_dir(self, d):
        c = d / CACHE_DIR
        c.mkdir(exist_ok=True)
        return c

    def meta(self, d, m):
        """Title / artist / cover from the source file's tags (cached)."""
        src = Path(m.get("input", ""))
        cache = self.cache_dir(d) / "meta.json"
        stamp = src.stat().st_mtime if src.exists() else 0
        if cache.exists():
            c = json.loads(cache.read_text())
            if c.get("stamp") == stamp:
                return c
        c = {"stamp": stamp, "title": m["song"], "artist": "", "cover": False}
        if src.exists():
            with self._lock(os.fspath(cache)):
                try:
                    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=title,artist",
                                          "-of", "json", os.fspath(src)], capture_output=True, text=True, check=True).stdout
                    tags = {k.lower(): v for k, v in (json.loads(out).get("format", {}).get("tags") or {}).items()}
                    c["title"] = tags.get("title") or m["song"]
                    c["artist"] = tags.get("artist", "")
                    cover = self.cache_dir(d) / "cover.jpg"
                    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", os.fspath(src), "-an",
                                        "-map", "0:v:0", "-frames:v", "1", "-vf", "scale=360:360:force_original_aspect_ratio=increase,crop=360:360",
                                        os.fspath(cover)], capture_output=True)
                    c["cover"] = r.returncode == 0 and cover.exists()
                except (subprocess.CalledProcessError, ValueError):
                    pass
                cache.write_text(json.dumps(c, ensure_ascii=False))
        return c

    def describe(self, d, m):
        def entry(rel, info, group):
            f = d / rel
            return {"id": rel, "part": info.get("part") or rel, "group": group, "parent": info.get("parent"),
                    "v": int(f.stat().st_mtime), "peak_dbfs": info.get("peak_dbfs")}

        mix = [entry(rel, info, "mix") for rel, info in m["mix_files"].items() if (d / rel).exists()]
        mix_parts = {t["part"] for t in mix}
        # an extra with the same part name as a mix track (lead, keyboards, ...) would only duplicate it
        extra = [entry(rel, info, "extra") for rel, info in m.get("files", {}).items()
                 if (d / rel).exists() and info.get("part") not in mix_parts]
        meta = self.meta(d, m)
        return {"name": m["song"], "dir": d.name, "frames": m["format"]["frames"],
                "sample_rate": m["format"]["sample_rate"], "tracks": mix + extra,
                "title": meta["title"], "artist": meta["artist"], "cover": meta["cover"],
                "added": m.get("generated"), "profile": m.get("profile"),
                "absent": list((m.get("detection") or {}).get("absent") or {}), **self.summary(d)}

    def summary(self, d):
        """BPM / key / main meter for the song list, from the cached analysis (if it is there yet)."""
        f = d / CACHE_DIR / "analysis.json"
        if not f.exists():
            return {"bpm": None, "key": None, "meter": None}
        st = f.stat().st_mtime
        hit = self.summaries.get(f)
        if hit and hit[0] == st:
            return hit[1]
        a = json.loads(f.read_text())
        bpm, key, sig = a.get("tempo"), (a.get("key") or {}), None
        starts = a.get("downbeats") or []
        if len(starts) > 1 and bpm:
            lens = np.diff(starts)
            main = int(np.bincount(lens).argmax())
            from .meter import signature
            sig = signature(main, bpm)
            if sig.endswith("/8") and main in (6, 9, 12):
                bpm = bpm / 3                     # dotted quarters, as musicians count 6/8
        names = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
        out = {"bpm": round(bpm) if bpm else None, "meter": sig,
               "key": f"{key.get('name', '')[:-1] if key.get('minor') else key.get('name', '')} {'minor' if key.get('minor') else 'major'}"
               if key else None}
        self.summaries[f] = (st, out)
        return out

    # ---- notes (MIDI transcription)

    def notes_path(self, d, track):
        return d / "notes" / (track.replace("/", "__") + ".json")

    def notes(self, song):
        """{track id: {status, model, notes?}} for every pitched track of the song."""
        d, m = self.manifest(song)
        if not m:
            raise LookupError("no such song")
        out = {}
        mix_parts = {i.get("part") for i in m["mix_files"].values()}
        for rel, info in {**m["mix_files"], **m.get("files", {})}.items():
            group = "mix" if rel in m["mix_files"] else "extra"
            if not transcribe.is_pitched(info.get("part", "")) or not (d / rel).exists():
                continue
            if group == "extra" and info.get("part") in mix_parts:      # same as the mixer: no duplicates
                continue
            f = self.notes_path(d, rel)
            st = self.tr_status.get((d.name, rel))
            entry = {"part": info.get("part"), "group": group, "status": "none"}
            if f.exists():
                c = json.loads(f.read_text())
                if c.get("stamp") == (d / rel).stat().st_mtime and c.get("version") == transcribe.VERSION:
                    entry.update(status="done", model=c["model"], notes=c["notes"])
                elif not st:
                    # made with an older version (or the track was separated again): redo it with the same model
                    self.queue_transcription(d.name, [rel], c.get("requested") or c.get("model", "basic"))
                    st = self.tr_status.get((d.name, rel))
            if st and (st["status"] in ("queued", "running") or (st["status"] == "error" and entry["status"] != "done")):
                entry.update(status=st["status"], message=st.get("message", ""), progress=st.get("progress", 0.0))
            out[rel] = entry
        return out

    def queue_transcription(self, song, tracks, model):
        d, m = self.manifest(song)
        if not m:
            raise LookupError("no such song")
        if model not in transcribe.MODELS:
            raise ValueError("model must be one of " + ", ".join(transcribe.MODELS))
        known = {**m["mix_files"], **m.get("files", {})}
        for t in tracks:
            if t not in known:
                raise LookupError(f"no such track: {t}")
        with self.tr_cond:
            for t in tracks:
                self.tr_status[(d.name, t)] = {"status": "queued", "model": model}
                self.tr_queue.append((d.name, t, model))
            self.tr_cond.notify_all()

    def transcribe_worker(self):
        while True:
            with self.tr_cond:
                while not self.tr_queue:
                    self.tr_cond.wait()
                song, track, model = self.tr_queue.pop(0)
                self.tr_status[(song, track)] = {"status": "running", "model": model, "progress": 0.0}
            try:
                d, m = self.manifest(song)
                part = ({**m["mix_files"], **m.get("files", {})}.get(track) or {}).get("part", "other")
                src = d / track
                t = time.time()
                st = self.tr_status[(song, track)]
                requested = model
                notes, raw, model = transcribe.transcribe(src, part, model, lambda f, st=st: st.__setitem__("progress", round(f, 3)))
                f = self.notes_path(d, track)
                f.parent.mkdir(exist_ok=True)
                f.write_text(json.dumps({"version": transcribe.VERSION, "stamp": src.stat().st_mtime, "model": model,
                                         "requested": requested, "part": part, "raw_notes": raw, "notes": notes}))
                self.tr_status.pop((song, track), None)
                print(f"[{song}] notes {track}: {len(notes)} notes (raw {raw}) in {time.time() - t:.1f}s [{model}]", flush=True)
            except Exception as e:  # noqa: BLE001 — report on the page, keep the worker alive
                traceback.print_exc()
                self.tr_status[(song, track)] = {"status": "error", "message": f"{type(e).__name__}: {e}"}

    def delete(self, song):
        """Move output/<song> to output/.trash/<song>-<time> (recoverable)."""
        d, m = self.manifest(song)
        if not m:
            raise LookupError("no such song")
        trash = self.out_root / ".trash"
        trash.mkdir(exist_ok=True)
        dst = trash / f"{d.name}-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.move(os.fspath(d), os.fspath(dst))
        return dst

    # ---- analysis and waveform peaks (cached; invalidated when a source file changes)

    def details(self, song):
        """{analysis, peaks} for the page; computed on first use (a few seconds)."""
        d, m = self.manifest(song)
        if not m:
            raise LookupError("no such song")
        with self._lock(f"details:{d}"):
            return {"analysis": self._analysis(d, m), "peaks": self._peaks(d, m)}

    def _analysis(self, d, m):
        mix = {info["part"]: d / rel for rel, info in m["mix_files"].items() if (d / rel).exists()}
        stamp = {p: f.stat().st_mtime for p, f in mix.items()}
        cache = self.cache_dir(d) / "analysis.json"
        if cache.exists():
            c = json.loads(cache.read_text())
            if c.get("stamp") == stamp and c.get("version") == analysis.VERSION:
                return c
        t = time.time()
        c = analysis.analyze(mix)
        c["stamp"] = stamp
        cache.write_text(json.dumps(c))
        print(f"[{d.name}] analysis: {time.time() - t:.1f}s tempo {c.get('tempo')} key {(c.get('key') or {}).get('name')}", flush=True)
        return c

    def _peaks(self, d, m):
        total = m["format"]["frames"]
        tracks = {**m.get("mix_files", {}), **m.get("files", {})}
        cache = self.cache_dir(d) / "peaks.json"
        c = json.loads(cache.read_text()) if cache.exists() else {}
        changed = False
        for rel in tracks:
            f = d / rel
            if not f.exists():
                continue
            if c.get(rel, {}).get("stamp") != f.stat().st_mtime:
                if f.suffix in (".flac", ".wav"):
                    read = lambda s, n, f=f: self.read_float(f, s, n)
                else:   # decode in memory: no 60 MB .s16 cache for parts nobody plays
                    whole = audio.fit_length(audio.decode(f), total, CHANNELS)
                    read = lambda s, n, a=whole: a[s:s + n]
                p, top = analysis.peaks(read, total)
                c[rel] = {"stamp": f.stat().st_mtime, "p": p, "top": top}
                changed = True
        if changed:
            cache.write_text(json.dumps(c))
        return {rel: {"p": v["p"], "top": v["top"]} for rel, v in c.items() if rel in tracks}

    def prewarm(self):
        """Analyse every song in the background so the first page load is quick."""
        for song in self.songs():
            try:
                self.details(song["dir"])
            except Exception:  # noqa: BLE001 — a broken song must not stop the others
                traceback.print_exc()

    # ---- tempo / key variants

    def variant(self, d, m, track, tempo, semi, key):
        """Raw s16le of `track` rendered at tempo/semi (rubberband), cached under var/<key>/."""
        src = d / track
        vdir = self.cache_dir(d) / "var" / key
        out = vdir / (track.replace("/", "__") + ".s16")
        with self._lock(os.fspath(out)):
            if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
                return out
            vdir.mkdir(parents=True, exist_ok=True)
            part = ({**m["mix_files"], **m.get("files", {})}.get(track) or {}).get("part")
            af = f"rubberband=tempo={tempo}:pitch={2 ** (semi / 12):.6f}:pitchq=quality"
            if semi and part in VOCAL_PARTS:
                af += ":formant=preserved"
            tmp = out.with_suffix(".tmp")
            t = time.time()
            subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", os.fspath(src),
                            "-af", af, "-f", "s16le", "-ac", str(CHANNELS), "-ar", "44100", os.fspath(tmp)], check=True)
            os.replace(tmp, out)
            print(f"[{d.name}] render {key} {track}: {time.time() - t:.1f}s", flush=True)
        self.prune_variants(d, keep=key)
        return out

    def prune_variants(self, d, keep):
        root = d / CACHE_DIR / "var"
        if not root.exists():
            return
        (root / keep).touch(exist_ok=True)
        dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in dirs[VARIANTS_KEPT:]:
            if old.name != keep:
                shutil.rmtree(old, ignore_errors=True)

    def prepare(self, song, tracks, tempo, semi):
        """Render the given tracks for tempo/semi in parallel; return the variant length in frames."""
        d, m = self.manifest(song)
        if not m:
            raise LookupError("no such song")
        tempo, semi, key = variant_key(tempo, semi)
        known = {**m["mix_files"], **m.get("files", {})}
        bad = [t for t in tracks if t not in known]
        if bad:
            raise LookupError(f"no such track: {bad[0]}")
        if key:
            list(RENDER_POOL.map(lambda t: self.variant(d, m, t, tempo, semi, key), tracks))
        return {"tempo": tempo, "semi": semi, "frames": self.variant_frames(m, tempo)}

    @staticmethod
    def variant_frames(m, tempo):
        return int(round(m["format"]["frames"] / tempo))

    def track_path(self, song, track):
        d, m = self.manifest(song)
        if not m or (track not in m.get("mix_files", {}) and track not in m.get("files", {})):
            return None, None
        return d / track, m

    def _lock(self, key):
        with self.locks_guard:
            return self.locks.setdefault(key, threading.Lock())

    def decoded(self, src):
        """Raw s16le stereo cache of an m4a (soundfile cannot read AAC)."""
        cache = src.parent / CACHE_DIR / (src.name + ".s16")
        with self._lock(os.fspath(cache)):
            if not cache.exists() or cache.stat().st_mtime < src.stat().st_mtime:
                cache.parent.mkdir(exist_ok=True)
                tmp = cache.with_suffix(".tmp")
                subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                                "-i", os.fspath(src), "-f", "s16le", "-ac", str(CHANNELS), "-ar", "44100",
                                os.fspath(tmp)], check=True)
                os.replace(tmp, cache)
        return cache

    def read_float(self, path, start, frames, raw_s16=None):
        """(frames, 2) float32 for [start, start+frames) (of `raw_s16` instead, when given)."""
        if raw_s16 is not None:
            raw = np.memmap(raw_s16, dtype="<i2", mode="r").reshape(-1, CHANNELS)
            a = raw[start:start + frames].astype(np.float32) / 32768
        elif path.suffix in (".flac", ".wav"):
            with sf.SoundFile(os.fspath(path)) as f:
                f.seek(start)
                a = f.read(frames, dtype="float32", always_2d=True)[:, :CHANNELS]
        else:
            raw = np.memmap(self.decoded(path), dtype="<i2", mode="r").reshape(-1, CHANNELS)
            a = raw[start:start + frames].astype(np.float32) / 32768
        return audio.fit_length(a, frames, CHANNELS)

    def export(self, song, gains, start, end, tempo=1.0, semi=0, pans=None):
        """Mix the tracks with the given linear gains (and pans, -1..1) into an AAC m4a.

        start/end are frames of the original song; with a tempo change the range is
        taken from the rendered variant. Returns (bytes, filename, log line)."""
        d, m = self.manifest(song)
        if not m:
            raise LookupError("no such song")
        known = {**m.get("mix_files", {}), **m.get("files", {})}
        total, sr = m["format"]["frames"], m["format"]["sample_rate"]
        tempo, semi, key = variant_key(tempo, semi)
        start = max(0, min(int(start), total))
        end = total if end is None else max(start, min(int(end), total))
        if end - start < sr // 10:
            raise ValueError("range too short")
        tag = "" if (start, end) == (0, total) else f"_{start // sr // 60}m{start // sr % 60:02d}-{end // sr // 60}m{end // sr % 60:02d}"
        if key:
            tag += (f"_x{tempo:g}" if tempo != 1.0 else "") + (f"_{semi:+d}" if semi else "")
            start, end = int(round(start / tempo)), int(round(end / tempo))
        mix = np.zeros((end - start, CHANNELS), dtype=np.float32)
        used = []
        for track, g in gains.items():
            g = float(g)
            if track not in known or not (d / track).exists():
                raise LookupError(f"no such track: {track}")
            if not 0 < g <= MAX_GAIN * MAX_GAIN:   # track fader x master fader
                continue
            raw = self.variant(d, m, track, tempo, semi, key) if key else None
            a = g * self.read_float(d / track, start, end - start, raw)
            pan = float((pans or {}).get(track, 0.0))
            if pan:
                # equal-power stereo pan, same formula as the page's StereoPannerNode
                pan = min(max(pan, -1.0), 1.0)
                x = pan + 1 if pan <= 0 else pan
                if pan <= 0:
                    a = np.stack([a[:, 0] + a[:, 1] * np.cos(x * np.pi / 2), a[:, 1] * np.sin(x * np.pi / 2)], axis=1)
                else:
                    a = np.stack([a[:, 0] * np.cos(x * np.pi / 2), a[:, 1] + a[:, 0] * np.sin(x * np.pi / 2)], axis=1)
            mix += a
            used.append(f"{known[track].get('part', track)} {20 * np.log10(g):+.1f}dB" + (f" pan {pan:+.2f}" if pan else ""))
        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        norm = ""
        if peak > EXPORT_CEILING:
            mix *= EXPORT_CEILING / peak
            norm = f", lowered {20 * np.log10(EXPORT_CEILING / peak):.1f}dB to avoid clipping"
        name = f"{m['song']}_mix{tag}.m4a"
        with tempfile.TemporaryDirectory(prefix="mixer-export-") as tmp:
            out = Path(tmp) / "mix.m4a"
            audio.write_m4a(out, mix, sr, codec="aac", bitrate="256k",
                            metadata={"title": Path(name).stem, "album": m["song"]})
            data = out.read_bytes()
        return data, name, f"export {name}: {', '.join(used) or 'silence'}{norm}"

    def pcm(self, path, total_frames, start, frames, raw_s16=None):
        """Interleaved s16le bytes for [start, start+frames), zero-padded past the end."""
        start = max(0, min(start, total_frames))
        frames = max(0, min(frames, MAX_REQUEST_FRAMES, total_frames - start))
        if raw_s16 is not None:
            with open(raw_s16, "rb") as f:
                f.seek(start * CHANNELS * 2)
                data = f.read(frames * CHANNELS * 2)
        elif path.suffix in (".flac", ".wav"):
            with sf.SoundFile(os.fspath(path)) as f:
                f.seek(start)
                a = f.read(frames, dtype="int16", always_2d=True)
            data = np.ascontiguousarray(a[:, :CHANNELS]).tobytes()
        else:
            cache = self.decoded(path)
            with open(cache, "rb") as f:
                f.seek(start * CHANNELS * 2)
                data = f.read(frames * CHANNELS * 2)
        want = frames * CHANNELS * 2
        return data + b"\0" * (want - len(data))


class Handler(BaseHTTPRequestHandler):
    server_version = "stem-mixer"
    lib: Library
    jobs: Jobs
    token: str

    def do_PUT(self):
        url = urlsplit(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if not self.authed():
            return self.send(HTTPStatus.FORBIDDEN, "forbidden")
        if url.path != "/api/upload":
            return self.send(HTTPStatus.NOT_FOUND, "not found")
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_UPLOAD:
                return self.send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "ファイルが大きすぎます（1GB まで）")
            profile = json.loads(q.get("profile") or "{}")
            from .profiles import normalize
            normalize(profile)                         # reject a bad selection before receiving the file
            dst = self.jobs.inbox_path(q.get("name", ""))
            tmp = dst.with_name(dst.name + ".part")
            left = size
            with open(tmp, "wb") as f:
                while left:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        raise ConnectionError("upload interrupted")
                    f.write(chunk)
                    left -= len(chunk)
            os.replace(tmp, dst)
            job = self.jobs.add(dst, profile)
            self.log_message("upload %s (%.1f MB) -> job %s", dst.name, size / 1e6, job["id"])
            return self.send(HTTPStatus.OK, json.dumps(job, ensure_ascii=False), "application/json; charset=utf-8")
        except ValueError as e:
            return self.send(HTTPStatus.BAD_REQUEST, str(e))
        except (ConnectionError, OSError) as e:
            try:
                tmp.unlink(missing_ok=True)
            except NameError:
                pass
            return self.send(HTTPStatus.BAD_REQUEST, f"アップロードに失敗しました: {e}")

    def do_POST(self):
        url = urlsplit(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if not self.authed():
            return self.send(HTTPStatus.FORBIDDEN, "forbidden")
        try:
            if url.path == "/api/retry":
                return self.send(HTTPStatus.OK, json.dumps(self.jobs.retry(q.get("id", ""))), "application/json")
            if url.path == "/api/dismiss":
                self.jobs.dismiss(q.get("id", ""))
                return self.send(HTTPStatus.OK, "{}", "application/json")
            if url.path == "/api/transcribe":
                self.lib.queue_transcription(q.get("song", ""), json.loads(q.get("tracks", "[]")), q.get("model", "auto"))
                return self.send(HTTPStatus.OK, "{}", "application/json")
            if url.path == "/api/reseparate":
                d, m = self.lib.manifest(q.get("song", ""))
                if not m:
                    raise LookupError("no such song")
                src = Path(m.get("input", ""))
                if not src.exists():
                    raise LookupError(f"元の音声ファイルが見つかりません: {src}")
                if any(j["song"] == d.name and j["status"] in ("queued", "running", "analyzing") for j in self.jobs.list()):
                    raise ValueError("この曲はいま処理中です")
                job = self.jobs.add(src, json.loads(q.get("profile") or "{}"))
                return self.send(HTTPStatus.OK, json.dumps(job, ensure_ascii=False), "application/json; charset=utf-8")
            if url.path == "/api/delete":
                dst = self.lib.delete(q.get("song", ""))
                self.log_message("deleted %s -> %s", q.get("song"), dst)
                return self.send(HTTPStatus.OK, "{}", "application/json")
        except (LookupError, ValueError) as e:
            return self.send(HTTPStatus.BAD_REQUEST, str(e))
        return self.send(HTTPStatus.NOT_FOUND, "not found")

    def log_message(self, fmt, *args):
        if not self.path.startswith("/api/pcm"):
            super().log_message(fmt, *args)

    def authed(self):
        for part in self.headers.get("Cookie", "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == COOKIE and hmac.compare_digest(v, self.token):
                return True
        return False

    def send(self, code, body, ctype="text/plain; charset=utf-8", headers=()):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        url = urlsplit(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if "k" in q:
            if not hmac.compare_digest(q["k"], self.token):
                return self.send(HTTPStatus.FORBIDDEN, "トークンが違います")
            cookie = f"{COOKIE}={self.token}; Max-Age={400 * 86400}; Path=/; HttpOnly; SameSite=Lax"
            return self.send(HTTPStatus.SEE_OTHER, "", headers=[("Set-Cookie", cookie), ("Location", "/")])
        if not self.authed():
            return self.send(HTTPStatus.FORBIDDEN,
                             "このページを開くには、サーバーで ./mixer.sh を起動したときに表示される URL（?k=... 付き）を一度開いてください。")
        try:
            if url.path in ("/", "/index.html"):
                return self.static("index.html", cache="no-cache")
            if url.path == "/api/songs":
                body = json.dumps(self.lib.songs(), ensure_ascii=False)
                return self.send(HTTPStatus.OK, body, "application/json; charset=utf-8", [("Cache-Control", "no-cache")])
            if url.path == "/api/jobs":
                return self.send(HTTPStatus.OK, json.dumps(self.jobs.list(), ensure_ascii=False),
                                 "application/json; charset=utf-8", [("Cache-Control", "no-store")])
            if url.path == "/api/notes":
                return self.send(HTTPStatus.OK, json.dumps(self.lib.notes(q.get("song", ""))),
                                 "application/json", [("Cache-Control", "no-store")])
            if url.path == "/api/song":
                body = json.dumps(self.lib.details(q.get("song", "")), ensure_ascii=False)
                return self.send(HTTPStatus.OK, body, "application/json; charset=utf-8", [("Cache-Control", "no-cache")])
            if url.path == "/api/cover":
                d, m = self.lib.manifest(q.get("song", ""))
                f = d / CACHE_DIR / "cover.jpg" if d else None
                if not f or not f.exists():
                    return self.send(HTTPStatus.NOT_FOUND, "no cover")
                return self.send(HTTPStatus.OK, f.read_bytes(), "image/jpeg", [("Cache-Control", "private, max-age=86400")])
            if url.path == "/api/prepare":
                body = self.lib.prepare(q.get("song", ""), json.loads(q.get("tracks", "[]")),
                                        float(q.get("t", 1)), int(q.get("p", 0)))
                return self.send(HTTPStatus.OK, json.dumps(body), "application/json", [("Cache-Control", "no-store")])
            if url.path == "/api/export":
                end = q.get("end")
                data, name, line = self.lib.export(q.get("song", ""), json.loads(q.get("g", "{}")),
                                                   int(q.get("start", 0)), None if end in (None, "") else int(end),
                                                   float(q.get("t", 1)), int(q.get("p", 0)), json.loads(q.get("pan", "{}")))
                self.log_message("%s", line)
                disp = f"attachment; filename=\"mix.m4a\"; filename*=UTF-8''{quote(name)}"
                return self.send(HTTPStatus.OK, data, "audio/mp4",
                                 [("Content-Disposition", disp), ("Cache-Control", "no-store")])
            if url.path == "/api/pcm":
                path, m = self.lib.track_path(q.get("song", ""), q.get("track", ""))
                if path is None or not path.exists():
                    return self.send(HTTPStatus.NOT_FOUND, "no such track")
                tempo, semi, key = variant_key(q.get("t", 1), q.get("p", 0))
                d = self.lib.manifest(q["song"])[0]
                raw = self.lib.variant(d, m, q["track"], tempo, semi, key) if key else None
                total = self.lib.variant_frames(m, tempo) if key else m["format"]["frames"]
                data = self.lib.pcm(path, total, int(q.get("start", 0)), int(q.get("frames", 0)), raw)
                # the URL carries the file mtime (v=), so a cached chunk is never stale
                return self.send(HTTPStatus.OK, data, "application/octet-stream",
                                 [("Cache-Control", "private, max-age=604800")])
        except (BrokenPipeError, ConnectionResetError):
            return
        except (LookupError, ValueError) as e:
            return self.send(HTTPStatus.BAD_REQUEST, str(e))
        except Exception as e:  # noqa: BLE001 — report to the page instead of dropping the connection
            self.log_error("%s: %s", type(e).__name__, e)
            return self.send(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(e).__name__}: {e}")
        return self.send(HTTPStatus.NOT_FOUND, "not found")

    def static(self, name, cache="private, max-age=3600"):
        name = name or "index.html"
        f = (STATIC / name).resolve()
        if not f.is_relative_to(STATIC) or not f.is_file():
            return self.send(HTTPStatus.NOT_FOUND, "not found")
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        return self.send(HTTPStatus.OK, f.read_bytes(), ctype, [("Cache-Control", cache)])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mixer.sh", description="LAN web mixer for separated stems.")
    ap.add_argument("--host", default="0.0.0.0", help="listen address (default: all interfaces, i.e. reachable from the LAN)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--output", default=os.fspath(ROOT / "output"))
    args = ap.parse_args(argv)

    Handler.lib = Library(args.output)
    Handler.jobs = Jobs(Handler.lib)
    Handler.token = load_token()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    host = lan_ip() if args.host in ("0.0.0.0", "") else args.host
    print(f"stem mixer: {len(Handler.lib.songs())} songs in {Handler.lib.out_root}", flush=True)
    threading.Thread(target=Handler.lib.prewarm, daemon=True).start()
    print("スマホ・タブレット・PC のブラウザで、最初に一度だけ次の URL を開いてください（以後は ?k= なしでよい）:", flush=True)
    print(f"  http://{host}:{args.port}/?k={quote(Handler.token)}", flush=True)
    print("停止: Ctrl+C", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
