"""Cascade stem separation pipeline for keyboard transcription.

Steps run model-major (each model is loaded once and applied to every song
that needs it). Every intermediate file lives in output/<song>/work/, and a
step is skipped when its outputs exist, are newer than its input, and were
produced by the model currently configured (unless --force).
"""

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

from . import audio

ROOT = Path(__file__).resolve().parent.parent
UPPER_INPUTS = ["instrumental", "drums", "bass", "guitar", "piano", "sw_other"]
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif", ".alac", ".ape", ".wv"}


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def song_name_for(path):
    """Keep Japanese / spaces / '!!!!!' as is; only drop characters that break paths."""
    name = re.sub(r"[\x00-\x1f/\\]", "_", Path(path).stem).strip()
    name = name.lstrip(".")
    return name or "untitled"


class Song:
    def __init__(self, src, out_root):
        self.src = Path(src).resolve()
        self.name = song_name_for(src)
        self.dir = out_root / self.name
        self.work = self.dir / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.state_path = self.work / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"steps": {}}
        self.failed = False
        self.log_fh = open(self.dir / "log.txt", "a", encoding="utf-8")

    def wav(self, name):
        return self.work / f"{name}.wav"

    def log(self, msg):
        line = f"[{now()}] {msg}"
        self.log_fh.write(line + "\n")
        self.log_fh.flush()
        print(f"[{self.name}] {msg}", flush=True)

    def save_state(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2))
        os.replace(tmp, self.state_path)

    def record(self, step, **info):
        self.state["steps"][step] = {**info, "finished": now()}
        self.save_state()


def is_fresh(song, step_name, outputs, inputs, signature, force):
    """True when the step can be skipped."""
    if force:
        return False
    rec = song.state["steps"].get(step_name)
    if not rec or rec.get("signature") != signature:
        return False
    out_paths = [song.wav(o) for o in outputs]
    if not all(p.exists() for p in out_paths):
        return False
    oldest_out = min(p.stat().st_mtime for p in out_paths)
    return all(song.wav(i).stat().st_mtime <= oldest_out for i in inputs if song.wav(i).exists())


# ---------------------------------------------------------------- models

class ModelCache:
    def __init__(self, cfg):
        self.model_dir = (ROOT / cfg["model_dir"]).resolve()
        self.source = cfg.get("download_source", "huggingface")
        self.device = cfg.get("device", "cuda")
        self.sha_cache_path = self.model_dir / ".sha256_cache.json"
        self.sha_cache = json.loads(self.sha_cache_path.read_text()) if self.sha_cache_path.exists() else {}

    def resolve(self, name):
        from pymss import download_model, resolve_model

        info = resolve_model(name, model_dir=self.model_dir, require_supported=True, require_exists=False)
        if info.get("source") != "user" and not Path(info["model_path"]).exists():
            print(f"downloading {name} from {self.source} ...", flush=True)
            download_model(name, model_dir=self.model_dir, source=self.source)
            info = resolve_model(name, model_dir=self.model_dir, require_supported=True, require_exists=True)
        return info

    def sha256(self, path):
        path = Path(path)
        st = path.stat()
        key = os.fspath(path)
        hit = self.sha_cache.get(key)
        if hit and hit["size"] == st.st_size and hit["mtime"] == st.st_mtime:
            return hit["sha256"]
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        self.sha_cache[key] = {"size": st.st_size, "mtime": st.st_mtime, "sha256": h.hexdigest()}
        self.sha_cache_path.write_text(json.dumps(self.sha_cache, indent=2))
        return h.hexdigest()

    def describe(self, name):
        info = self.resolve(name)
        return {
            "name": name,
            "model_type": info.get("model_type"),
            "path": os.fspath(Path(info["model_path"]).relative_to(ROOT)) if Path(info["model_path"]).is_relative_to(ROOT) else info["model_path"],
            "sha256": self.sha256(info["model_path"]),
        }

    def load(self, name, inference_params=None, use_tta=False):
        from pymss import MSSeparator

        self.resolve(name)
        quiet = logging.getLogger("stem-sep.pymss")
        quiet.handlers[:] = [logging.NullHandler()]
        quiet.propagate = False
        params = {"normalize": False}
        params.update(inference_params or {})
        return MSSeparator.from_model_name(
            name, model_dir=self.model_dir, download=False, device=self.device,
            logger=quiet, inference_params=params, use_tta=use_tta,
        )


class VRAM:
    """Peak CUDA memory (MiB) between reset() and peak()."""

    def __init__(self):
        import torch
        self.torch = torch
        self.ok = torch.cuda.is_available()

    def reset(self):
        if self.ok:
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()

    def peak(self):
        if not self.ok:
            return None
        self.torch.cuda.synchronize()
        return {
            "allocated_mib": round(self.torch.cuda.max_memory_allocated() / 2**20),
            "reserved_mib": round(self.torch.cuda.max_memory_reserved() / 2**20),
        }


def fmt_vram(v):
    return "n/a" if not v else f"{v['allocated_mib']} MiB alloc / {v['reserved_mib']} MiB reserved"


def separate(sep, song, input_name, wanted):
    """Run one separator on work/<input_name>.wav; return {model_stem_lower: (frames, ch)}."""
    mix, sr = audio.read(song.wav(input_name))
    res = sep.separate(mix.T, pbar=True, stems=None if wanted is None else list(wanted))
    out = {k.lower(): np.asarray(v, dtype=np.float32) for k, v in res.items()}
    return out, mix.shape[0], sr


# ---------------------------------------------------------------- steps

def step_convert(songs, cfg, force):
    sr = cfg["sample_rate"]
    for song in songs:
        sig = {"src": os.fspath(song.src), "src_mtime": song.src.stat().st_mtime, "sr": sr}
        rec = song.state["steps"].get("convert")
        if not force and rec and rec.get("signature") == sig and song.wav("original").exists():
            song.log("Step 0 convert: skip (exists)")
            continue
        t = time.time()
        try:
            audio.convert_to_wav(song.src, song.wav("original"), sr)
        except Exception as e:
            song.failed = True
            song.log(f"Step 0 convert: FAILED {e}")
            continue
        dur = time.time() - t
        a, _ = audio.read(song.wav("original"))
        song.log(f"Step 0 convert: {dur:.1f}s ({a.shape[0] / sr:.1f}s of audio)")
        song.record("convert", signature=sig, time_s=round(dur, 2), frames=a.shape[0])


def step_model(step, songs, cfg, models, vram, force):
    name, model = step["name"], step["model"]
    inp, outputs = step["input"], step["outputs"]
    params = step.get("inference_params") or {}
    tta = bool(step.get("use_tta", cfg.get("use_tta", False)))
    sig = {"model": model, "outputs": outputs, "inference_params": params}
    if tta:
        sig["use_tta"] = True          # only when on, so existing results stay valid

    todo = []
    for song in songs:
        if song.failed:
            continue
        if not song.wav(inp).exists():
            song.log(f"{name}: skip (input '{inp}' missing)")
            continue
        if is_fresh(song, name, outputs.values(), [inp], sig, force):
            song.log(f"{name}: skip (exists)")
            continue
        todo.append(song)
    if not todo:
        return

    vram.reset()
    t = time.time()
    try:
        sep = models.load(model, params, use_tta=tta)
    except Exception as e:
        for song in todo:
            song.log(f"{name}: FAILED to load model {model}: {e}")
            song.failed = True
        return
    load_s, load_vram = time.time() - t, vram.peak()

    try:
        for song in todo:
            vram.reset()
            t = time.time()
            try:
                res, frames, sr = separate(sep, song, inp, outputs.keys())
                for stem, dst in outputs.items():
                    if stem.lower() not in res:
                        raise KeyError(f"model returned {list(res)}; '{stem}' not found")
                    audio.write(song.wav(dst), audio.fit_length(res[stem.lower()], frames), sr)
            except Exception as e:
                song.failed = True
                song.log(f"{name}: FAILED {type(e).__name__}: {e}")
                song.log_fh.write(traceback.format_exc())
                continue
            dur, peak = time.time() - t, vram.peak()
            song.log(f"{name}: {dur:.1f}s (+model load {load_s:.1f}s), VRAM peak {fmt_vram(peak)} [{model}]")
            song.record(name, signature=sig, model=model, time_s=round(dur, 2), load_s=round(load_s, 2),
                        vram_peak=peak, vram_load=load_vram)
    finally:
        sep.close()


def step_upper(songs, cfg, force):
    method = cfg.get("upper_method", "subtract")
    if method not in ("subtract", "sum"):
        raise SystemExit(f"upper_method must be subtract or sum, got {method!r}")
    sig = {"method": method}
    outs = ["upper_subtract", "upper_sum", "upper"]
    for song in songs:
        if song.failed:
            continue
        missing = [i for i in UPPER_INPUTS if not song.wav(i).exists()]
        if missing:
            song.log(f"upper: skip (missing {missing})")
            continue
        if is_fresh(song, "upper", outs, UPPER_INPUTS, sig, force):
            song.log("upper: skip (exists)")
            continue
        t = time.time()
        a = {k: audio.read(song.wav(k))[0] for k in UPPER_INPUTS}
        sr = cfg["sample_rate"]
        sub = a["instrumental"] - a["drums"] - a["bass"]
        add = a["guitar"] + a["piano"] + a["sw_other"]
        audio.write(song.wav("upper_subtract"), sub, sr)
        audio.write(song.wav("upper_sum"), add, sr)
        audio.write(song.wav("upper"), sub if method == "subtract" else add, sr)
        diff = audio.stats(sub - add)
        song.log(f"upper: {time.time() - t:.1f}s method={method}, subtract-sum residual peak {diff['peak_dbfs']} dBFS, rms {diff['rms_dbfs']} dBFS")
        song.record("upper", signature=sig, time_s=round(time.time() - t, 2), residual=diff)


def step_combine(step, songs, cfg, force):
    """Built-in step: work/<output>.wav = sum of work files ("-name" is subtracted).
    Used for cascades: what is left after taking instruments out."""
    name, srcs, out = step["name"], step["combine"], step["output"]
    inputs = [s.lstrip("-") for s in srcs]
    sig = {"combine": srcs}
    for song in songs:
        if song.failed:
            continue
        missing = [i for i in inputs if not song.wav(i).exists()]
        if missing:
            song.log(f"{name}: skip (missing {missing})")
            continue
        if is_fresh(song, name, [out], inputs, sig, force):
            song.log(f"{name}: skip (exists)")
            continue
        t = time.time()
        frames = audio.read(song.wav(inputs[0]))[0].shape[0]
        total = None
        for s in srcs:
            a = audio.fit_length(audio.read(song.wav(s.lstrip("-")))[0], frames)
            total = (-a if s.startswith("-") else a) if total is None else (total - a if s.startswith("-") else total + a)
        audio.write(song.wav(out), total, cfg["sample_rate"])
        song.log(f"{name}: {time.time() - t:.1f}s {out} = {' '.join(srcs)}")
        song.record(name, signature=sig, time_s=round(time.time() - t, 2))


def step_explore(songs, cfg, models, vram, force):
    ex = cfg["explore"]
    model, inp = ex["model"], ex["input"]
    params = ex.get("inference_params") or {}
    sig = {"model": model, "input": inp, "inference_params": params}
    todo = []
    for song in songs:
        if song.failed or not song.wav(inp).exists():
            continue
        done = song.dir / "explore" / ".done.json"
        if not force and done.exists() and json.loads(done.read_text()) == sig and done.stat().st_mtime >= song.wav(inp).stat().st_mtime:
            song.log("explore: skip (exists)")
            continue
        todo.append(song)
    if not todo:
        return
    vram.reset()
    t = time.time()
    sep = models.load(model, params)
    load_s = time.time() - t
    try:
        for song in todo:
            vram.reset()
            t = time.time()
            try:
                # pymss keeps the overlap-add buffer for every requested stem on the GPU
                # (stems x full song length), so run the stems in groups to bound VRAM.
                instruments = list(sep.config.training.instruments)
                per_pass = int(ex.get("stems_per_pass") or len(instruments))
                exdir = song.dir / "explore"
                n = 0
                for i in range(0, len(instruments), per_pass):
                    res, frames, sr = separate(sep, song, inp, instruments[i:i + per_pass])
                    for stem, a in res.items():
                        audio.write(exdir / f"{stem}.wav", audio.fit_length(a, frames), sr)
                    n += len(res)
                    del res
                (exdir / ".done.json").write_text(json.dumps(sig))
            except Exception as e:
                song.failed = True
                song.log(f"explore: FAILED {type(e).__name__}: {e}")
                song.log_fh.write(traceback.format_exc())
                continue
            peak = vram.peak()
            song.log(f"explore: {time.time() - t:.1f}s (+model load {load_s:.1f}s), {n} stems in passes of {per_pass}, VRAM peak {fmt_vram(peak)} [{model}]")
            song.record("explore", signature=sig, model=model, time_s=round(time.time() - t, 2), vram_peak=peak)
    finally:
        sep.close()


def song_deliverables(song, cfg):
    """{part: [work names]} for this song, with song_overrides.<song>.merge applied.

    A deliverable is one work name or a list of them; "-name" is subtracted.
    A merge {new_part: [part_a, part_b]} replaces those parts with one file that
    is their sum, placed where the first of them was."""
    parts = {k: list(v) if isinstance(v, list) else [v] for k, v in cfg["deliverables"].items()}
    over = (cfg.get("song_overrides") or {}).get(song.name) or {}
    for new_part, members in (over.get("merge") or {}).items():
        unknown = [m for m in members if m not in parts]
        if unknown:
            raise SystemExit(f"song_overrides.{song.name}.merge.{new_part}: unknown parts {unknown}")
        merged, inserted = {}, False
        for part, srcs in parts.items():
            if part in members:
                if not inserted:
                    merged[new_part] = [s for m in members for s in parts[m]]
                    inserted = True
            else:
                merged[part] = srcs
        parts = merged
    return parts


def assemble(song, parts, frames, sr, label):
    """Sum/subtract work files: {part: [work names]} -> ({part: audio}, {parts that are a single exact work file})."""
    tracks, exact = {}, set()
    for out_name, srcs in parts.items():
        missing = [s.lstrip("-") for s in srcs if not song.wav(s.lstrip("-")).exists()]
        if missing:
            song.log(f"{label}: {out_name}: missing work/{missing[0]}.wav")
            continue
        total, ok = None, True
        for s in srcs:
            sign, name = (-1.0, s[1:]) if s.startswith("-") else (1.0, s)
            a, file_sr = audio.read(song.wav(name))
            if file_sr != sr:
                song.log(f"{label}: {out_name}: unexpected sample rate {file_sr} in {s}")
                ok = False
                break
            if srcs == [name] and a.shape == (frames, 2):
                exact.add(out_name)
            a = sign * audio.fit_length(a, frames)
            total = a if total is None else total + a
        if ok:
            tracks[out_name] = total
    merged = {k: v for k, v in parts.items() if len(v) > 1 and k in tracks}
    if merged:
        expr = lambda v: " ".join(("- " + s[1:]) if s.startswith("-") else ("+ " + s) for s in v).lstrip("+ ")
        song.log(f"{label}: combined " + ", ".join(f"{k} = {expr(v)}" for k, v in merged.items()))
    return tracks, exact


def drop_silent(song, cfg, tracks, ref_db, label, keep=(), threshold=None):
    """Remove (in place) tracks where the model found nothing (only faint bleed left); return {part: rel dB}."""
    dropped = {}
    ds = cfg.get("drop_silent") or {}
    if not ds.get("enabled", False):
        return dropped
    if threshold is None:
        threshold = ds.get("threshold_db", -30.0)
    for out_name in list(tracks):
        if out_name in keep:
            continue
        rel = audio.loudest_window_dbfs(tracks[out_name], cfg["sample_rate"]) - ref_db
        if rel < threshold:
            dropped[out_name] = round(float(rel), 1)
            del tracks[out_name]
    if dropped:
        song.log(f"{label}: not saved (nothing detected): " + ", ".join(f"{k} ({v} dB)" for k, v in dropped.items()))
    return dropped


def source_expr(srcs):
    return " ".join(("-" if s.startswith("-") else "+") + f"work/{s.lstrip('-')}.wav" for s in srcs).lstrip("+")


def build_deliverables(song, cfg, models, steps):
    sr = cfg["sample_rate"]
    orig, _ = audio.read(song.wav("original"))
    frames = orig.shape[0]
    parts = song_deliverables(song, cfg)
    tracks, exact = assemble(song, parts, frames, sr, "deliverables")
    ref_db = audio.loudest_window_dbfs(orig, sr)
    dropped = drop_silent(song, cfg, tracks, ref_db, "deliverables",
                          keep=set((cfg.get("drop_silent") or {}).get("keep_always", [])))
    exact &= set(tracks)

    # Mixer tracks (output/<song>/mix/): a set of parts that do not overlap, so that
    # playing all of them at 0 dB gives back the original.
    mix_cfg = cfg.get("mixer") or {}
    mix_parts, mix_tracks, mix_dropped = {}, {}, {}
    if mix_cfg.get("enabled", False):
        mix_parts = {k: list(v) if isinstance(v, list) else [v] for k, v in (mix_cfg.get("tracks") or {}).items()}
        mix_tracks, _ = assemble(song, mix_parts, frames, sr, "mix")
        if len(mix_tracks) == len(mix_parts):
            resid = orig - sum(mix_tracks.values())
            song.log(f"mix: sum of {len(mix_tracks)} tracks vs original: residual peak {audio.stats(resid)['peak_dbfs']} dBFS")
        # stricter than for the Audacity files: the mixer tracks must still add up to the original,
        # so only parts that are really empty are left out
        mix_dropped = drop_silent(song, cfg, mix_tracks, ref_db, "mix", threshold=mix_cfg.get("drop_threshold_db", -50.0))

    fmt = cfg.get("output_format") or {}
    codec = fmt.get("codec", "wav")
    if codec not in ("wav", "aac", "alac"):
        raise SystemExit(f"output_format.codec must be wav, aac or alac, got {codec!r}")
    ext = ".wav" if codec == "wav" else ".m4a"
    # lossy AAC overshoots the input peak slightly, so leave 1 dB of headroom there
    ceiling = 10 ** (-1 / 20) if codec == "aac" else 0.999

    # the same gain for deliverables and mix tracks keeps their levels comparable in the mixer
    peak = max((float(np.max(np.abs(a))) for a in [*tracks.values(), *mix_tracks.values()]), default=0.0)
    gain = 1.0
    if cfg.get("global_gain_on_clip", True) and peak > ceiling:
        gain = ceiling / peak
        song.log(f"deliverables: max peak {peak:.4f} > {ceiling:.3f} -> global gain {20 * np.log10(gain):.2f} dB applied to all files")

    pattern = cfg.get("filename_pattern", "{song}_{part}")
    order = list(parts)
    names = {part: pattern.format(song=song.name, part=part, nn=f"{order.index(part):02d}") + ext for part in tracks}

    # remove outputs of earlier runs that this run does not produce (dropped part,
    # changed format or filename pattern, old NN_ naming)
    old = set()
    mf = song.dir / "manifest.json"
    if mf.exists():
        try:
            old |= set(json.loads(mf.read_text()).get("files", {}))
        except ValueError:
            pass
    old |= {f.name for pat in ("[0-9][0-9]_*.wav", "[0-9][0-9]_*.m4a") for f in song.dir.glob(pat)}
    for name in old - set(names.values()):
        (song.dir / name).unlink(missing_ok=True)
    mix_dir = song.dir / "mix"
    # always lossless: with AAC the parts no longer cancel out exactly and their sum is ~-26 dB off the original
    mix_names = {part: f"mix/{part}.flac" for part in mix_tracks}
    if mix_dir.exists():
        for f in mix_dir.iterdir():
            if f.is_file() and f"mix/{f.name}" not in mix_names.values():
                f.unlink()

    def save(dst, a, title, link_src=None):
        if gain != 1.0:
            a = a * gain
        if dst.suffix == ".flac":
            audio.write_flac(dst, a, sr)
        elif codec != "wav":
            audio.write_m4a(dst, a, sr, codec=codec, bitrate=fmt.get("bitrate", "256k"),
                            metadata={"title": title, "album": song.name})
        elif gain == 1.0 and link_src is not None:
            # identical content: hardlink (no extra disk) — survives `--clean` of work/
            tmp = dst.with_name(dst.stem + ".tmp.wav")
            tmp.unlink(missing_ok=True)
            os.link(link_src, tmp)
            os.replace(tmp, dst)
        else:
            audio.write(dst, a, sr)
        return audio.stats(a)

    files = {}
    for out_name, a in tracks.items():
        dst = song.dir / names[out_name]
        link_src = song.wav(parts[out_name][0]) if out_name in exact else None
        files[dst.name] = {"part": out_name, "source": source_expr(parts[out_name]),
                           **save(dst, a, f"{song.name}_{out_name}", link_src)}
    mix_files = {}
    for out_name, a in mix_tracks.items():
        dst = song.dir / mix_names[out_name]
        dst.parent.mkdir(exist_ok=True)
        mix_files[mix_names[out_name]] = {"part": out_name, "source": source_expr(mix_parts[out_name]),
                                          **save(dst, a, f"{song.name}_mix_{out_name}")}
        parent = (mix_cfg.get("parents") or {}).get(out_name)
        if parent:
            mix_files[mix_names[out_name]]["parent"] = parent      # split from this part (mixer "さらに分離")

    used = {}
    for step in steps:
        if "model" in step and step["name"] in song.state["steps"]:
            used[step["name"]] = models.describe(step["model"])
    if "explore" in song.state["steps"]:
        used["explore"] = models.describe(cfg["explore"]["model"])

    manifest = {
        "song": song.name,
        "input": os.fspath(song.src),
        "generated": now(),
        "format": {"sample_rate": sr, "channels": 2, "codec": codec,
                   "bitrate": fmt.get("bitrate", "256k") if codec == "aac" else None,
                   "frames": frames, "seconds": round(frames / sr, 3)},
        "upper_method": cfg.get("upper_method", "subtract"),
        "global_gain": gain,
        "models": used,
        "steps": song.state["steps"],
        "files": files,
        "dropped_silent": dropped,
        "mix_files": mix_files,
        "mix_dropped_silent": mix_dropped,
        "profile": cfg.get("profile"),             # separation settings chosen in the mixer (None = config.yaml)
    }
    (song.dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    total = sum(s.get("time_s", 0) + s.get("load_s", 0) for s in song.state["steps"].values())
    song.log(f"deliverables: {len(files)} files + {len(mix_files)} mix tracks written, cumulative step time {total:.1f}s")


# ---------------------------------------------------------------- main

def collect_inputs(paths):
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix.lower() in AUDIO_EXTS)
        elif p.exists():
            out.append(p)
        else:
            print(f"not found: {p}", file=sys.stderr)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="separate.sh", description="Cascade stem separation for keyboard transcription.")
    ap.add_argument("inputs", nargs="+", help="audio files or folders")
    ap.add_argument("--force", action="store_true", help="ignore existing outputs and reprocess everything")
    ap.add_argument("--explore", action="store_true", help="also save every stem of the explore model to output/<song>/explore/")
    ap.add_argument("--optional", action="store_true", help="also run steps marked optional (comparison models)")
    ap.add_argument("--keep-work", action="store_true", help="keep work/ (default behaviour; accepted for compatibility)")
    ap.add_argument("--clean", action="store_true", help="delete work/ after building outputs (later runs must redo all steps)")
    ap.add_argument("--config", default=os.fspath(ROOT / "config.yaml"))
    ap.add_argument("--output", default=os.fspath(ROOT / "output"))
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    inputs = collect_inputs(args.inputs)
    if not inputs:
        print("no input files", file=sys.stderr)
        return 2

    out_root = Path(args.output).resolve()
    songs, seen = [], set()
    for p in inputs:
        s = Song(p, out_root)
        if s.name in seen:
            print(f"skip duplicate song name: {p}", file=sys.stderr)
            continue
        seen.add(s.name)
        songs.append(s)

    for s in songs:
        s.log(f"===== run start: {s.src} (force={args.force}, explore={args.explore}, optional={args.optional})")

    models, vram = ModelCache(cfg), VRAM()
    steps = [st for st in cfg["steps"] if args.optional or not st.get("optional")]
    t0 = time.time()

    step_convert(songs, cfg, args.force)
    for st in steps:
        if st["name"] == "upper":
            step_upper(songs, cfg, args.force)
        elif "combine" in st:
            step_combine(st, songs, cfg, args.force)
        else:
            step_model(st, songs, cfg, models, vram, args.force)
    if args.explore:
        step_explore(songs, cfg, models, vram, args.force)

    rc = 0
    for s in songs:
        if s.wav("original").exists():
            try:
                build_deliverables(s, cfg, models, steps)
            except Exception as e:
                s.failed = True
                s.log(f"deliverables: FAILED {e}")
                s.log_fh.write(traceback.format_exc())
        if s.failed:
            rc = 1
            s.log("===== finished WITH ERRORS (re-run to resume)")
        else:
            s.log(f"===== finished OK ({time.time() - t0:.1f}s wall for this run)")
            if args.clean:
                shutil.rmtree(s.work)
                s.log("work/ removed (--clean)")
        s.log_fh.close()
    return rc
