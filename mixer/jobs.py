"""Separation jobs for songs added from the page.

Uploaded files go to inbox/. One worker thread runs the normal pipeline (the same as
./separate.sh <file>) for one song at a time, then the mixer analysis (beats / chords), and
reports progress from the pipeline's log lines ("[song] vocals: 8.4s ...") and pymss's
percentages. The job list is kept in .mixer_jobs.json, so jobs that were queued or running
when the server stopped are started again (the pipeline resumes where it stopped).
"""

import json
import os
import re
import subprocess
import threading
import time
import traceback
import uuid
from pathlib import Path

import yaml

from pipeline.run import AUDIO_EXTS, song_name_for

from . import profiles

ROOT = Path(__file__).resolve().parent.parent
JOBS_FILE = ROOT / ".mixer_jobs.json"
KEEP_FINISHED = 30            # finished / failed jobs kept in the list
PCT = re.compile(r"(\d{1,3})%\|")


def pipeline_steps(cfg):
    return ["convert"] + [s["name"] for s in cfg["steps"] if not s.get("optional")] + ["deliverables"]


class Jobs:
    def __init__(self, lib):
        self.lib = lib
        self.lock = threading.Condition()
        self.jobs = []
        if JOBS_FILE.exists():
            try:
                self.jobs = json.loads(JOBS_FILE.read_text())
            except ValueError:
                self.jobs = []
        for j in self.jobs:                      # interrupted by a restart: run again
            if j["status"] in ("running", "analyzing"):
                j.update(status="queued", message="サーバー再起動のため再開待ち")
        self.save()
        threading.Thread(target=self.worker, daemon=True).start()

    # ---- bookkeeping

    def save(self):
        tmp = JOBS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.jobs, ensure_ascii=False, indent=1))
        os.replace(tmp, JOBS_FILE)

    def update(self, job, **kw):
        with self.lock:
            job.update(kw)
            self.save()

    def list(self):
        with self.lock:
            return [dict(j) for j in self.jobs]

    def inbox_path(self, filename):
        """A free inbox/ path for an uploaded file; the song name must not clash with an existing song."""
        ext = Path(filename).suffix.lower()
        if ext not in AUDIO_EXTS:
            raise ValueError(f"対応していない形式です: {ext or '(拡張子なし)'}")
        base = song_name_for(filename)
        taken = {p.name for p in self.lib.out_root.iterdir()} | {j["song"] for j in self.jobs if j["status"] != "error"}
        name, n = base, 2
        while name in taken or (ROOT / "inbox" / f"{name}{ext}").exists():
            name, n = f"{base} ({n})", n + 1
        (ROOT / "inbox").mkdir(exist_ok=True)
        return ROOT / "inbox" / f"{name}{ext}"

    def add(self, path, profile=None):
        profile = profiles.normalize(profile)          # raises ValueError for a bad selection
        job = {"id": uuid.uuid4().hex[:12], "file": os.fspath(path), "song": song_name_for(path), "profile": profile,
               "status": "queued", "step": "", "progress": 0.0, "message": "", "added": time.time()}
        with self.lock:
            self.jobs.append(job)
            done = [j for j in self.jobs if j["status"] in ("done", "error")]
            for old in done[:-KEEP_FINISHED]:
                self.jobs.remove(old)
            self.save()
            self.lock.notify_all()
        return job

    def retry(self, job_id):
        with self.lock:
            for j in self.jobs:
                if j["id"] == job_id and j["status"] == "error":
                    j.update(status="queued", message="", progress=0.0, step="")
                    self.save()
                    self.lock.notify_all()
                    return j
        raise LookupError("no such failed job")

    def dismiss(self, job_id):
        with self.lock:
            self.jobs = [j for j in self.jobs if not (j["id"] == job_id and j["status"] in ("done", "error"))]
            self.save()

    # ---- worker

    def worker(self):
        while True:
            with self.lock:
                while not any(j["status"] == "queued" for j in self.jobs):
                    self.lock.wait()
                job = next(j for j in self.jobs if j["status"] == "queued")
                job.update(status="running", started=time.time(), progress=0.0, message="")
                self.save()
            try:
                self.run(job)
            except Exception as e:  # noqa: BLE001 — a failed song must not stop the queue
                traceback.print_exc()
                self.update(job, status="error", message=f"{type(e).__name__}: {e}")

    def run(self, job):
        base = yaml.safe_load((ROOT / "config.yaml").read_text())
        cfg = profiles.build_config(base, job.get("profile"))
        cmd = [os.fspath(ROOT / ".venv" / "bin" / "python"), "-m", "pipeline", job["file"]]
        if cfg is not None:
            # the song's own config (kept next to its outputs, so a later re-run can be compared)
            d = self.lib.out_root / job["song"]
            d.mkdir(parents=True, exist_ok=True)
            path = d / "separation.yaml"
            path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
            cmd += ["--config", os.fspath(path)]
        steps = pipeline_steps(cfg or base)
        done = set()
        tail = []
        env = dict(os.environ, PYTHONPATH=os.fspath(ROOT), PYTHONUNBUFFERED="1")
        print(f"[job {job['id']}] separate {job['file']}", flush=True)
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        buf = b""
        last_save = 0.0
        while True:
            chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            *lines, buf = re.split(rb"[\r\n]", buf)
            for raw in lines:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                m = re.match(r"\[(.+?)\] (?:Step 0 )?([\w-]+):", line)
                if m and m.group(2) in steps:
                    done.add(m.group(2))
                    tail.append(line)
                    tail = tail[-8:]
                    if "FAILED" in line:
                        job["message"] = line.split("] ", 1)[-1][:300]
                cur = next((s for s in steps if s not in done), steps[-1])
                frac = 0.0
                p = PCT.search(line)
                if p:
                    frac = min(int(p.group(1)), 100) / 100
                job["step"] = cur
                job["progress"] = round((len(done) + frac) / len(steps) * 0.95, 3)   # last 5%: analysis
            if time.time() - last_save > 1.0:
                self.update(job)
                last_save = time.time()
        rc = proc.wait()
        if rc != 0:
            self.update(job, status="error", message=job.get("message") or (tail[-1] if tail else f"exit code {rc}"))
            return
        self.update(job, status="analyzing", step="analysis", progress=0.96)
        self.lib.details(job["song"])
        self.update(job, status="done", step="", progress=1.0, finished=time.time())
        print(f"[job {job['id']}] done: {job['song']}", flush=True)
