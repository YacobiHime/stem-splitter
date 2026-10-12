"""Named playlists (ordered song lists) for the mixer, persisted in .mixer_playlists.json.

The format is [{"name": str, "dirs": [song dir, ...]}, ...]. Song dirs that were deleted
from output/ are kept in the list and skipped at render / playback time, so re-adding a
song (or restoring it from output/.trash/) puts it back in its place.
"""

import json
import os
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLAYLISTS_FILE = ROOT / ".mixer_playlists.json"


class Playlists:
    def __init__(self):
        self.lock = threading.Lock()
        self.items = []
        if PLAYLISTS_FILE.exists():
            try:
                self.items = json.loads(PLAYLISTS_FILE.read_text())
            except ValueError:
                self.items = []

    def save(self):                                   # same atomic write as mixer/jobs.py
        tmp = PLAYLISTS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.items, ensure_ascii=False, indent=1))
        os.replace(tmp, PLAYLISTS_FILE)

    def list(self):
        with self.lock:
            return [dict(p, dirs=list(p["dirs"])) for p in self.items]

    def _find(self, name):
        for p in self.items:
            if p["name"] == name:
                return p
        raise LookupError(f"no such playlist: {name}")

    def _clean(self, name):
        name = (name or "").strip()
        if not name:
            raise ValueError("プレイリスト名が空です")
        return name

    def create(self, name):
        name = self._clean(name)
        with self.lock:
            if any(p["name"] == name for p in self.items):
                raise ValueError(f"プレイリスト「{name}」は既にあります")
            self.items.append({"name": name, "dirs": []})
            self.save()

    def rename(self, name, to):
        to = self._clean(to)
        with self.lock:
            p = self._find(name)
            if to != p["name"] and any(q["name"] == to for q in self.items):
                raise ValueError(f"プレイリスト「{to}」は既にあります")
            p["name"] = to
            self.save()

    def delete(self, name):
        with self.lock:
            self._find(name)
            self.items = [p for p in self.items if p["name"] != name]
            self.save()

    def add(self, name, dir, lib):
        d, m = lib.manifest(dir)                      # also rejects "/" and leading "." in dir
        if not m:
            raise LookupError(f"no such song: {dir}")
        with self.lock:
            p = self._find(name)
            if dir not in p["dirs"]:
                p["dirs"].append(dir)
                self.save()

    def remove(self, name, dir):
        with self.lock:
            p = self._find(name)
            if dir in p["dirs"]:
                p["dirs"].remove(dir)
                self.save()

    def move(self, name, dir, to):
        to = max(0, int(to))
        with self.lock:
            p = self._find(name)
            if dir not in p["dirs"]:
                raise LookupError(f"no such song: {dir}")
            p["dirs"].remove(dir)
            p["dirs"].insert(min(to, len(p["dirs"])), dir)
            self.save()
