"""Separation settings chosen when a song is added -> a pipeline config.

profile = {
  "mode": "standard" | "4stem" | "2stem" | "custom" | "dnr",
  "parts": {                       # custom only
    "vocals": "one" | "split" | null,      split = lead + chorus
    "drums":  "one" | "split" | null,      split = kick / snare / toms / hihat / cymbals (+ rest)
    "guitar": "one" | "split" | null,      split = acoustic + electric
    "bass" | "piano" | "keys" | "synth" | "organ" | "strings" | "woodwind" | "brass": true / false
  },
  "hifi": bool                     # test-time augmentation (about 3x slower) + lossless ALAC files
}

standard = config.yaml as it is (keyboard transcription set). The other modes are cascades:
every chosen part is taken out of what is left so far, and the rest is "other", so the mixer
tracks always add up to the original exactly, whatever is chosen.
"""

import copy

VOCALS = "BS-Roformer-Resurrection.ckpt"
KARAOKE = "bs_roformer_karaoke_frazer_becruily.ckpt"
SIX = "BS-Roformer-SW.ckpt"
DRUMSEP = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
ACOUSTIC = "bs_mega_53stem_acoustic-guitar_mvsep.ckpt"
DNR = "model_bandit_plus_dnr_sdr_11.47.ckpt"
# parts taken from the remainder one after another: (part, model, model stem)
CASCADE = [
    ("strings", "gilliaan_bowedstrings_bs_v1.ckpt", "strings"),
    ("woodwind", "bs_mega_53stem_woodwind_mvsep.ckpt", "woodwind"),
    ("brass", "bs_mega_53stem_brass_mvsep.ckpt", "brass"),
    ("organ", "bs_mega_53stem_organ_mvsep.ckpt", "organ"),
    ("synth", "bs_mega_53stem_synth_mvsep.ckpt", "synth"),
    ("keyboards", "bs_mega_53stem_keys_mvsep.ckpt", "keys"),
]
SIX_PARTS = ("drums", "bass", "guitar", "piano")
TOGGLES = ("bass", "piano", "keys", "synth", "organ", "strings", "woodwind", "brass")


def normalize(profile):
    p = dict(profile or {})
    mode = p.get("mode", "standard")
    if mode not in ("standard", "4stem", "2stem", "custom", "dnr"):
        raise ValueError(f"unknown separation mode: {mode}")
    parts = {}
    if mode == "custom":
        src = p.get("parts") or {}
        for k in ("vocals", "drums", "guitar"):
            v = src.get(k)
            parts[k] = v if v in ("one", "split") else None
        for k in TOGGLES:
            parts[k] = bool(src.get(k))
        if not any(parts.values()):
            raise ValueError("分離するパートを1つ以上選んでください")
    return {"mode": mode, "parts": parts, "hifi": bool(p.get("hifi"))}


def model_steps(p):
    """Number of model runs (for the time estimate on the page)."""
    return sum(1 for s in build_steps(p)[0] if "model" in s)


def build_steps(p):
    """(steps, mixer tracks {part: [work names]}) for a normalized non-standard profile."""
    mode, parts = p["mode"], p["parts"]
    steps, tracks = [], {}
    if mode == "dnr":
        steps.append({"name": "dnr", "model": DNR, "input": "original",
                      "outputs": {"speech": "speech", "music": "music"}})
        steps.append({"name": "dnr_rest", "combine": ["original", "-speech", "-music"], "output": "effects"})
        return steps, {"speech": ["speech"], "music": ["music"], "effects": ["effects"]}

    steps.append({"name": "vocals", "model": VOCALS, "input": "original",
                  "outputs": {"vocals": "vocals", "other": "instrumental"}})
    if mode == "2stem":
        return steps, {"vocals": ["vocals"], "instrumental": ["instrumental"]}
    if mode == "4stem":
        parts = {"vocals": "one", "drums": "one", "bass": True}

    vocals = parts.get("vocals")
    if vocals == "split":
        steps.append({"name": "karaoke", "model": KARAOKE, "input": "vocals",
                      "outputs": {"Vocals": "lead", "Instrumental": "backing"}})
        tracks["lead"], tracks["chorus"] = ["lead"], ["backing"]
    elif vocals == "one":
        tracks["vocals"] = ["vocals"]

    rest = "instrumental"
    six = [k for k in SIX_PARTS if parts.get(k)]
    if six:
        steps.append({"name": "six_stem", "model": SIX, "input": "instrumental",
                      "outputs": {k: k for k in SIX_PARTS}})
        steps.append({"name": "rest_six", "combine": ["instrumental"] + [f"-{k}" for k in six], "output": "rest_six"})
        rest = "rest_six"
        if parts.get("drums") == "split":
            pieces = {"kick": "kick", "snare": "snare", "toms": "toms", "hh": "hihat", "ride": "ride", "crash": "crash"}
            steps.append({"name": "drumsep", "model": DRUMSEP, "input": "drums", "outputs": pieces})
            steps.append({"name": "drums_rest", "combine": ["drums"] + [f"-{v}" for v in pieces.values()],
                          "output": "drums_rest"})
            tracks.update({"kick": ["kick"], "snare": ["snare"], "toms": ["toms"], "hihat": ["hihat"],
                           "cymbals": ["ride", "crash"], "drums-other": ["drums_rest"]})
        elif parts.get("drums"):
            tracks["drums"] = ["drums"]
        if parts.get("bass"):
            tracks["bass"] = ["bass"]
        if parts.get("guitar") == "split":
            steps.append({"name": "acoustic", "model": ACOUSTIC, "input": "guitar",
                          "outputs": {"acoustic-guitar": "acoustic_guitar"}})
            tracks["acoustic-guitar"] = ["acoustic_guitar"]
            tracks["electric-guitar"] = ["guitar", "-acoustic_guitar"]
        elif parts.get("guitar"):
            tracks["guitar"] = ["guitar"]
        if parts.get("piano"):
            tracks["piano"] = ["piano"]

    for part, model, stem in CASCADE:
        key = "keys" if part == "keyboards" else part
        if not parts.get(key):
            continue
        steps.append({"name": f"take_{part}", "model": model, "input": rest, "outputs": {stem: f"take_{part}"}})
        steps.append({"name": f"rest_{part}", "combine": [rest, f"-take_{part}"], "output": f"rest_{part}"})
        tracks[part] = [f"take_{part}"]
        rest = f"rest_{part}"

    tracks["other"] = [rest] + (["vocals"] if not vocals else [])
    return steps, tracks


def build_config(base, profile):
    """A full pipeline config for this profile (None = use config.yaml unchanged)."""
    p = normalize(profile)
    if p["mode"] == "standard" and not p["hifi"]:
        return None
    cfg = copy.deepcopy(base)
    cfg["profile"] = p
    if p["mode"] != "standard":
        steps, tracks = build_steps(p)
        cfg["steps"] = steps
        cfg["mixer"] = {"enabled": True, "tracks": tracks}
        # Audacity files: the original + the same parts as the mixer
        cfg["deliverables"] = {"original": "original", **tracks}
        cfg["drop_silent"] = {**(cfg.get("drop_silent") or {}), "keep_always": ["original"]}
        cfg["song_overrides"] = {}
    if p["hifi"]:
        cfg["use_tta"] = True
        cfg["output_format"] = {"codec": "alac"}
    return cfg
