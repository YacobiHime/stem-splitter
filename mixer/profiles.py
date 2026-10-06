"""Separation settings chosen when a song is added -> a pipeline config.

profile = {
  "mode": "standard" | "4stem" | "2stem" | "custom" | "dnr",
  "parts": {                       # custom only
    "vocals": "one" | "split" | null,      split = lead + chorus
    "drums":  "one" | "split" | null,      split = kick / snare / toms / hihat / cymbals (+ rest)
    "guitar": "one" | "split" | null,      split = acoustic + electric
    "bass" | "piano" | "keys" | "synth" | "organ" | "strings" | "woodwind" | "brass": true / false
  },
  "hifi": bool,                    # test-time augmentation (about 3x slower) + lossless ALAC files
  "refine": {part: method}         # split a mixer part further (any mode), see REFINE
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

# ---- splitting one mixer part further ("refine"). Every method keeps the sum exact: what the
# models leave is the "<part>-rest" track (dropped when empty).
GENDER = "model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt"      # male / female chorus (Sucial)
DUET = "model_mel_band_roformer_ep_0_sdr_7.9319_fixed.ckpt"        # two singers
SATB = "model_scnet_ep_36_sdr_5.4596.ckpt"                         # soprano / alto / tenor / bass (choir)
PIANO = "bs_mega_53stem_piano_mvsep.ckpt"
VOCAL_PARTS = ("chorus", "vocals", "lead")
INSTRUMENT_PARTS = ("keyboards", "upper", "other", "instrumental", "music", "rest_six")
# method -> (parts it applies to, [(model, {model stem: piece suffix})] applied one after another to the remainder)
REFINE = {
    "voice2": (VOCAL_PARTS, [(GENDER, {"male": "v1", "female": "v2"})]),
    "duet": (VOCAL_PARTS, [(DUET, {"singer_1": "s1", "singer_2": "s2"})]),
    "satb": (VOCAL_PARTS, [(SATB, {"soprano": "soprano", "alto": "alto", "tenor": "tenor", "bass": "bass"})]),
    "lead_chorus": (("vocals",), [(KARAOKE, {"Vocals": "lead"})]),
    "kit": (("drums",), [(DRUMSEP, {"kick": "kick", "snare": "snare", "toms": "toms", "hh": "hihat",
                                    "ride": "ride", "crash": "crash"})]),
    "acoustic": (("guitar",), [(ACOUSTIC, {"acoustic-guitar": "acoustic-guitar"})]),
    "instruments": (INSTRUMENT_PARTS, [(PIANO, {"piano": "piano"})] +
                    [(m, {stem: "keys" if part == "keyboards" else part}) for part, m, stem in CASCADE]),
}
# piece names that read better without the parent prefix (if the name is still free)
PLAIN = {"kick", "snare", "toms", "hihat", "piano", "strings", "woodwind", "brass", "organ", "synth", "keys", "acoustic-guitar"}
REST_NAME = {"acoustic": "electric-guitar"}          # what is left has a proper name for some methods


def refine_methods(part):
    """Methods that can split this mixer part further."""
    return [m for m, (parts, _) in REFINE.items() if part in parts or (part.startswith("chorus") and "chorus" in parts)]


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
    refine = {}
    for part, method in (p.get("refine") or {}).items():
        if method not in REFINE:
            raise ValueError(f"unknown split method: {method}")
        refine[str(part)] = method
    return {"mode": mode, "parts": parts, "hifi": bool(p.get("hifi")), "refine": refine}


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


def apply_refine(steps, tracks, refine):
    """Replace each refined mixer part by its pieces (in place); returns {piece: parent}."""
    parents = {}
    for part, method in refine.items():
        if part not in tracks:
            continue                                   # e.g. a part that this mode does not have
        allowed, chain = REFINE[method]
        if part not in allowed and not (part.startswith("chorus") and "chorus" in allowed):
            raise ValueError(f"{part} cannot be split with {method}")
        key = part.replace("-", "_")
        srcs = tracks[part]
        if len(srcs) == 1 and not srcs[0].startswith("-"):
            rest = srcs[0]
        else:                                          # the part is a sum: make it a file first
            rest = f"ref_{key}_in"
            steps.append({"name": f"ref_{key}_in", "combine": list(srcs), "output": rest})
        pieces = {}
        for i, (model, stems) in enumerate(chain):
            outs = {stem: f"ref_{key}_{suffix.replace('-', '_')}" for stem, suffix in stems.items()}
            steps.append({"name": f"ref_{key}_{i}", "model": model, "input": rest, "outputs": outs})
            taken = list(outs.values())
            nxt = f"ref_{key}_rest{i}"
            steps.append({"name": f"ref_{key}_rest{i}", "combine": [rest] + [f"-{w}" for w in taken], "output": nxt})
            for stem, suffix in stems.items():
                name = suffix if suffix in PLAIN else f"{part}-{suffix}"
                if name in tracks and name != part:
                    name = f"{part}-{suffix}"
                pieces.setdefault(name, []).append(outs[stem])
            rest = nxt
        if method == "kit":                            # cymbals = ride + crash, like the custom drum split
            pieces = {("cymbals" if k.endswith("ride") else k): v for k, v in pieces.items() if not k.endswith("crash")}
            pieces["cymbals"] = [f"ref_{key}_ride", f"ref_{key}_crash"]
        if method == "lead_chorus":
            pieces = {"lead": [f"ref_{key}_lead"]}
            pieces["chorus"] = [rest]
        else:
            pieces[REST_NAME.get(method, f"{part}-rest")] = [rest]
        new = {}
        for k, v in tracks.items():                    # keep the order: the pieces go where the part was
            if k == part:
                new.update(pieces)
            else:
                new[k] = v
        tracks.clear()
        tracks.update(new)
        parents.update({piece: part for piece in pieces})
    return parents


def build_config(base, profile):
    """A full pipeline config for this profile (None = use config.yaml unchanged)."""
    p = normalize(profile)
    if p["mode"] == "standard" and not p["hifi"] and not p["refine"]:
        return None
    cfg = copy.deepcopy(base)
    cfg["profile"] = p
    if p["mode"] == "standard" and p["refine"]:
        tracks = {k: list(v) if isinstance(v, list) else [v] for k, v in cfg["mixer"]["tracks"].items()}
        cfg["mixer"]["parents"] = apply_refine(cfg["steps"], tracks, p["refine"])
        cfg["mixer"]["tracks"] = tracks
    if p["mode"] != "standard":
        steps, tracks = build_steps(p)
        parents = apply_refine(steps, tracks, p["refine"])
        cfg["steps"] = steps
        cfg["mixer"] = {"enabled": True, "tracks": tracks, "parents": parents,
                        "drop_threshold_db": (base.get("mixer") or {}).get("drop_threshold_db", -50.0)}
        # Audacity files: the original + the same parts as the mixer
        cfg["deliverables"] = {"original": "original", **tracks}
        cfg["drop_silent"] = {**(cfg.get("drop_silent") or {}), "keep_always": ["original"]}
        cfg["song_overrides"] = {}
    if p["hifi"]:
        cfg["use_tta"] = True
        cfg["output_format"] = {"codec": "alac"}
    return cfg
