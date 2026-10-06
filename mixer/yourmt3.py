"""YourMT3+ (Chang et al. 2024, "YourMT3+: Multi-instrument Music Transcription with Enhanced Transformer
Architectures and Cross-dataset Stem Augmentation") on one track, run as its own process:

    python -m mixer.yourmt3 <audio file> <out.json>

Prints "progress <0..1>" lines while it runs and writes [[start s, end s, midi pitch, GM program], ...].
Code and weights: vendor/YourMT3 (the authors' HuggingFace Space mimbres/YourMT3, checkpoint
"YPTF.MoE+Multi (noPS)"; see models/MODELS.md). A separate process because its top-level packages
(utils, model, config) clash with vendor/BTC-ISMIR19 used by the mixer, and the GPU memory is freed when done.
"""

import contextlib
import io
import json
import sys
import types
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
YMT3_DIR = ROOT / "vendor" / "YourMT3"
CHECKPOINT = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
# the Space's arguments for this checkpoint (app.py)
ARGS = [CHECKPOINT, "-p", "2024", "-tk", "mc13_full_plus_256", "-dec", "multi-t5", "-nl", "26", "-enc", "perceiver-tf",
        "-sqr", "1", "-ff", "moe", "-wf", "4", "-nmoe", "8", "-kmoe", "2", "-act", "silu", "-epe", "rope",
        "-rp", "1", "-ac", "spec", "-hop", "300", "-atc", "1", "-pr", "16", "-w", "0"]
CHUNK = 16          # segments (2.048 s each) per inference call, so progress can be reported


def available():
    return (YMT3_DIR / "amt" / "logs" / "2024" / CHECKPOINT.split("@")[0] / "checkpoints" / CHECKPOINT.split("@")[1]).exists()


def load(device):
    sys.path[:0] = [str(YMT3_DIR), str(YMT3_DIR / "amt" / "src")]
    # wandb is only used for training logs; keep vendor/ unmodified and stub it
    wandb = sys.modules.setdefault("wandb", types.ModuleType("wandb"))
    wandb.Table = lambda **kw: None
    warnings.filterwarnings("ignore")
    from config import config as cfg
    cfg.shared_cfg["WANDB"]["save_dir"] = str(YMT3_DIR / "amt" / "logs")   # relative to the Space's cwd otherwise
    from model_helper import load_model_checkpoint
    with contextlib.redirect_stdout(io.StringIO()):
        return load_model_checkpoint(args=ARGS, device="cpu").to(device)


def transcribe(path, progress=lambda f: None):
    """[(start, end, pitch, program)] for one audio file (drums dropped)."""
    import torch
    from pipeline import audio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(device)
    from utils.audio import slice_padded_array
    from utils.event2note import merge_zipped_note_events_and_ties_to_notes
    from utils.note2event import mix_notes

    sr, n = model.audio_cfg["sample_rate"], model.audio_cfg["input_frames"]
    y = audio.decode(path, sample_rate=sr, channels=1)[:, 0]
    seg = torch.from_numpy(slice_padded_array(y[None], n, n).astype("float32")).unsqueeze(1)
    pred = []
    with torch.inference_mode(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        for i in range(0, len(seg), CHUNK):
            out, _ = model.inference_file(bsz=8, audio_segments=seg[i:i + CHUNK].to(device))
            pred += out
            progress(min(len(seg), i + CHUNK) / len(seg))
    starts = [n * i / sr for i in range(len(seg))]
    channels = []
    for ch in range(model.task_manager.num_decoding_channels):
        zipped, _, _ = model.task_manager.detokenize_list_batches([a[:, ch, :] for a in pred], starts, return_events=True)
        channels.append(merge_zipped_note_events_and_ties_to_notes(zipped)[0])
    return [(float(x.onset), float(x.offset), int(x.pitch), int(x.program))
            for x in mix_notes(channels) if not x.is_drum and x.offset > x.onset]


if __name__ == "__main__":
    src, dst = sys.argv[1:3]
    notes = transcribe(src, lambda f: print(f"progress {f:.3f}", flush=True))
    Path(dst).write_text(json.dumps(notes))
