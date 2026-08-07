#!/usr/bin/env python3
"""
digest_movie.py — turn a local video into something Claude can actually read.

Claude cannot decode video or hear audio. This script does the mechanical half
so the model can do the reasoning half: it transcribes the spoken audio with
timestamps and exports one small keyframe per scene. Claude then reads
`transcript.md` and views the frames.

The audio track is the point. A screen recording where someone narrates a bug
is USELESS as frames alone — the whole report is in the voice. Transcribe first,
read the transcript, THEN look at the frames.

Outputs (under --out, default <video>.digest):
  transcript.md    timestamped, grep-friendly  <- READ THIS FIRST
  transcript.srt   subtitles
  transcript.json  raw {start, end, text} segments
  frames/          NNNN_HHhMMmSSs.jpg keyframes (downscaled)
  frames_index.md  frame -> timestamp + change score + pointer table
  digest.md        transcript + frames interleaved by time
  report.html      self-contained review (embedded images, email-ready)
  bug_report.md    structured findings (with --analyze)
  clicks.json      detected click/action moments (diff mode only)
  manifest.json    metadata, params, full frame list, transcript paths

Optimized for QA / bug-report screen recordings. By default it keeps only the
frames that CHANGED (diff-based selection, not fixed intervals — a static screen
stops flooding you with duplicates) and localizes the POINTER on each: the
centroid of what changed vs the previous frame is roughly where the cursor /
action was, the one thing raw frames never tell you. --no-dedup restores plain
scene/interval sampling.

Requires: ffmpeg + ffprobe on PATH  (brew install ffmpeg).
Diff mode uses Pillow + numpy (auto-detected; falls back to interval if absent).
Optional: faster-whisper (transcription), scenedetect (used only with --no-dedup).

Usage:
  python3 digest_movie.py CLIP.mov
  python3 digest_movie.py CLIP.mov --out ./CLIP.digest --model tiny --max-frames 25
  python3 digest_movie.py CLIP.mov --no-frames        # transcript only, fast

Authored for Bloody Finger Software 2026-07-23. Portable — no machine-specific
paths. See SKILL.md for the agent workflow and the filename/model gotchas.
"""

import argparse
import base64
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from html import escape

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


def eprint(*a):
    print(*a, file=sys.stderr)


def hms(seconds: float, compact: bool = False) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h{m:02d}m{s:02d}s" if compact else f"{h:02d}:{m:02d}:{s:02d}"


def srt_time(seconds: float) -> str:
    total_ms = int(max(0.0, float(seconds)) * 1000)
    h, rem = divmod(total_ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def require_tool(name: str):
    if shutil.which(name) is None:
        eprint(f"ERROR: '{name}' not found on PATH. Install ffmpeg (brew install ffmpeg).")
        sys.exit(2)


def probe(video: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", video],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        eprint("ERROR: ffprobe failed:", out.stderr.strip())
        sys.exit(2)
    data = json.loads(out.stdout)
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    fmt = data.get("format", {})
    duration = float(fmt.get("duration") or v.get("duration") or 0.0)
    fps = 0.0
    rate = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/0"
    try:
        n, d = rate.split("/")
        fps = float(n) / float(d) if float(d) else 0.0
    except Exception:
        pass
    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    return {
        "duration_seconds": duration,
        "duration_hms": hms(duration),
        "width": v.get("width"),
        "height": v.get("height"),
        "fps": round(fps, 3),
        "video_codec": v.get("codec_name"),
        "format": fmt.get("format_name"),
        "size_bytes": int(fmt.get("size") or 0),
        "has_audio": has_audio,
    }


def extract_audio(video: str, out_wav: str):
    out = subprocess.run(
        ["ffmpeg", "-y", "-i", video, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", out_wav],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        eprint("ERROR: audio extraction failed:", out.stderr.strip()[-800:])
        sys.exit(2)


def transcribe(wav: str, model_size: str, language, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        eprint("WARN: faster-whisper not installed; skipping transcription.")
        eprint("      pip install faster-whisper")
        return None
    eprint(f"[transcribe] loading model '{model_size}' ({compute_type})...")
    model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
    segments, info = model.transcribe(
        wav, language=language, vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    segs = []
    next_report = 50
    for s in segments:
        text = s.text.strip()
        if text:
            segs.append({
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": text,
                "avg_logprob": round(s.avg_logprob, 3) if hasattr(s, 'avg_logprob') else None,
                "no_speech_prob": round(s.no_speech_prob, 3) if hasattr(s, 'no_speech_prob') else None,
            })
        if len(segs) >= next_report:
            eprint(f"[transcribe] {len(segs)} segments... ({hms(s.end)})")
            next_report += 50
    return {"language": getattr(info, "language", language), "segments": segs}


def is_low_confidence(seg: dict) -> bool:
    lp = seg.get("avg_logprob")
    ns = seg.get("no_speech_prob")
    return (lp is not None and lp < -1.0) or (ns is not None and ns > 0.5)


def write_transcript(outdir: str, transcript: dict, meta: dict) -> dict:
    paths, segs = {}, transcript["segments"]

    md = os.path.join(outdir, "transcript.md")
    with open(md, "w") as f:
        f.write("# Transcript\n\n")
        f.write(f"- Duration: {meta['duration_hms']}\n")
        f.write(f"- Language: {transcript.get('language')}\n")
        f.write(f"- Segments: {len(segs)}\n\n")
        if any(is_low_confidence(s) for s in segs):
            f.write("**Note:** Some segments marked with ⚠️ have low confidence scores. The text may be "
                    "fabricated or misheard. Re-run with a larger `--model` (e.g., small, medium) before trusting these lines.\n\n")
        for s in segs:
            text = s['text']
            if is_low_confidence(s):
                text += " ⚠️ low-confidence"
            f.write(f"[{hms(s['start'])}] {text}\n")
    paths["transcript_md"] = md

    srt = os.path.join(outdir, "transcript.srt")
    with open(srt, "w") as f:
        for i, s in enumerate(segs, 1):
            f.write(f"{i}\n{srt_time(s['start'])} --> {srt_time(s['end'])}\n{s['text']}\n\n")
    paths["transcript_srt"] = srt

    js = os.path.join(outdir, "transcript.json")
    with open(js, "w") as f:
        json.dump(transcript, f, indent=2)
    paths["transcript_json"] = js
    return paths


def detect_scenes(video: str, threshold: float, min_scene_len: float, duration: float):
    """(start_s, end_s) list, or None to signal the interval fallback."""
    try:
        from scenedetect import detect, ContentDetector
    except ImportError:
        eprint("WARN: scenedetect not installed; falling back to interval sampling.")
        eprint("      pip install scenedetect")
        return None
    try:
        scene_list = detect(video, ContentDetector(threshold=threshold, min_scene_len=int(min_scene_len)))
    except Exception as e:
        # scenedetect imports but its opencv backend can be missing/broken —
        # observed 2026-07-23. Fall back rather than die.
        eprint(f"WARN: scene detection failed ({e}); falling back to interval sampling.")
        return None
    scenes = [(s.get_seconds(), e.get_seconds()) for s, e in scene_list]
    return scenes or None


def interval_scenes(duration: float, n: int):
    if duration <= 0 or n <= 0:
        return []
    step = duration / n
    return [(i * step, (i + 1) * step) for i in range(n)]


def pick_scenes(scenes, max_frames: int):
    """Evenly downsample to <= max_frames, keeping first and last."""
    if max_frames <= 0:
        return [], []
    if len(scenes) <= max_frames:
        return list(range(len(scenes))), scenes
    if max_frames == 1:
        return [0], [scenes[0]]
    idxs = sorted({round(i * (len(scenes) - 1) / (max_frames - 1)) for i in range(max_frames)})
    return idxs, [scenes[i] for i in idxs]


def extract_frame(video: str, ts: float, out_path: str, width: int) -> bool:
    out = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", video,
         "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3", out_path],
        capture_output=True, text=True,
    )
    return out.returncode == 0 and os.path.exists(out_path)


# --- QA mode: diff-based keyframe selection + pointer localization ----------
# A screen recording is ~90% static. Sampling every N seconds gives a pile of
# near-identical frames. Instead, sample densely, then keep only the frames that
# CHANGED from the last kept one (a block placed, a menu opened, a cable drawn).
# The same frame diff also localizes the action: the centroid of the changed
# pixels vs the previous frame is roughly where the cursor is — the one thing a
# QA review needs and raw frames never tell you.

def _np_gray(path: str, downscale_to: int = 320):
    from PIL import Image
    import numpy as np
    im = Image.open(path).convert("L")
    if downscale_to and im.width > downscale_to:
        h = round(im.height * downscale_to / im.width)
        im = im.resize((downscale_to, h))
    return np.asarray(im, dtype="int16")


def region_label(nx: float, ny: float) -> str:
    col = "left" if nx < 1 / 3 else ("right" if nx > 2 / 3 else "center")
    row = "top" if ny < 1 / 3 else ("bottom" if ny > 2 / 3 else "middle")
    return "center" if (row == "middle" and col == "center") else f"{row}-{col}"


def pointer_of(prev_gray, cur_gray, pixel_thresh: int = 28, min_pixels: int = 25):
    """Centroid of the changed region between two frames ~= where the action is.
    Returns {nx, ny, region, changed_fraction} or None when nothing moved."""
    import numpy as np
    diff = np.abs(cur_gray - prev_gray)
    mask = diff > pixel_thresh
    n = int(mask.sum())
    if n < min_pixels:
        return None
    ys, xs = np.nonzero(mask)
    nx = float(xs.mean()) / mask.shape[1]
    ny = float(ys.mean()) / mask.shape[0]
    return {"nx": round(nx, 3), "ny": round(ny, 3),
            "region": region_label(nx, ny), "changed_fraction": round(n / mask.size, 4)}


def dense_frames(video: str, fps: float, width: int, tmpdir: str):
    """One ffmpeg pass -> tmpdir/dNNNNN.jpg at `fps`. Returns sorted paths."""
    out = subprocess.run(
        ["ffmpeg", "-y", "-i", video, "-vf", f"fps={fps},scale={width}:-2",
         "-q:v", "4", os.path.join(tmpdir, "d%05d.jpg")],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        eprint("WARN: dense frame pass failed:", out.stderr.strip()[-400:])
        return []
    return sorted(glob.glob(os.path.join(tmpdir, "d*.jpg")))


def select_keyframes(files, fps: float, diff_threshold: float, max_frames: int):
    """Keep frames that changed from the last kept frame; annotate each with the
    pointer (where the change was). Returns [{src, ts, score, ptr}]."""
    import numpy as np
    if not files:
        return []
    # Only three frames are ever needed at once (current, previous, last-kept);
    # caching every decoded frame costs ~500MB on a 40-min clip at 2fps.
    grays = {}

    def g(i):
        if i not in grays:
            grays[i] = _np_gray(files[i])
        return grays[i]

    kept = [{"src": files[0], "ts": 0.0, "score": 0.0, "ptr": None}]
    last = 0
    for i in range(1, len(files)):
        score = float(np.abs(g(i) - g(last)).mean())
        if score >= diff_threshold:
            kept.append({"src": files[i], "ts": i / fps, "score": round(score, 2),
                         "ptr": pointer_of(g(i - 1), g(i))})
            last = i
        for j in [k for k in grays if k < min(last, i - 1)]:
            del grays[j]
    end = len(files) - 1
    if last != end:  # the final state matters for QA even if change was gradual
        kept.append({"src": files[end], "ts": end / fps,
                     "score": round(float(np.abs(g(end) - g(last)).mean()), 2),
                     "ptr": pointer_of(g(end - 1), g(end))})
    if len(kept) > max_frames:  # keep first, last, and the biggest changes
        mid = sorted(kept[1:-1], key=lambda k: k["score"], reverse=True)[:max_frames - 2]
        kept = [kept[0]] + sorted(mid, key=lambda k: k["ts"]) + [kept[-1]]
    return kept


# --- Enhanced output: interleaved digest, click detection, OCR, HTML, confidence ---

def digest_interleaved(outdir: str, transcript: dict, frames: list, meta: dict):
    """Emit digest.md: transcript and keyframes woven together by timestamp."""
    path = os.path.join(outdir, "digest.md")
    with open(path, "w") as f:
        f.write(f"# Digest: {meta.get('duration_hms')}\n\n")
        segs = transcript.get("segments", []) if transcript else []
        emitted_frame_indices = set()
        for i, seg in enumerate(segs):
            f.write(f"**[{hms(seg['start'])}]** {seg['text']}\n")
            for fr in frames:
                if seg["start"] <= fr["timestamp_seconds"] < seg["end"]:
                    emitted_frame_indices.add(fr["index"])
                    p = fr.get("pointer")
                    ptxt = f" — *{p['region']} ({p['nx']:.2f}×{p['ny']:.2f})*" if p else ""
                    f.write(f"![{fr['timestamp_hms']}]({fr['file']}){ptxt}\n")
            if i < len(segs) - 1:
                f.write("\n")

        # Append unmatched frames in a trailing section
        unmatched = [fr for fr in frames if fr["index"] not in emitted_frame_indices]
        if unmatched:
            if segs:
                f.write("\n---\n\n")
            f.write("## Unmatched Frames\n\n")
            f.write("Frames outside speech windows (silence gaps or after the last segment):\n\n")
            for fr in sorted(unmatched, key=lambda x: x["timestamp_seconds"]):
                p = fr.get("pointer")
                ptxt = f" — *{p['region']} ({p['nx']:.2f}×{p['ny']:.2f})*" if p else ""
                f.write(f"![{fr['timestamp_hms']}]({fr['file']}){ptxt}\n")
    return path


def detect_clicks(outdir: str, frames: list):
    """Flag frames where change is small + localized (likely a click flash)."""
    clicks = []
    for fr in frames:
        score = fr.get("change_score", 0)
        p = fr.get("pointer")
        if score < 3.0 and p:  # small, localized change
            clicks.append({"frame": fr["index"], "ts": fr["timestamp_seconds"],
                           "score": score, "region": p["region"],
                           "nx": p["nx"], "ny": p["ny"]})
    if clicks:
        path = os.path.join(outdir, "clicks.json")
        with open(path, "w") as f:
            json.dump(clicks, f, indent=2)
        return path
    return None


def emit_html_report(outdir: str, transcript: dict, frames: list, meta: dict):
    """Self-contained HTML report: transcript + frames + pointers embedded as base64."""
    path = os.path.join(outdir, "report.html")
    fps = meta.get("fps", 0)
    segs = transcript.get("segments", []) if transcript else []
    html = ["<!doctype html><html><meta charset=utf-8><style>"]
    html.append("body{font-family:sans-serif;max-width:1200px;margin:20px auto;background:#f5f5f5}")
    html.append("h1{color:#333}.seg{background:#fff;padding:20px;margin:10px 0;border-left:4px solid #0066cc}")
    html.append(".seg p{margin:0 0 10px}img{max-width:100%;height:auto;border:1px solid #ddd;margin:5px 0}")
    html.append(".pointer{font-size:0.9em;color:#666;font-style:italic}")
    html.append("</style><body><h1>Video Digest</h1>")
    emitted_frame_indices = set()
    for seg in segs:
        html.append(f"<div class=seg><p><strong>[{hms(seg['start'])}]</strong> {escape(seg['text'])}</p>")
        for fr in frames:
            if seg["start"] <= fr["timestamp_seconds"] < seg["end"]:
                emitted_frame_indices.add(fr["index"])
                p = fr.get("pointer")
                ptxt = f"<div class=pointer>{p['region']} ({p['nx']:.2f}×{p['ny']:.2f})</div>" if p else ""
                # Embed frame as base64 data: URI for true self-contained HTML
                frame_path = os.path.join(outdir, fr['file'])
                if os.path.isfile(frame_path):
                    with open(frame_path, "rb") as fh:
                        b64 = base64.b64encode(fh.read()).decode()
                        src = f"data:image/jpeg;base64,{b64}"
                else:
                    src = fr['file']  # fallback to relative path if file missing
                html.append(f"<img src=\"{src}\" alt=\"{fr['timestamp_hms']}\">{ptxt}")
        html.append("</div>")

    # Append unmatched frames in a trailing section
    unmatched = [fr for fr in frames if fr["index"] not in emitted_frame_indices]
    if unmatched:
        html.append("<div class=seg><h2>Unmatched Frames</h2>")
        html.append("<p>Frames outside speech windows (silence gaps or after the last segment):</p>")
        for fr in sorted(unmatched, key=lambda x: x["timestamp_seconds"]):
            p = fr.get("pointer")
            ptxt = f"<div class=pointer>{p['region']} ({p['nx']:.2f}×{p['ny']:.2f})</div>" if p else ""
            frame_path = os.path.join(outdir, fr['file'])
            if os.path.isfile(frame_path):
                with open(frame_path, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode()
                    src = f"data:image/jpeg;base64,{b64}"
            else:
                src = fr['file']
            html.append(f"<img src=\"{src}\" alt=\"{fr['timestamp_hms']}\">{ptxt}")
        html.append("</div>")

    html.append("</body></html>")
    with open(path, "w") as f:
        f.write("\n".join(html))
    return path


def config_path() -> str:
    return os.path.expanduser("~/.movie-digest.json")


def load_or_init_config() -> dict:
    path = config_path()
    if os.path.isfile(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def prompt_config() -> dict:
    """First-run setup: ask user for mode and report preference."""
    eprint("\n╔════════════════════════════════════════════════════════════╗")
    eprint("║         movie-digest: First-Run Configuration              ║")
    eprint("║     Built by ShreddyKrueger75 / Bloody Finger Software     ║")
    eprint("╚════════════════════════════════════════════════════════════╝\n")
    eprint("Welcome to movie-digest! This tool does four things:")
    eprint("  1. Transcribes your video's audio (what you said)")
    eprint("  2. Extracts keyframes (what changed on screen)")
    eprint("  3. Localizes pointers (where you clicked/interacted)")
    eprint("  4. Synthesizes bug findings (with --analyze)")
    eprint("")
    eprint("Output: digest.md (transcript + frames woven), report.html")
    eprint("        (self-contained review), bug_report.md (structured findings),")
    eprint("        clicks.json (detected actions)\n")

    eprint("━━━ (1) Frame selection mode ━━━\n")
    eprint("How comprehensive should frame capture be?\n")

    eprint("  INSANO — Forensic analysis")
    eprint("    • 100+ frames per 2-min clip")
    eprint("    • Captures every pixel shift, cursor twitch, tiny adjustment")
    eprint("    • Use when: Debugging UX interactions pixel-by-pixel")
    eprint("    • Example: Testing a drag-and-drop interaction")
    eprint("")

    eprint("  STRICT — Capture everything (DEFAULT)")
    eprint("    • 20–40 frames per 2-min clip")
    eprint("    • Captures every parameter tweak, every menu click, every action")
    eprint("    • Strict adherence: nothing gets missed")
    eprint("    • Use when: Deep-diving into features, detailed walkthroughs,")
    eprint("               parameter-by-parameter reviews")
    eprint("    • Example: Reviewing all EQ knob changes on an effect block")
    eprint("")

    eprint("  STANDARD — Balanced (recommended for first-time users)")
    eprint("    • 10–20 frames per 2-min clip")
    eprint("    • Catches major moments: blocks placed, menus opened, dialogs")
    eprint("    • Skips small tweaks, focuses on state changes")
    eprint("    • Use when: Most bug reports, UI reviews, typical feedback")
    eprint("    • Example: 'Show me the grid connector bug' (captures placements)")
    eprint("")

    eprint("  LENIENT — Landmark moments only")
    eprint("    • 5–10 frames per 2-min clip")
    eprint("    • Only major transitions: tab switches, big UI shifts")
    eprint("    • Use when: High-level demos, quick reviews, quick surveys")
    eprint("    • Example: Walking through the app's main navigation")
    eprint("")

    eprint("  ━━ COST ESTIMATE (Claude API input tokens) ━━")
    eprint("  For a 5-minute screen recording:\n")
    eprint("    • Transcript:  ~250–500 tokens (audio narration)")
    eprint("    • INSANO:      ~80,000 tokens (200 frames × 400 tokens/frame)")
    eprint("    • STRICT:      ~16,000 tokens (40 frames × 400 tokens/frame)")
    eprint("    • STANDARD:    ~8,000 tokens (20 frames × 400 tokens/frame)")
    eprint("    • LENIENT:     ~4,000 tokens (10 frames × 400 tokens/frame)")
    eprint("")
    eprint("  (Frames are base64-encoded images; ~400 tokens per frame.)")
    eprint("  (Use lenient for cost-sensitive analysis; strict for detail.)\n")

    eprint("  ➜ Pick one: i=insano, s=strict (default), t=standard, l=lenient")
    eprint("  ➜ Override later with: --mode insano|strict|standard|lenient\n")
    mode_choice = input("  Enter your choice [default: s]: ").strip().lower()
    mode_map = {"i": "insano", "s": "strict", "t": "standard", "l": "lenient"}
    mode = mode_map.get(mode_choice, "strict")
    eprint(f"  ✓ {mode.upper()}\n")

    eprint("━━━ (2) Generate HTML report by default? ━━━\n")
    eprint("  The report is a self-contained HTML page showing:")
    eprint("    • Transcript on the left (what you said)")
    eprint("    • Keyframes on the right (what changed)")
    eprint("    • Pointer overlay (where you clicked)")
    eprint("  Email-friendly, no external assets, shareable as-is.\n")
    eprint("  ➜ Override with: --no-report (to skip)\n")
    report_choice = input("  Generate report? [default: yes, y/n]: ").strip().lower()
    no_report = report_choice == "n"
    eprint(f"  ✓ {'SKIP' if no_report else 'GENERATE'} reports\n")

    eprint("━━━ (3) Where should reports be saved? ━━━\n")
    eprint("  Reports go to <video>.digest by default. Customize the base:")
    eprint("    • Leave blank to save next to each video (default)")
    eprint("    • Or specify a folder: ~/Documents/video-digests")
    eprint("    • Or use an absolute path: /path/to/reports\n")
    eprint("  ➜ Override per-run: --out /path/to/custom/folder\n")
    output_dir = input("  Save reports to [default: same folder as video]: ").strip()
    if output_dir:
        output_dir = os.path.expanduser(output_dir)
    eprint(f"  ✓ {'CUSTOM: ' + output_dir if output_dir else 'DEFAULT: <video>.digest'}\n")

    cfg = {"mode": mode, "no_report": no_report}
    if output_dir:
        cfg["output_dir"] = output_dir
    try:
        with open(config_path(), "w") as f:
            json.dump(cfg, f, indent=2)
        eprint(f"✓ Config saved to {config_path()}")
        eprint(f"  Reconfigure anytime: rm {config_path()}")
        eprint("  Or override per-run: --mode CHOICE --no-report --out PATH\n")
    except Exception as e:
        eprint(f"WARN: could not save config: {e}\n")
    return cfg


def analyze_digest(outdir: str, transcript: dict, model: str = "claude-haiku-4-5-20251001", frames: list = None) -> str:
    """Use Claude to synthesize bug findings from the digest."""
    if Anthropic is None:
        eprint("ERROR: --analyze requires: pip install anthropic")
        return None

    # Read the digest.md we just created
    digest_path = os.path.join(outdir, "digest.md")
    if not os.path.isfile(digest_path):
        eprint("ERROR: digest.md not found, cannot analyze")
        return None

    with open(digest_path) as f:
        digest_content = f.read()

    eprint("[analyze] calling Claude to synthesize findings...")

    client = Anthropic()

    # Build message content: text + images
    content = [{
        "type": "text",
        "text": f"""You are a QA engineer analyzing a screen recording digest.

The digest below contains a transcript (what was said) woven with keyframes
(what changed on screen). Timestamps are like [00:00:15].

Synthesize a structured bug report from this digest:
1. **Issue title** — one-line summary
2. **Steps to reproduce** — numbered, clear
3. **Expected behavior** — what should happen
4. **Actual behavior** — what actually happened
5. **Affected areas** — UI surfaces, features impacted
6. **Key timestamps** — moments that prove the issue (e.g., "[00:00:15] widget fails to update")

Keep it concise and actionable — this goes into a GitHub issue.

---

{digest_content}
"""
    }]

    # Attach up to 10 frames with highest change_score
    if frames:
        sorted_frames = sorted(
            [fr for fr in frames if fr.get("change_score") is not None],
            key=lambda x: x["change_score"],
            reverse=True
        )[:10]
        for fr in sorted_frames:
            frame_path = os.path.join(outdir, fr['file'])
            if os.path.isfile(frame_path):
                try:
                    with open(frame_path, "rb") as fh:
                        b64 = base64.b64encode(fh.read()).decode()
                    p = fr.get("pointer")
                    pointer_text = f"pointer {p['region']}" if p else "no pointer"
                    content.append({
                        "type": "text",
                        "text": f"Frame at {fr['timestamp_hms']}, {pointer_text}"
                    })
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64
                        }
                    })
                except Exception as e:
                    eprint(f"[analyze] WARN: could not attach frame {fr['file']}: {e}")

    msg = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": content}]
    )

    findings = next((b.text for b in msg.content if getattr(b, "type", None) == "text"), None)
    if not findings:
        eprint("ERROR: --analyze got no text back from the model")
        return None

    # Write findings to bug_report.md
    report_path = os.path.join(outdir, "bug_report.md")
    with open(report_path, "w") as f:
        f.write("# Bug Report (Synthesized from Screen Recording)\n\n")
        f.write(findings)
        f.write("\n\n---\n")
        f.write("*Generated by movie-digest --analyze*\n")

    eprint(f"[analyze] ✓ bug_report.md written")
    return report_path


def main():
    ap = argparse.ArgumentParser(description="Prepare a local video for Claude to digest.")
    ap.add_argument("video", help="Path to the local video file")
    ap.add_argument("--out", default=None, help="Output dir (default: <video>.digest)")
    ap.add_argument("--model", default="small",
                    help="Whisper size: tiny/base/small/medium/large-v3 (default small; tiny fabricates on long clips)")
    ap.add_argument("--language", default=None, help="Force language code (e.g. en); default auto-detect")
    ap.add_argument("--compute-type", default="int8", help="faster-whisper compute type (default int8)")
    ap.add_argument("--max-frames", type=int, default=60, help="Max keyframes to export (default 60)")
    ap.add_argument("--frame-width", type=int, default=640, help="Keyframe width in px (default 640)")
    ap.add_argument("--scene-threshold", type=float, default=27.0, help="ContentDetector threshold (default 27)")
    ap.add_argument("--min-scene-len", type=float, default=1.0, help="Min scene length, frames (default ~1s)")
    ap.add_argument("--no-dedup", action="store_true",
                    help="Disable diff-based selection + pointer (fall back to scene/interval sampling)")
    ap.add_argument("--sample-fps", type=float, default=2.0,
                    help="Dense sample rate for diff mode (default 2/s)")
    ap.add_argument("--mode", choices=["insano", "strict", "standard", "lenient"], default=None,
                    help="Diff threshold preset: insano (100+), strict (20–40), standard (10–20), lenient (5–10) per 2-min clip (default standard, or saved config)")
    ap.add_argument("--diff-threshold", type=float, default=None,
                    help="Mean gray delta to count a frame as changed (overrides --mode; lower = more frames)")
    ap.add_argument("--no-transcribe", action="store_true", help="Skip transcription")
    ap.add_argument("--no-frames", action="store_true", help="Skip frame extraction")
    ap.add_argument("--no-report", action="store_true", help="Skip HTML report generation")
    ap.add_argument("--report", action="store_true",
                    help="Force HTML report generation (overrides a saved no_report config)")
    ap.add_argument("--analyze", action="store_true", help="Synthesize a structured bug report from the digest (requires ANTHROPIC_API_KEY)")
    ap.add_argument("--analyze-model", default="claude-haiku-4-5-20251001", help="Claude model for --analyze (default claude-haiku-4-5-20251001)")
    ap.add_argument("--json", action="store_true", help="Output JSON summary instead of human-readable text")
    ap.add_argument("--keep-audio", action="store_true", help="Keep the extracted wav")
    args = ap.parse_args()

    # Load or initialize config
    cfg = load_or_init_config()
    if not cfg:
        if sys.stdin.isatty():
            cfg = prompt_config()
        else:
            # Non-interactive: use defaults and write to config file
            cfg = {"mode": "standard", "no_report": False}
            try:
                os.makedirs(os.path.dirname(config_path()), exist_ok=True)
                with open(config_path(), "w") as f:
                    json.dump(cfg, f, indent=2)
                eprint(f"[config] Using defaults (saved to {config_path()})")
                eprint(f"[config] Reconfigure: delete the file and rerun interactively")
            except Exception as e:
                eprint(f"[config] WARN: could not save defaults: {e}")
                eprint(f"[config] Continuing with in-memory defaults")

    # Precedence: explicit CLI flag > saved config > built-in default.
    if args.mode is None:
        args.mode = cfg.get("mode", "standard")
    if args.report:
        args.no_report = False
    elif not args.no_report:
        args.no_report = cfg.get("no_report", False)

    # Map mode to diff_threshold unless explicitly overridden
    if args.diff_threshold is None:
        mode_thresholds = {"insano": 0.2, "strict": 0.8, "standard": 1.5, "lenient": 2.5}
        args.diff_threshold = mode_thresholds.get(args.mode, 1.5)

    require_tool("ffmpeg")
    require_tool("ffprobe")

    video = os.path.abspath(args.video)
    if not os.path.isfile(video):
        eprint(f"ERROR: file not found: {video}")
        # The most common cause here is a filename with a narrow no-break space
        # (U+202F) before "PM" in macOS screen-recording names — a literal path
        # copied from a message won't match. Resolve with a glob or copy first.
        eprint("       If this is a macOS 'Screen Recording ... PM.mov', the space before")
        eprint("       'PM' may be U+202F. Resolve with a glob, e.g.:")
        eprint("         f=$(ls *Recording*2.18*.mov); python3 digest_movie.py \"$f\"")
        sys.exit(2)

    # Determine output directory: CLI arg > config default > <video>.digest
    if args.out:
        outdir = os.path.abspath(args.out)
    elif "output_dir" in cfg and cfg["output_dir"]:
        # Use config's output_dir but create a subdir for each video
        base_dir = cfg["output_dir"]
        os.makedirs(base_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video))[0]
        outdir = os.path.join(base_dir, video_name + ".digest")
    else:
        outdir = video + ".digest"

    frames_dir = os.path.join(outdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    eprint(f"[probe] {video}")
    meta = probe(video)
    eprint(f"[probe] {meta['duration_hms']}  {meta['width']}x{meta['height']}  "
           f"{meta['fps']}fps  audio={'yes' if meta['has_audio'] else 'NO'}")

    # Warn if using tiny model on long clips
    if args.model == "tiny" and meta["duration_seconds"] > 180:
        eprint(f"WARN: 'tiny' on a {meta['duration_hms']} recording — this model fabricates plausible-sounding")
        eprint(f"      text on long or sparsely-narrated clips. Re-run with --model small if the")
        eprint(f"      transcript reads like nonsense.")

    manifest = {
        "video": video, "output_dir": outdir, "metadata": meta,
        "params": {"model": args.model, "language": args.language,
                   "max_frames": args.max_frames, "frame_width": args.frame_width,
                   "scene_threshold": args.scene_threshold, "mode": args.mode,
                   "diff_threshold": args.diff_threshold, "sample_fps": args.sample_fps,
                   "dedup": not args.no_dedup},
        "transcript": None, "frames": [], "scene_source": None,
    }

    # --- transcription ---
    if not args.no_transcribe:
        if not meta["has_audio"]:
            eprint("[audio] no audio track — skipping transcription (frames only).")
        else:
            tmpdir = tempfile.mkdtemp(prefix="digest_")
            wav = os.path.join(outdir if args.keep_audio else tmpdir, "audio.wav")
            eprint("[audio] extracting...")
            extract_audio(video, wav)
            transcript = transcribe(wav, args.model, args.language, args.compute_type)
            if transcript is not None:
                paths = write_transcript(outdir, transcript, meta)
                manifest["transcript"] = {"language": transcript.get("language"),
                                          "segment_count": len(transcript["segments"]), **paths}
                n = len(transcript["segments"])
                eprint(f"[transcribe] done: {n} segments"
                       + ("  (no speech detected)" if n == 0 else ""))
            shutil.rmtree(tmpdir, ignore_errors=True)

    # --- frames ---
    if not args.no_frames:
        dedup = not args.no_dedup
        if dedup:
            try:
                import PIL  # noqa: F401
                import numpy  # noqa: F401
            except ImportError:
                eprint("WARN: Pillow/numpy not installed; diff mode off (pip install Pillow numpy).")
                dedup = False

        frames = []
        if dedup:
            manifest["scene_source"] = "diff"
            tmpdir = tempfile.mkdtemp(prefix="digest_frames_")
            try:
                eprint(f"[frames] dense sampling @ {args.sample_fps}/s for diff selection...")
                files = dense_frames(video, args.sample_fps, args.frame_width, tmpdir)
                kept = select_keyframes(files, args.sample_fps, args.diff_threshold, args.max_frames)
                eprint(f"[frames] {len(files)} sampled -> {len(kept)} kept (changed frames only)")
                for out_i, k in enumerate(kept):
                    ts = k["ts"]
                    fname = f"{out_i:04d}_{hms(ts, compact=True)}.jpg"
                    shutil.copyfile(k["src"], os.path.join(frames_dir, fname))
                    frames.append({"index": out_i, "file": os.path.join("frames", fname),
                                   "timestamp_seconds": round(ts, 3), "timestamp_hms": hms(ts),
                                   "change_score": k["score"], "pointer": k["ptr"]})
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            scenes = detect_scenes(video, args.scene_threshold, args.min_scene_len, meta["duration_seconds"])
            manifest["scene_source"] = "scenedetect" if scenes else "interval"
            if not scenes:
                scenes = interval_scenes(meta["duration_seconds"], args.max_frames)
            manifest["scene_count_detected"] = len(scenes)
            idxs, picked = pick_scenes(scenes, args.max_frames)
            eprint(f"[frames] {len(scenes)} scenes -> exporting {len(picked)} keyframes ({manifest['scene_source']})")
            for out_i, (scene_i, (start, end)) in enumerate(zip(idxs, picked)):
                mid = start + (end - start) / 2.0 if end > start else start
                fname = f"{out_i:04d}_{hms(mid, compact=True)}.jpg"
                if extract_frame(video, mid, os.path.join(frames_dir, fname), args.frame_width):
                    frames.append({"index": out_i, "file": os.path.join("frames", fname),
                                   "timestamp_seconds": round(mid, 3), "timestamp_hms": hms(mid),
                                   "scene_start": round(start, 3), "scene_end": round(end, 3)})
                if (out_i + 1) % 20 == 0:
                    eprint(f"[frames] {out_i + 1}/{len(picked)}...")

        manifest["frames"] = frames

        fi = os.path.join(outdir, "frames_index.md")
        with open(fi, "w") as f:
            f.write("# Keyframes\n\n")
            f.write(f"{len(frames)} frames from {meta['duration_hms']} ({manifest['scene_source']} sampling)\n\n")
            if manifest["scene_source"] == "diff":
                f.write("`pointer` = where the screen changed vs the previous frame (~cursor/action). "
                        "`change` = how much changed.\n\n")
                f.write("| # | timestamp | change | pointer | file |\n|---|---|---|---|---|\n")
                for fr in frames:
                    p = fr.get("pointer")
                    ptxt = f"{p['region']} ({p['nx']:.2f},{p['ny']:.2f})" if p else "—"
                    f.write(f"| {fr['index']} | {fr['timestamp_hms']} | {fr['change_score']} | {ptxt} | {fr['file']} |\n")
            else:
                f.write("| # | timestamp | file |\n|---|-----------|------|\n")
                for fr in frames:
                    f.write(f"| {fr['index']} | {fr['timestamp_hms']} | {fr['file']} |\n")
        manifest["frames_index_md"] = fi
        eprint(f"[frames] done: {len(frames)} exported")

    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # --- enhanced outputs ---
    outputs = {"manifest": os.path.join(outdir, "manifest.json")}
    tr = None
    if manifest.get("transcript"):
        try:
            with open(manifest["transcript"].get("transcript_json", ""), "r") as f:
                tr = json.load(f)
        except Exception:
            eprint("WARN: could not load transcript.json")

    if manifest["frames"]:
        try:
            if tr:
                eprint("[output] digest.md (interleaved)...")
                outputs["digest"] = digest_interleaved(outdir, tr, manifest["frames"], meta)
            if not args.no_report:
                eprint("[output] HTML report...")
                outputs["html"] = emit_html_report(outdir, tr or {"segments": []}, manifest["frames"], meta)
            if manifest.get("scene_source") == "diff":
                eprint("[output] clicks.json...")
                clicks = detect_clicks(outdir, manifest["frames"])
                if clicks:
                    outputs["clicks"] = clicks
        except Exception as e:
            eprint(f"WARN: enhanced output failed: {e}")

    # --- analyze for bug findings (optional) ---
    if args.analyze:
        try:
            if tr:
                result = analyze_digest(outdir, tr, model=args.analyze_model, frames=manifest.get("frames"))
                if result:
                    outputs["bug_report"] = result
            else:
                eprint("WARN: --analyze requires transcript (no audio or --no-transcribe)")
        except Exception as e:
            eprint(f"WARN: --analyze failed: {e}")
            eprint("      Set ANTHROPIC_API_KEY environment variable to use this feature")

    if args.json:
        # Output JSON summary
        json_output = {
            "output_dir": outdir,
            "manifest": os.path.join(outdir, "manifest.json"),
            "outputs": {k: v for k, v in outputs.items() if k != "manifest"}
        }
        print(json.dumps(json_output, indent=2))
    else:
        # Human-readable output
        print("\n=== DIGEST PREP COMPLETE ===")
        print(f"output_dir: {outdir}")
        print(f"duration:   {meta['duration_hms']}")
        if manifest["transcript"]:
            print(f"transcript: {manifest['transcript']['segment_count']} segments -> transcript.md")
        else:
            print("transcript: (none — silent clip or --no-transcribe)")
        src = manifest.get("scene_source")
        ptr = " (frames_index.md has the pointer/change columns)" if src == "diff" else ""
        print(f"frames:     {len(manifest['frames'])} -> frames/  ({src} sampling){ptr}")
        for key, path in outputs.items():
            if key != "manifest":
                print(f"{key:12} {os.path.basename(path)}")
        print(f"\nKey outputs:")
        print(f"  • digest.md — transcript + keyframes woven by time")
        print(f"  • report.html — self-contained review document")
        print(f"  • frames_index.md — pointer + change scores for each frame")
        print(f"  • clicks.json — suspected click/action moments (diff mode only)")


if __name__ == "__main__":
    main()
