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
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile


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
    for s in segments:
        text = s.text.strip()
        if text:
            segs.append({"start": round(s.start, 3), "end": round(s.end, 3), "text": text})
        if segs and len(segs) % 50 == 0:
            eprint(f"[transcribe] {len(segs)} segments... ({hms(s.end)})")
    return {"language": getattr(info, "language", language), "segments": segs}


def write_transcript(outdir: str, transcript: dict, meta: dict) -> dict:
    paths, segs = {}, transcript["segments"]

    md = os.path.join(outdir, "transcript.md")
    with open(md, "w") as f:
        f.write("# Transcript\n\n")
        f.write(f"- Duration: {meta['duration_hms']}\n")
        f.write(f"- Language: {transcript.get('language')}\n")
        f.write(f"- Segments: {len(segs)}\n\n")
        for s in segs:
            f.write(f"[{hms(s['start'])}] {s['text']}\n")
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
    if len(scenes) <= max_frames:
        return list(range(len(scenes))), scenes
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
    grays = [None] * len(files)

    def g(i):
        if grays[i] is None:
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
    end = len(files) - 1
    if last != end:  # the final state matters for QA even if change was gradual
        kept.append({"src": files[end], "ts": end / fps,
                     "score": round(float(np.abs(g(end) - g(last)).mean()), 2),
                     "ptr": pointer_of(g(end - 1), g(end))})
    if len(kept) > max_frames:  # keep first, last, and the biggest changes
        mid = sorted(kept[1:-1], key=lambda k: k["score"], reverse=True)[:max_frames - 2]
        kept = [kept[0]] + sorted(mid, key=lambda k: k["ts"]) + [kept[-1]]
    return kept


def main():
    ap = argparse.ArgumentParser(description="Prepare a local video for Claude to digest.")
    ap.add_argument("video", help="Path to the local video file")
    ap.add_argument("--out", default=None, help="Output dir (default: <video>.digest)")
    ap.add_argument("--model", default="base",
                    help="Whisper size: tiny/base/small/medium/large-v3 (default base; tiny for quick screen-recordings)")
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
    ap.add_argument("--diff-threshold", type=float, default=1.5,
                    help="Mean gray delta to count a frame as changed (default 1.5; lower = more frames)")
    ap.add_argument("--no-transcribe", action="store_true", help="Skip transcription")
    ap.add_argument("--no-frames", action="store_true", help="Skip frame extraction")
    ap.add_argument("--keep-audio", action="store_true", help="Keep the extracted wav")
    args = ap.parse_args()

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

    outdir = os.path.abspath(args.out) if args.out else video + ".digest"
    frames_dir = os.path.join(outdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    eprint(f"[probe] {video}")
    meta = probe(video)
    eprint(f"[probe] {meta['duration_hms']}  {meta['width']}x{meta['height']}  "
           f"{meta['fps']}fps  audio={'yes' if meta['has_audio'] else 'NO'}")

    manifest = {
        "video": video, "output_dir": outdir, "metadata": meta,
        "params": {"model": args.model, "language": args.language,
                   "max_frames": args.max_frames, "frame_width": args.frame_width,
                   "scene_threshold": args.scene_threshold},
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

    print("\n=== DIGEST PREP COMPLETE ===")
    print(f"output_dir: {outdir}")
    print(f"duration:   {meta['duration_hms']}")
    if manifest["transcript"]:
        print(f"transcript: {manifest['transcript']['segment_count']} segments -> transcript.md  <-- READ THIS FIRST")
    else:
        print("transcript: (none — silent clip or --no-transcribe)")
    src = manifest.get("scene_source")
    ptr = " (frames_index.md has the pointer/change columns)" if src == "diff" else ""
    print(f"frames:     {len(manifest['frames'])} -> frames/  ({src} sampling){ptr}")
    print(f"manifest:   {os.path.join(outdir, 'manifest.json')}")
    print("\nNext: read transcript.md, THEN view the frames. The voice is the brief.")


if __name__ == "__main__":
    main()
