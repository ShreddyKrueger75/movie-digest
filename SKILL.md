---
name: movie-digest
description: >-
  Watch and digest a local video file — transcribe its narration and export
  keyframes so Claude can actually read what happens. Use whenever someone
  points at a video on disk (.mp4, .mkv, .mov, .webm, .avi) and wants it
  watched, summarized, broken down, analyzed, or when a screen recording
  narrates a bug/feature/review out loud: "digest this movie", "watch this",
  "what's in this footage", "transcribe and break down this clip", or any
  bug-report / UI-review screen recording. Local files only.
---

# movie-digest

Claude can't decode video or hear audio. This skill splits the job: a script
does the mechanical part — transcribes the spoken audio with timestamps and
exports one keyframe per scene — and then **Claude reads the transcript and
views the frames** to write the digest.

**The audio is the point.** A narrated screen recording (someone talking through
a bug) is nearly useless as frames alone — the actual report is in the voice.
Transcribe first, read `transcript.md`, *then* look at the frames. Skipping the
transcript and pulling frames with raw `ffmpeg` is the #1 way to miss the whole
message.

## Prerequisites

```bash
brew install ffmpeg
pip install faster-whisper scenedetect
```

`ffmpeg`/`ffprobe` are required. `faster-whisper` and `scenedetect` are
optional and auto-detected — without the first it skips the transcript; without
the second it samples frames at even time intervals (fine for screen
recordings).

## Setup (first run)

On first run, the script will prompt for your defaults:

```
(1) Diff-threshold mode — how aggressive frame selection is
(2) Generate HTML report by default?
```

Your choices save to `~/.movie-digest.json` — reconfigure anytime by deleting that file.

## Run (the one path)

```bash
python3 scripts/digest_movie.py "/path/to/CLIP.mov" \
  --out "/path/to/CLIP.digest" --model tiny --max-frames 25
```

Then **read `<out>/transcript.md` first**, then view the frames listed in
`<out>/frames_index.md` in batches of ~10–15. Anchor every claim to a timestamp.

Flags that matter:
- `--mode insano|strict|standard|lenient` — frame selection comprehensiveness (saves to config; default standard).
- `--no-report` — skip HTML report (frames + digest.md only).
- `--analyze` — synthesize a structured bug report from the digest (requires `ANTHROPIC_API_KEY`).
- `--model tiny|base|small|medium|large-v3` — `tiny` for a quick screen
  recording, `base`/`small` for a real film. Bigger = more accurate, slower.
- `--max-frames N` — cap on keyframes (default 60; 12–20 for a short clip).
- `--diff-threshold N` — override mode's threshold (lower = more frames, default 1.5 for standard).
- `--no-dedup` — turn OFF diff selection + pointer; use plain interval/scene
  sampling instead.
- `--no-frames` — transcript only, fast. Use when you only need the narration.
- `--no-transcribe` — frames only (silent footage).
- `--language en` — skip auto-detect.

Outputs under `--out`:

```
transcript.md      timestamped, grep-friendly   <- READ FIRST
transcript.srt     subtitles
transcript.json    raw {start,end,text} segments
frames/            NNNN_HHhMMmSSs.jpg keyframes (640px)
frames_index.md    frame -> timestamp + change score + POINTER table
digest.md          transcript + keyframes woven by time
report.html        self-contained HTML review document
clicks.json        suspected click/flash moments (diff mode)
manifest.json      metadata + full frame list (+ pointer) + transcript paths
```

## QA mode (default): changed frames only + pointer

This is built for **bug-report / UI-review screen recordings**, so by default it
does two things a plain frame-dump can't:

- **Diff-based selection.** A screen recording is ~90% static. Instead of one
  frame every N seconds (a pile of duplicates), it samples densely, then keeps
  only the frames that *changed* from the last kept one — a block placed, a menu
  opened, a cable drawn. 169 sampled → ~12 meaningful.
- **Pointer localization.** Each kept frame's `pointer` column is the centroid of
  what changed vs the previous frame ≈ **where the cursor / action was**, as a
  region (`top-right`, `center`, …) + normalized `(x,y)`. `change` is the
  magnitude — a big number is a new screen/dialog; a small one is a local edit.

Read `frames_index.md` and let the pointer + change columns tell you *where to
look* in each frame before you open it. A **failure often shows as the absence of
change** — the user says "wire it across" and the next frames don't change: that
gap IS the bug. Needs Pillow + numpy (auto-detected; falls back to interval if
missing). `--no-dedup` restores plain sampling.

## Enhanced outputs

Every digest now includes:

- **digest.md** — the quick read: transcript segments woven together with
  keyframes that fall within each segment's time window. Pointer/region marked
  inline. One document = one bug report.
- **report.html** — self-contained (no external assets): transcript on the left,
  keyframes on the right, pointer overlay. Shareable, no post-processing needed.
- **clicks.json** — detects small, localized changes (likely click flashes or
  menu appearances). Frames tagged with the suspected action moment. For QA mode
  only; empty if no candidate clicks found.
- **bug_report.md** (with `--analyze`) — Claude synthesizes a structured bug report:
  issue title, steps to reproduce, expected/actual behavior, affected areas, and
  key timestamps. Requires `ANTHROPIC_API_KEY` environment variable.

The first three are auto-generated and graceful when transcript is absent (frames only).
Bug report generation is opt-in via `--analyze`.

## Gotchas (learned the hard way)

- **macOS screen-recording filenames contain a narrow no-break space (U+202F)**
  before "PM" — `Screen Recording 2026-07-23 at 9.40.00 PM.mov`. A literal path
  copied from a message will NOT match on the command line. Resolve with a glob,
  or copy to a space-free path first:
  ```bash
  f=$(ls *Recording*9.40*.mov); cp "$f" /tmp/clip.mov
  python3 scripts/digest_movie.py /tmp/clip.mov --out /tmp/clip.digest --model tiny
  ```
  The script prints this exact hint if it can't find the file.
- **`--model tiny` mishears a word or two** — it heard "tempo" as "VPN" and
  "chorus" fine but garbled a product name once. Cross-check any load-bearing
  term against the matching frame before quoting it as fact.
- **`scenedetect` can import but fail** if its OpenCV backend is missing/broken.
  The script catches that and falls back to interval sampling automatically —
  you'll see `WARN: scene detection failed ... falling back`. Not an error.
- **Transcription is the slow part.** On Apple Silicon `tiny`/`base` run faster
  than real time; `large-v3` is much slower. For a 1–2 min screen recording,
  `tiny --no-frames` returns in seconds.
- **Silent clip → 0 segments.** Expected; lean on the frames.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ERROR: file not found` on a path that exists | Filename has a U+202F space — glob it (see Gotchas). |
| `WARN: faster-whisper not installed` | `pip install faster-whisper` — transcript was skipped. |
| `WARN: scenedetect not installed` / `scene detection failed` | Harmless; frames sampled at intervals instead. |
| 0 segments on a clip you know has talking | Wrong `--language`, or the audio track is silent/very quiet — try `--model base` and confirm `audio=yes` in the `[probe]` line. |
