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

## Run (the one path)

```bash
python3 scripts/digest_movie.py "/path/to/CLIP.mov" \
  --out "/path/to/CLIP.digest" --model tiny --max-frames 25
```

Then **read `<out>/transcript.md` first**, then view the frames listed in
`<out>/frames_index.md` in batches of ~10–15. Anchor every claim to a timestamp.

Flags that matter:
- `--model tiny|base|small|medium|large-v3` — `tiny` for a quick screen
  recording, `base`/`small` for a real film. Bigger = more accurate, slower.
- `--max-frames N` — cap on keyframes (default 60; 20–30 for a short clip).
- `--no-frames` — transcript only, fast. Use when you only need the narration.
- `--no-transcribe` — frames only (silent footage).
- `--language en` — skip auto-detect.

Outputs under `--out`:

```
transcript.md      timestamped, grep-friendly   <- READ FIRST
transcript.srt     subtitles
transcript.json    raw {start,end,text} segments
frames/            NNNN_HHhMMmSSs.jpg keyframes (640px)
frames_index.md    frame -> timestamp table
manifest.json      metadata + full frame list + transcript paths
```

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
