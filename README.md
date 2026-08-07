# movie-digest

A [Claude Code](https://claude.com/claude-code) **skill** that lets Claude
**watch and digest a local video file** — by transcribing its audio and
exporting keyframes, so the model can actually read what happens.

Claude can't decode video or hear audio. This skill splits the work:

- `scripts/digest_movie.py` transcribes the spoken dialogue with timestamps
  (faster-whisper) and exports keyframes.
- **Optimized for QA / bug-report recordings.** By default it keeps only the
  frames that *changed* (diff-based selection — no more 20 duplicate frames of a
  static screen) and localizes the **pointer** on each: where the screen changed
  vs the previous frame ≈ where the cursor / action was. A failure often shows as
  the *absence* of change. `--no-dedup` restores plain scene/interval sampling.
- Claude then reads `transcript.md` and views the frames to write a structured
  digest.

**The audio is the point.** A narrated screen recording — someone talking
through a bug or reviewing a UI — is nearly useless as frames alone; the actual
report is in the voice. Transcribe first, read the transcript, *then* look at
the frames.

## Install as a skill

Drop the folder into your Claude Code skills directory:

```bash
cp -R movie-digest ~/.claude/skills/movie-digest
# or, per-project:  cp -R movie-digest <project>/.claude/skills/movie-digest
```

Install the runtime deps once:

```bash
brew install ffmpeg
pip install Pillow numpy faster-whisper scenedetect
```

Then tell Claude Code: **"digest this video: /path/to/clip.mov"** — or just hand
it a screen recording and ask what happens.

## Run the script directly

```bash
python3 scripts/digest_movie.py "/path/to/clip.mov" \
  --out "/path/to/clip.digest" --model small --max-frames 25 --analyze
```

- `--model tiny|base|small|medium|large-v3` — accuracy vs. speed. `tiny` only safe
  for short (<~2 min) continuously-narrated clips; fabricates text otherwise.
  `small` is the safe default; bigger models slower but more accurate.
- `--analyze-model MODEL` — Claude model used by `--analyze` (default claude-haiku-4-5-20251001).
- `--json` — machine-readable summary: JSON object with output_dir/manifest/outputs instead of human summary.
- `--max-frames N` — keyframe cap (default 60).
- `--analyze` — synthesize a structured bug report (needs `ANTHROPIC_API_KEY` env var).
- `--no-frames` — transcript only, fast.
- `--no-transcribe` — frames only (silent footage).
- `--report` / `--no-report` — force the HTML report on/off, overriding the saved config.

See `SKILL.md` for the full agent workflow, all flags, and the gotchas
(macOS narrow-space filenames, `tiny` mishearings, the scenedetect/OpenCV
fallback).

## What you get

Every digest includes:

**Primary outputs** (the QA workflow):
- **digest.md** — transcript + keyframes woven by time. Read top-to-bottom; pointer + region marked inline on each frame. Includes "Unmatched frames" section for silent gaps and frames outside transcript segments.
- **report.html** — self-contained HTML review (email-friendly, no external deps). Transcript on left, keyframes on right, pointer overlay. Shareable as-is. Also shows unmatched frames.
- **transcript.md** — timestamped narration. Segments marked with `⚠️ low-confidence` may be misheard (low Whisper confidence); re-check those with a larger `--model` before quoting.
- **bug_report.md** (with `--analyze`) — Claude-synthesized structured bug report: issue title, repro steps, expected/actual, affected areas, key timestamps. Ready to paste into GitHub.
- **clicks.json** — detected click/action moments (small, localized changes). JSON list with frame index, estimated timestamp, and pointer region for each suspected interaction.

**Reference outputs**:
- **frames_index.md** — pointer + change-score table for every kept frame. Glanceable; tells you where to look before opening an image.
- **transcript files** — `.md`, `.srt`, `.json` for grepping, remixing, or feeding to other tools.
- **manifest.json** — metadata, frame list with pointers, and transcript paths.

## Modes — diff-threshold presets

`--mode insano|strict|standard|lenient` — tune how comprehensive frame capture is.

- **`insano`** — every single change. 100+ frames per 2-min clip. For forensic analysis of every interaction, cursor twitch, pixel shift.
- **`strict`** — capture everything. 20–40 frames per 2-min clip. Strict adherence to what you said/did; nothing missed. For detailed walkthroughs and parameter-level specs.
- **`standard`** (default) — balanced. 10–20 frames per 2-min clip. Catches blocks placed, menus opened, dialogs appeared.
- **`lenient`** — landmark moments only. 5–10 frames per 2-min clip. For high-level demos or quick reviews where you only need major state shifts.

All modes keep the pointer and change-score columns, digest.md, and report.html. Only the number of kept frames changes.

## Requirements

- `ffmpeg` + `ffprobe` on PATH — required.
- `faster-whisper` — optional (no transcript without it).
- `scenedetect` — optional (interval frame sampling without it).

## Notes

- **Local files only.** DRM-protected streaming (Netflix, etc.) can't be
  captured.
- Silent footage → `transcript.md` has 0 segments (expected); the digest leans
  on the frames.
- Cost and time scale with `--max-frames` and Whisper `--model` size.

## Development

```bash
pip install pytest
python3 -m pytest tests/
```

The tests cover the pure logic (timestamp formatting, frame selection, pointer
math, click detection, markdown emitters) — no ffmpeg or Whisper needed.

## License

MIT — see `LICENSE`.

Built by Bloody Finger Software.
