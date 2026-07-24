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
  --out "/path/to/clip.digest" --model tiny --max-frames 25
```

- `--model tiny|base|small|medium|large-v3` — accuracy vs. speed. `tiny` for a
  quick screen recording; `base`/`small` for a real film.
- `--max-frames N` — keyframe cap (default 60).
- `--no-frames` — transcript only, fast.
- `--no-transcribe` — frames only (silent footage).

See `SKILL.md` for the full agent workflow, all flags, and the gotchas
(macOS narrow-space filenames, `tiny` mishearings, the scenedetect/OpenCV
fallback).

## What you get

Every digest includes:

1. **digest.md** — transcript + keyframes woven by time (the 1-document read)
2. **report.html** — self-contained HTML review (email-friendly, no deps)
3. **clicks.json** — detected click/action moments (QA mode only)
4. **frames_index.md** — pointer + change-score table for each frame
5. **transcript files** — `.md`, `.srt`, `.json` for grepping + remixing

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

## License

MIT — see `LICENSE`.

Built by Bloody Finger Software.
