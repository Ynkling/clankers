#!/usr/bin/env python3
"""Render and assemble the explainer video.

    python build.py render   [--quality low|med|high] [--only s03_task,s04_gate] [--jobs 2]
    python build.py frames   SCENE_CLASS [--quality low] [--every 4]     # PNG contact sheet for review
    python build.py assemble [--quality high]                          # one mp4 + srt + chapters
    python build.py list

Rendering needs ManimGL (github.com/3b1b/manim), a TeX installation, ffmpeg, and an X
display; on a headless machine the script wraps manimgl in xvfb-run. Narration is
synthesized on first render by Kokoro (see common/narration.py) and cached in .tts_cache/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scenes.manifest import SCENES  # noqa: E402  (file stem, Scene class, chapter title)

QUALITY_FLAGS = {"low": ["-l"], "med": ["-m"], "high": ["--hd"]}
BUILD = HERE / "build"
TITLE = "Multi-Channel Hebbian Plasticity in Multilayer BDH, explained"


def video_dir(quality: str) -> Path:
    return BUILD / "videos" / quality


def scene_path(quality: str, cls: str) -> Path:
    return video_dir(quality) / f"{cls}.mp4"


def render_one(stem: str, cls: str, quality: str) -> tuple[str, int, str]:
    cmd = ["manimgl", f"scenes/{stem}.py", cls, "-w", *QUALITY_FLAGS[quality],
           "--video_dir", str(video_dir(quality)), "--file_name", cls]
    if shutil.which("xvfb-run") and not os.environ.get("DISPLAY"):
        cmd = ["xvfb-run", "-a", "-s", "-screen 0 1920x1080x24"] + cmd
    env = dict(os.environ, PYTHONPATH=str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    log = BUILD / "logs" / f"{cls}.{quality}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, cwd=HERE, env=env, stdout=fh, stderr=subprocess.STDOUT)
        # concurrent renders can race for an X display number; retry those, nothing else
        if rc == 0 or "Xvfb failed" not in log.read_text(errors="ignore"):
            break
        time.sleep(2 + 3 * attempt)
    return cls, rc, str(log)


def cmd_render(args) -> int:
    only = set(args.only.split(",")) if args.only else None
    todo = [(s, c) for s, c, _ in SCENES if not only or s in only or c in only]
    video_dir(args.quality).mkdir(parents=True, exist_ok=True)
    failed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for cls, rc, log in ex.map(lambda sc: render_one(sc[0], sc[1], args.quality), todo):
            ok = rc == 0 and scene_path(args.quality, cls).exists()
            print(f"{'ok  ' if ok else 'FAIL'} {cls}  ({log})", flush=True)
            if not ok:
                failed.append(cls)
    return 1 if failed else 0


def probe_duration(path: Path) -> float:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                   "-of", "default=nw=1:nk=1", str(path)])
    return float(out.strip())


def probe_stream_duration(path: Path, kind: str) -> float:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", f"{kind}:0", "-show_entries",
                                   "stream=duration", "-of", "default=nw=1:nk=1", str(path)])
    return float(out.strip())


def has_audio(path: Path) -> bool:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                                   "stream=index", "-of", "csv=p=0", str(path)])
    return bool(out.strip())


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def wrap_cue(text: str, width: int = 64) -> list[str]:
    """Split a long sentence into cue-sized pieces of at most two lines."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return ["\n".join(lines[i:i + 2]) for i in range(0, len(lines), 2)]


def cmd_assemble(args) -> int:
    q = args.quality
    out_dir = BUILD / "final" / q
    out_dir.mkdir(parents=True, exist_ok=True)
    parts, missing = [], []
    for stem, cls, title in SCENES:
        p = scene_path(q, cls)
        (parts if p.exists() else missing).append((p, cls, title))
    if missing:
        print("missing renders:", ", ".join(c for _, c, _ in missing))
        if not args.allow_missing:
            return 1

    # Each part's audio track ends at its last narration, before the video does; pad every
    # part's audio to its exact video length so the concatenation cannot drift out of sync.
    padded_dir = out_dir / "parts"
    padded_dir.mkdir(exist_ok=True)
    padded = []
    for p, cls, _ in parts:
        vdur = probe_stream_duration(p, "v")
        out = padded_dir / f"{cls}.mp4"
        if has_audio(p):
            src = ["-i", str(p)]
            amap = ["-map", "0:a"]
        else:
            src = ["-i", str(p), "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono"]
            amap = ["-map", "1:a"]
        subprocess.check_call(["ffmpeg", "-v", "error", "-y", *src, "-map", "0:v", *amap, "-c:v", "copy",
                               "-af", "aresample=48000,apad", "-t", f"{vdur:.3f}", "-c:a", "pcm_s16le",
                               "-ar", "48000", "-ac", "1", str(out.with_suffix(".mov"))])
        padded.append(out.with_suffix(".mov"))
    concat = out_dir / "concat.txt"
    concat.write_text("".join(f"file '{p}'\n" for p in padded))
    raw = out_dir / "raw.mp4"
    subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
                           "-c:v", "copy", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                           "-c:a", "aac", "-b:a", "96k", "-ac", "1", "-ar", "48000", str(raw)])

    # subtitles and chapters, offset by each part's real duration
    cues, chapters, t0 = [], [], 0.0
    for p, cls, title in parts:
        dur = probe_stream_duration(p, "v")
        chapters.append((t0, t0 + dur, title))
        subs = BUILD / "subs" / f"{cls}.json"
        if subs.exists():
            for c in json.loads(subs.read_text())["cues"]:
                pieces = wrap_cue(c["text"])
                span = (c["end"] - c["start"]) / len(pieces)
                for i, piece in enumerate(pieces):
                    a = t0 + c["start"] + i * span
                    cues.append((a, min(a + span, t0 + dur), piece))
        t0 += dur
    srt = out_dir / "clankers_explained.srt"
    srt.write_text("".join(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{txt}\n\n"
                           for i, (a, b, txt) in enumerate(cues, 1)))
    meta = out_dir / "chapters.txt"
    meta.write_text(";FFMETADATA1\ntitle=" + TITLE + "\n" + "".join(
        f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={int(a * 1000)}\nEND={int(b * 1000)}\ntitle={t}\n"
        for a, b, t in chapters))
    final = out_dir / "clankers_explained.mp4"
    subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-i", str(srt), "-i", str(meta),
                           "-map", "0:v", "-map", "0:a", "-map", "1:s", "-map_metadata", "2",
                           "-map_chapters", "2", "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text",
                           "-metadata:s:s:0", "language=eng", "-movflags", "+faststart", str(final)])
    raw.unlink()
    shutil.rmtree(padded_dir)
    print(f"{final}  ({t0 / 60:.1f} min, {final.stat().st_size / 1e6:.1f} MB)")
    print(f"{srt}  ({len(cues)} cues)")
    for a, _, t in chapters:
        print(f"  {int(a // 60):02d}:{int(a % 60):02d}  {t}")
    return 0


def cmd_frames(args) -> int:
    """Write a contact sheet of frames (one every --every seconds) for visual review."""
    p = scene_path(args.quality, args.scene)
    out = BUILD / "frames" / args.scene
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    subprocess.check_call(["ffmpeg", "-v", "error", "-i", str(p), "-vf",
                           f"fps=1/{args.every},scale=640:-1", str(out / "f_%03d.png")])
    n = len(list(out.glob("f_*.png")))
    cols = 3
    rows = (n + cols - 1) // cols
    subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-i", str(out / "f_%03d.png"), "-vf",
                           f"tile={cols}x{max(rows, 1)}:padding=6:color=white", "-frames:v", "1",
                           str(out / "sheet.png")])
    print(f"{n} frames -> {out}/f_*.png ; contact sheet {out}/sheet.png ({probe_duration(p):.1f}s)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--quality", default="low", choices=QUALITY_FLAGS)
    r.add_argument("--only", default="")
    r.add_argument("--jobs", type=int, default=1)
    a = sub.add_parser("assemble")
    a.add_argument("--quality", default="high", choices=QUALITY_FLAGS)
    a.add_argument("--allow-missing", action="store_true")
    f = sub.add_parser("frames")
    f.add_argument("scene")
    f.add_argument("--quality", default="low", choices=QUALITY_FLAGS)
    f.add_argument("--every", type=float, default=4.0)
    sub.add_parser("list")
    args = ap.parse_args()
    if args.cmd == "list":
        for s, c, t in SCENES:
            print(f"{s:28s} {c:28s} {t}")
        return 0
    return {"render": cmd_render, "assemble": cmd_assemble, "frames": cmd_frames}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
