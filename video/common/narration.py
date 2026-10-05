"""Narration for ManimGL scenes: text-to-speech, audio placement and subtitle timing.

A scene speaks with a context manager:

    with self.voiceover("The gate decides which channel a token writes to.") as vo:
        self.play(FadeIn(gate), run_time=vo.duration / 2)
        self.play(Indicate(gate))
    # leaving the block waits until the narration has finished

The text is synthesized once with Kokoro (an 82M-parameter TTS model, run through
ONNX on the CPU) and cached by content hash, so re-rendering a scene costs nothing.
Each sentence's start and end time is logged and written to build/subs/<Scene>.json
when the scene ends; build.py turns those logs into an .srt file for the full video.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import wave
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

VIDEO_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = VIDEO_DIR / ".tts_cache"
SUBS_DIR = VIDEO_DIR / "build" / "subs"
MODEL_DIR = Path(os.environ.get("CLANKERS_TTS_DIR", VIDEO_DIR / "tts_models"))

VOICE = os.environ.get("CLANKERS_VOICE", "af_heart")
SPEED = float(os.environ.get("CLANKERS_SPEED", "1.0"))
SAMPLE_RATE = 24000
SENTENCE_GAP = 0.18  # seconds of silence between sentences
TAIL = 0.35          # silence after a block, before the next one may start

# Spoken forms for things the voice would otherwise mispronounce. Applied to the
# audio only; subtitles keep the written form.
PRONUNCIATION = [
    (r"\bBDH\b", "B D H"),
    (r"\bLLMs?\b", lambda m: "L L M" + ("s" if m.group(0).endswith("s") else "")),
    (r"\bGPU\b", "G P U"),
    (r"\bCPU\b", "C P U"),
    (r"\bCTX\b", "context"),
    (r"\bkWTA\b", "k-winners-take-all"),
    (r"\bJSON\b", "Jason"),
    (r"\bMcNemar\b", "Mac Nemar"),
    (r"\bLayerNorm\b", "layer norm"),
    (r"\bReLU\b", "rel-you"),
    (r"\bRoPE\b", "rope"),
    (r"\bη²", "eta squared"),
    (r"\bvs\.?\b", "versus"),
    (r"\be\.g\.", "for example"),
    (r"\bi\.e\.", "that is"),
    (r"\bS(\d+)\b", r"S \1"),
    (r"(\d+)\s*/\s*(\d+)", r"\1 out of \2"),
    (r"\bk\s*=\s*(\d+)", r"k equals \1"),
    (r"×", " times "),
    (r"−", " minus "),
    (r"≥", " at least "),
    (r"≤", " at most "),
    (r"%", " percent"),
]


def spoken_form(text: str) -> str:
    out = text
    for pattern, repl in PRONUNCIATION:
        out = re.sub(pattern, repl, out)
    return re.sub(r"\s+", " ", out).strip()


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    # split after . ! ? or ; when followed by a space and an uppercase letter, digit or quote
    parts = re.split(r"(?<=[.!?;])\s+(?=[A-Z0-9\"“(])", text)
    return [p.strip() for p in parts if p.strip()]


_kokoro = None


def _engine():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro

        _kokoro = Kokoro(str(MODEL_DIR / "kokoro-v1.0.onnx"), str(MODEL_DIR / "voices-v1.0.bin"))
    return _kokoro


def _write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.wav")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    os.replace(tmp, path)


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


@dataclass
class Synth:
    path: Path
    duration: float
    sentences: list[tuple[str, float, float]]  # (written text, start, end) relative to block start


def synthesize(text: str, spoken: str | None = None) -> Synth:
    """Synthesize `text` sentence by sentence; cache the joined wav and its timings."""
    sentences = split_sentences(text)
    spoken_sentences = split_sentences(spoken) if spoken else [spoken_form(s) for s in sentences]
    if spoken and len(spoken_sentences) != len(sentences):
        # timings fall back to one cue for the whole block
        sentences = [" ".join(sentences)]
        spoken_sentences = [" ".join(spoken_sentences)]
    key_src = json.dumps([VOICE, SPEED, spoken_sentences])
    key = hashlib.sha1(key_src.encode()).hexdigest()[:16]
    wav_path = CACHE_DIR / f"{key}.wav"
    meta_path = CACHE_DIR / f"{key}.json"
    if wav_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return Synth(wav_path, meta["duration"], [tuple(s) for s in meta["sentences"]])

    engine = _engine()
    chunks, timings, t = [], [], 0.0
    gap = np.zeros(int(SENTENCE_GAP * SAMPLE_RATE), dtype=np.float32)
    for written, said in zip(sentences, spoken_sentences):
        samples, sr = engine.create(said, voice=VOICE, speed=SPEED, lang="en-us")
        assert sr == SAMPLE_RATE, sr
        samples = _trim(np.asarray(samples, dtype=np.float32))
        dur = len(samples) / SAMPLE_RATE
        timings.append((written, round(t, 3), round(t + dur, 3)))
        chunks += [samples, gap]
        t += dur + SENTENCE_GAP
    audio = np.concatenate(chunks[:-1]) if chunks else np.zeros(1, dtype=np.float32)
    _write_wav(wav_path, audio)
    duration = _wav_duration(wav_path)
    meta_path.write_text(json.dumps({"duration": duration, "sentences": timings, "voice": VOICE,
                                     "speed": SPEED, "spoken": spoken_sentences}, indent=1))
    return Synth(wav_path, duration, timings)


def _trim(samples: np.ndarray, threshold: float = 0.01, pad: int = 1200) -> np.ndarray:
    """Remove leading/trailing near-silence the model adds, keeping a short pad."""
    loud = np.nonzero(np.abs(samples) > threshold)[0]
    if len(loud) == 0:
        return samples
    a = max(0, loud[0] - pad)
    b = min(len(samples), loud[-1] + pad)
    return samples[a:b]


@dataclass
class Voiceover:
    scene: object
    synth: Synth
    start: float
    extra: float = 0.0

    @property
    def duration(self) -> float:
        return self.synth.duration

    def elapsed(self) -> float:
        return self.scene.time - self.start

    def remaining(self) -> float:
        return max(0.0, self.synth.duration - self.elapsed())

    def time_of(self, sentence_index: int) -> float:
        """Start time (relative to the block) of sentence `sentence_index`."""
        return self.synth.sentences[sentence_index][1]

    def wait_until(self, t: float) -> None:
        """Wait until `t` seconds into the block (no-op if already past)."""
        dt = t - self.elapsed()
        if dt > 1 / 60:
            self.scene.wait(dt)

    def wait_until_sentence(self, sentence_index: int) -> None:
        self.wait_until(self.time_of(sentence_index))


class NarrationMixin:
    """Mix into a manimlib Scene: adds voiceover() and writes subtitle timings."""

    def _subs(self) -> list:
        if not hasattr(self, "_subtitle_log"):
            self._subtitle_log = []
        return self._subtitle_log

    @contextmanager
    def voiceover(self, text: str, spoken: str | None = None, tail: float = TAIL):
        synth = synthesize(text, spoken)
        start = self.time
        self.add_sound(str(synth.path))
        vo = Voiceover(self, synth, start)
        for written, a, b in synth.sentences:
            self._subs().append({"text": written, "start": round(start + a, 3), "end": round(start + b, 3)})
        yield vo
        rest = vo.remaining() + tail
        if rest > 1 / 60:
            self.wait(rest)

    def write_subtitles(self) -> None:
        if getattr(self, "skip_animations", False) and not self._subs():
            return
        SUBS_DIR.mkdir(parents=True, exist_ok=True)
        out = {"scene": type(self).__name__, "duration": round(self.time, 3), "cues": self._subs()}
        (SUBS_DIR / f"{type(self).__name__}.json").write_text(json.dumps(out, indent=1))
