#!/usr/bin/env python3
"""
Create a Subtitle Edit compatible SRT for Odia speech.

Rules
-----
<SILENT></SILENT> = no audible sound / flat waveform
<NOISE></NOISE>    = background noise or background music
<FIL></FIL>          = mumbling / filler sounds such as hmm oh ah uh
normal text         = clean speech

Speech is transcribed and kept in Odia script.
It is NOT transliterated or translated to English.

Example
-------
Odia speech:    \u0bae\u0b41\u0d67 \u0b9f\u0b3f\u0b15\u0b37\u0b4d\u0b23\u0b3e
Output text:    mu tikshna
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy.fft import rfft

# Silero VAD is used only with an in-memory torch tensor, so we do not call
# silero_vad.read_audio() and therefore avoid its torchaudio/torchcodec loader.
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from transformers import AutoModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ASR_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
LANGUAGE_CODE = "or"  # Odia
SAMPLE_RATE = 16000

# VAD / timeline
MIN_SPEECH_MS = 80
MIN_SILENCE_MS = 250
SPEECH_PAD_MS = 80
MAX_ASR_CHUNK_SECONDS = 9.5

# Silence classification
ABSOLUTE_SILENCE_DB = -50.0

# Speech + background cue duration rules
BG_MIN_SECONDS = 2.10  # > 2 seconds
BG_MAX_SECONDS = 3.90  # < 4 seconds
BG_TARGET_SECONDS = 2.90

# Background detector
# A continuous noise floor that is close enough to speech level is considered
# significant background noise. Stable, tonal audio is treated as possible music.
BACKGROUND_RELATIVE_DB = 25.0
MUSIC_STABILITY_THRESHOLD = 0.86
MUSIC_FLATNESS_THRESHOLD = 0.16

MUMBLE_WORDS = {
    "hmm", "hm", "hmmm", "hmmmm", "mm", "mmm", "mmmm",
    "uh", "uhh", "uhhh", "um", "umm", "ummm", "er", "err", "errr",
    "ah", "ahh", "ahhh", "oh", "ohh", "ohhh", "ooh", "oohh",
    "ouch", "eh", "ehh", "huh", "ha", "he", "ho",
}


@dataclass
class Cue:
    start: float
    end: float
    text: str
    kind: str  # speech, mb, silent, noise, noise_speech


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def clean_odia_text(text: str) -> str:
    """Normalize ASR output while preserving Odia Unicode text."""
    text = unicodedata.normalize("NFKC", text or "").strip()

    # Remove punctuation/symbols, but keep letters, numbers, spaces and
    # combining marks used by the Odia script.
    kept = []
    for ch in text:
        category = unicodedata.category(ch)
        if ch.isspace() or category[0] in {"L", "M", "N"}:
            kept.append(ch)
        else:
            kept.append(" ")

    return re.sub(r"\s+", " ", "".join(kept)).strip()


# Odia/Indic filler sounds commonly produced as mumbling or hesitation.
# Keep this list conservative: words such as "ହଁ" can be meaningful speech,
# so they are not automatically treated as mumbling.
MUMBLE_WORDS_ODIA = {
    "ହମ୍", "ହମ", "ହୁମ୍", "ହୁମ",
    "ଉଁ", "ଉଁହୁଁ", "ଉହୁଁ",
    "ଅଁ", "ଅମ୍",
    "ଆହ୍", "ଆହ", "ଓହ୍", "ଓହ",
    "ଏହ୍", "ଏହ",
    "ଉଃ", "ଓଃ",
}


def is_mumble_text(text: str) -> bool:
    """Return True when the ASR result consists only of filler/mumble sounds."""
    words = clean_odia_text(text).split()
    if not words:
        return False
    return all(word in MUMBLE_WORDS_ODIA for word in words)

def clean_asr_result(result) -> str:
    """Handle either a plain string or a tuple/list returned by custom models."""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, (list, tuple)):
        parts = []
        for item in result:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return " ".join(parts).strip()
    return str(result).strip()


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def convert_to_wav(source: Path, destination: Path) -> None:
    """Convert any ffmpeg-supported source to 16 kHz mono PCM WAV."""
    command = [
        "ffmpeg", "-y",
        "-i", str(source),
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-sample_fmt", "s16",
        str(destination),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg was not found. Install ffmpeg and make sure ffmpeg.exe is in PATH."
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -100.0
    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    if rms <= 1e-10:
        return -100.0
    return 20.0 * math.log10(rms)


def interval_audio(audio: np.ndarray, start: float, end: float, sr: int) -> np.ndarray:
    a = max(0, int(round(start * sr)))
    b = min(len(audio), int(round(end * sr)))
    if b <= a:
        return np.empty(0, dtype=np.float32)
    return audio[a:b]


def spectral_features(samples: np.ndarray, sr: int) -> Tuple[float, float]:
    """Return spectral flatness and frame-to-frame spectral stability."""
    frame_len = int(0.50 * sr)
    hop = int(0.25 * sr)
    if len(samples) < frame_len:
        return 1.0, 0.0

    spectra = []
    flatness_values = []
    for start in range(0, len(samples) - frame_len + 1, hop):
        frame = samples[start:start + frame_len]
        spectrum = np.abs(rfft(frame * np.hanning(frame_len))) + 1e-10
        power = spectrum * spectrum
        geometric = math.exp(float(np.mean(np.log(power))))
        arithmetic = float(np.mean(power))
        flatness_values.append(geometric / max(arithmetic, 1e-12))
        spectra.append(spectrum)

    flatness = float(np.mean(flatness_values)) if flatness_values else 1.0
    similarities = []
    for left, right in zip(spectra, spectra[1:]):
        left = left / (np.linalg.norm(left) + 1e-10)
        right = right / (np.linalg.norm(right) + 1e-10)
        similarities.append(float(np.dot(left, right)))
    stability = float(np.mean(similarities)) if similarities else 0.0
    return flatness, stability


def background_present(
    samples: np.ndarray,
    sr: int,
    ambient_floor_db: float,
) -> bool:
    """Heuristic background-noise/music detector for speech segments."""
    if len(samples) < int(0.15 * sr):
        return False

    speech_level = dbfs(samples)
    if speech_level <= ABSOLUTE_SILENCE_DB:
        return False

    relative_gap = speech_level - ambient_floor_db
    continuous_noise = (
        ambient_floor_db > ABSOLUTE_SILENCE_DB
        and relative_gap <= BACKGROUND_RELATIVE_DB
    )

    flatness, stability = spectral_features(samples, sr)
    music_like = (
        stability >= MUSIC_STABILITY_THRESHOLD
        and flatness <= MUSIC_FLATNESS_THRESHOLD
    )

    return continuous_noise or music_like


def in_intervals(t: float, intervals: Sequence[Tuple[float, float]]) -> bool:
    return any(start <= t < end for start, end in intervals)


def merge_intervals(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 0.02:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def estimate_ambient_floor(
    audio: np.ndarray,
    sr: int,
    speech_intervals: Sequence[Tuple[float, float]],
) -> float:
    frame_len = int(0.25 * sr)
    values = []
    for start in range(0, max(0, len(audio) - frame_len + 1), frame_len):
        center = (start + frame_len / 2) / sr
        if not in_intervals(center, speech_intervals):
            values.append(dbfs(audio[start:start + frame_len]))

    if not values:
        return -60.0
    return float(np.percentile(values, 30))


def classify_gap(audio: np.ndarray, start: float, end: float, sr: int) -> str:
    chunk = interval_audio(audio, start, end, sr)
    if len(chunk) == 0:
        return "silent"

    level = dbfs(chunk)
    peak = float(np.max(np.abs(chunk)))
    if level <= ABSOLUTE_SILENCE_DB or peak < 1e-4:
        return "silent"
    return "noise"


# ---------------------------------------------------------------------------
# ASR helpers
# ---------------------------------------------------------------------------


def load_asr_model(model_id: str, token: Optional[str] = None):
    print(f"Loading Odia ASR model {model_id} on CPU...")
    try:
        kwargs = {"trust_remote_code": True}
        if token:
            kwargs["token"] = token
        return AutoModel.from_pretrained(model_id, **kwargs)
    except Exception as exc:
        message = str(exc)
        if "gated" in message.lower() or "401" in message or "403" in message:
            raise RuntimeError(
                "The AI4Bharat model is gated. Open the model page, accept its "
                "conditions, then run `huggingface-cli login` and try again."
            ) from exc
        raise


def make_asr_chunks(
    speech_intervals: Sequence[Tuple[float, float]],
    max_duration: float = MAX_ASR_CHUNK_SECONDS,
) -> List[Tuple[float, float]]:
    """
    Keep VAD speech regions separate.

    This is important for subtitle tagging: if several speech regions are
    merged into one ASR chunk, the silence between them disappears from the
    timeline and <SILENT></SILENT> can never be emitted for that gap.
    """
    chunks: List[Tuple[float, float]] = []

    for start, end in speech_intervals:
        # Split unusually long VAD regions without crossing a VAD boundary.
        cursor = start
        while cursor < end:
            chunk_end = min(end, cursor + max_duration)
            chunks.append((cursor, chunk_end))
            cursor = chunk_end

    return chunks

def transcribe_chunk(model, audio: np.ndarray, start: float, end: float, sr: int) -> str:
    chunk = interval_audio(audio, start, end, sr)
    if len(chunk) == 0:
        return ""

    tensor = torch.from_numpy(np.asarray(chunk, dtype=np.float32)).unsqueeze(0)
    with torch.inference_mode():
        result = model(tensor, LANGUAGE_CODE, "ctc")
    return clean_asr_result(result)


def transcribe_odia(model, audio: np.ndarray, sr: int, speech_intervals):
    chunks = make_asr_chunks(speech_intervals)
    cues: List[Cue] = []

    for index, (start, end) in enumerate(chunks, start=1):
        print(f"  ASR chunk {index}/{len(chunks)}: {start:.2f}s -> {end:.2f}s")
        try:
            raw = transcribe_chunk(model, audio, start, end, sr)
        except Exception as exc:
            print(f"  Warning: ASR failed for chunk {index}: {exc}")
            continue

        # Keep the actual Odia script; do not transliterate it.
        text = clean_odia_text(raw)
        if not text:
            print(f"  Warning: ASR returned text but transliteration produced nothing in chunk {index}")
            print(f"  Raw ASR: {raw!r}")
            continue
        print(f"  Text: {text}")

        kind = "mb" if is_mumble_text(text) else "speech"
        cues.append(Cue(start, end, text, kind))

    return cues


# ---------------------------------------------------------------------------
# Timeline / tagging
# ---------------------------------------------------------------------------


def fill_gaps(
    speech_cues: Sequence[Cue],
    audio: np.ndarray,
    sr: int,
    duration: float,
) -> List[Cue]:
    result: List[Cue] = []
    cursor = 0.0

    for cue in sorted(speech_cues, key=lambda c: c.start):
        if cue.start > cursor + 0.01:
            result.append(Cue(
                cursor,
                cue.start,
                "",
                classify_gap(audio, cursor, cue.start, sr),
            ))
        result.append(cue)
        cursor = max(cursor, cue.end)

    if cursor < duration - 0.01:
        result.append(Cue(
            cursor,
            duration,
            "",
            classify_gap(audio, cursor, duration, sr),
        ))

    return result


def apply_background_tags(
    cues: Sequence[Cue],
    audio: np.ndarray,
    sr: int,
    ambient_floor_db: float,
) -> List[Cue]:
    out = []
    for cue in cues:
        if cue.kind != "speech":
            out.append(cue)
            continue

        chunk = interval_audio(audio, cue.start, cue.end, sr)
        if background_present(chunk, sr, ambient_floor_db):
            out.append(Cue(cue.start, cue.end, cue.text, "noise_speech"))
        else:
            out.append(cue)
    return out


def split_text_proportionally(text: str, pieces: int) -> List[str]:
    words = text.split()
    if pieces <= 1 or len(words) <= 1:
        return [text]

    result = []
    for i in range(pieces):
        a = int(round(len(words) * i / pieces))
        b = int(round(len(words) * (i + 1) / pieces))
        if b <= a:
            b = min(len(words), a + 1)
        if a < len(words):
            result.append(" ".join(words[a:b]))
    return result or [text]


def enforce_background_duration(cues: Sequence[Cue]) -> List[Cue]:
    """Split long speech and let short speech borrow adjacent silence/noise.

    Clean speech and speech-with-background cues target 2–4 seconds. When a
    speech cue is shorter than 2 seconds, extend it into adjacent silent/noise
    cues if available, shortening/removing those gap cues so the timeline does
    not overlap. Mumble cues and noise/silence-only cues are not expanded.
    """
    # First split long speech cues proportionally across their words.
    split: List[Cue] = []
    for cue in sorted(cues, key=lambda c: (c.start, c.end)):
        if cue.kind not in {"speech", "noise_speech"}:
            split.append(cue)
            continue

        duration = cue.end - cue.start
        words = cue.text.split()
        if duration <= 4.0 or len(words) < 2:
            split.append(cue)
            continue

        pieces = max(2, math.ceil(duration / BG_TARGET_SECONDS))
        pieces = min(pieces, len(words))
        parts = split_text_proportionally(cue.text, pieces)
        if len(parts) < 2:
            split.append(cue)
            continue

        chunk_duration = duration / len(parts)
        for i, part in enumerate(parts):
            a = cue.start + i * chunk_duration
            b = cue.end if i == len(parts) - 1 else cue.start + (i + 1) * chunk_duration
            split.append(Cue(a, b, part, cue.kind))

    split.sort(key=lambda c: (c.start, c.end))

    # Expand short speech into neighboring silence/noise only (never into
    # another speech cue). The gap cue is trimmed or removed accordingly.
    for i, cue in enumerate(split):
        if cue.kind not in {"speech", "noise_speech"}:
            continue
        need = max(0.0, 2.0 - (cue.end - cue.start))
        if need <= 0.001:
            continue

        # Borrow from the immediately preceding silence/noise cue first.
        if i > 0 and need > 0.001:
            prev = split[i - 1]
            if prev.kind in {"silent", "noise"} and abs(prev.end - cue.start) <= 0.03:
                take = min(need, max(0.0, prev.end - prev.start))
                cue.start -= take
                prev.end -= take
                need -= take
                if prev.end <= prev.start + 0.001:
                    # Keep a valid Cue object during iteration; zero-length
                    # cues are removed in the final filtering step.
                    prev.end = prev.start

        # Then borrow from the following silence/noise cue.
        if i + 1 < len(split) and need > 0.001:
            nxt = split[i + 1]
            if nxt is not None and nxt.kind in {"silent", "noise"} and abs(nxt.start - cue.end) <= 0.03:
                take = min(need, max(0.0, nxt.end - nxt.start))
                cue.end += take
                nxt.start += take
                need -= take
                if nxt.end <= nxt.start + 0.001:
                    # Do not insert None into split: the loop below expects Cue.
                    nxt.start = nxt.end

    return sorted([c for c in split if c.end > c.start], key=lambda c: (c.start, c.end))


def render_text(cue: Cue) -> str:
    if cue.kind == "silent":
        return "<SILENT></SILENT>"
    if cue.kind == "noise":
        return "<NOISE></NOISE>"
    if cue.kind == "mb":
        return f"<FIL>{cue.text}</FIL>"
    if cue.kind == "noise_speech":
        return f"<NOISE>{cue.text}</NOISE>"
    return cue.text


def srt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(cues: Sequence[Cue], output: Path) -> None:
    lines = []
    number = 1
    for cue in cues:
        if cue.end <= cue.start:
            continue
        lines.append(str(number))
        lines.append(f"{srt_time(cue.start)} --> {srt_time(cue.end)}")
        lines.append(render_text(cue))
        lines.append("")
        number += 1
    output.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a Subtitle Edit compatible tagged SRT from Odia speech."
    )
    parser.add_argument("audio", type=Path, help="Input audio/video file")
    parser.add_argument(
        "--output", type=Path,
        help="Output SRT path. Default: same filename with .srt",
    )
    parser.add_argument(
        "--model", default=ASR_MODEL_ID,
        help=f"Hugging Face Odia ASR model (default: {ASR_MODEL_ID})",
    )
    parser.add_argument(
        "--hf-token", default=None,
        help="Optional Hugging Face read token. `huggingface-cli login` is preferred.",
    )
    args = parser.parse_args()

    if not args.audio.exists():
        raise SystemExit(f"Input file not found: {args.audio}")

    output = args.output or args.audio.with_suffix(".srt")

    token = args.hf_token
    if token is None:
        import os
        token = os.environ.get("HF_TOKEN")

    model = load_asr_model(args.model, token=token)

    with tempfile.TemporaryDirectory(prefix="subtitle_tag_") as tmp:
        wav_file = Path(tmp) / "audio.wav"
        print("Converting audio...")
        convert_to_wav(args.audio, wav_file)

        audio, sr = sf.read(wav_file, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = np.asarray(audio, dtype=np.float32)
        duration = len(audio) / sr

        print("Detecting speech...")
        vad_model = load_silero_vad()
        wav_tensor = torch.from_numpy(audio)
        speech_timestamps = get_speech_timestamps(
            wav_tensor,
            vad_model,
            sampling_rate=SAMPLE_RATE,
            return_seconds=True,
            min_speech_duration_ms=MIN_SPEECH_MS,
            min_silence_duration_ms=MIN_SILENCE_MS,
            speech_pad_ms=SPEECH_PAD_MS,
        )

        speech_intervals = merge_intervals([
            (float(item["start"]), float(item["end"]))
            for item in speech_timestamps
        ])

        ambient_floor_db = estimate_ambient_floor(audio, sr, speech_intervals)
        print(f"Estimated ambient floor: {ambient_floor_db:.1f} dBFS")

        print("Transcribing Odia speech...")
        speech_cues = transcribe_odia(model, audio, sr, speech_intervals)

        print("Building timeline...")
        cues = fill_gaps(speech_cues, audio, sr, duration)
        cues = apply_background_tags(cues, audio, sr, ambient_floor_db)
        cues = enforce_background_duration(cues)

        cleaned = []
        for cue in cues:
            start = max(0.0, min(duration, cue.start))
            end = max(0.0, min(duration, cue.end))
            if end <= start:
                continue
            cleaned.append(Cue(start, end, cue.text, cue.kind))

        write_srt(cleaned, output)

    print(f"Done: {output}")


if __name__ == "__main__":
    main()
