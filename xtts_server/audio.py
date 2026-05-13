"""
audio.py — audio encoding and file I/O utilities.

All synthesis produces float32 PCM at 24 000 Hz (mono).  This module converts
that raw array into any of the four supported container formats before sending
it to a client or writing it to disk.

Format routing
--------------
  WAV   — soundfile  (lossless, float32 → PCM-16 for max compatibility)
  FLAC  — soundfile  (lossless, PCM-24 for better dynamic range than PCM-16)
  OGG   — soundfile  (lossy Vorbis; widely supported in browsers)
  MP3   — pydub + ffmpeg  (lossy; requires ffmpeg on PATH)

pydub is only imported when MP3 is requested so the server starts fine even
if ffmpeg is absent, as long as nobody requests MP3.
"""

import io
import os
from typing import Literal

import numpy as np
import soundfile as sf

from logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Types and constants
# ---------------------------------------------------------------------------

AudioFormat = Literal["wav", "mp3", "ogg", "flac"]

SUPPORTED_FORMATS: set[str] = {"wav", "mp3", "ogg", "flac"}

# MIME type for each format — used in HTTP Content-Type headers.
MIME_TYPES: dict[str, str] = {
    "wav":  "audio/wav",
    "mp3":  "audio/mpeg",
    "ogg":  "audio/ogg",
    "flac": "audio/flac",
}

# soundfile format/subtype pairs for the three non-MP3 formats.
# PCM_16 for WAV/OGG keeps file sizes reasonable; PCM_24 for FLAC preserves
# more of the float32 dynamic range without much size penalty.
_SF_PARAMS: dict[str, tuple[str, str]] = {
    "wav":  ("WAV",  "PCM_16"),
    "ogg":  ("OGG",  "VORBIS"),
    "flac": ("FLAC", "PCM_24"),
}


# ---------------------------------------------------------------------------
# Core conversion helpers
# ---------------------------------------------------------------------------

def _float32_to_int16(audio: np.ndarray) -> np.ndarray:
    """
    Clip float32 samples to [-1, 1] then scale to int16 range.
    Hard clipping is intentional — values outside [-1, 1] indicate synthesis
    artefacts and should be clamped rather than wrapping around.
    """
    return (np.clip(audio, -1.0, 1.0) * 32_767).astype(np.int16)


def audio_to_bytes(
    audio: np.ndarray,
    fmt: AudioFormat,
    sample_rate: int = 24_000,
) -> bytes:
    """
    Encode a float32 PCM array to the requested format and return raw bytes.

    Parameters
    ----------
    audio       : float32 numpy array, shape (N,), values in [-1, 1]
    fmt         : one of "wav", "mp3", "ogg", "flac"
    sample_rate : source sample rate (default 24 000 Hz for XTTS-v2)

    Returns
    -------
    Encoded audio bytes suitable for writing to disk or an HTTP response body.
    """
    buf = io.BytesIO()

    if fmt == "mp3":
        _encode_mp3(audio, sample_rate, buf)
    else:
        sf_format, sf_subtype = _SF_PARAMS[fmt]
        # soundfile accepts float32 directly and handles the conversion
        # to the target subtype internally.
        sf.write(buf, audio, sample_rate, format=sf_format, subtype=sf_subtype)

    return buf.getvalue()


def _encode_mp3(audio: np.ndarray, sample_rate: int, buf: io.BytesIO) -> None:
    """Encode to MP3 via pydub (requires ffmpeg on PATH)."""
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise RuntimeError(
            "MP3 encoding requires pydub and ffmpeg. "
            "Install pydub and ensure ffmpeg is on PATH."
        ) from exc

    pcm_int16 = _float32_to_int16(audio)
    segment = AudioSegment(
        data=pcm_int16.tobytes(),
        sample_width=2,       # 2 bytes = int16
        frame_rate=sample_rate,
        channels=1,
    )
    segment.export(buf, format="mp3")


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def save_audio(
    audio: np.ndarray,
    path: str,
    fmt: AudioFormat,
    sample_rate: int = 24_000,
) -> int:
    """
    Encode and write audio to `path`.

    The directory is created if it does not exist.
    Returns the number of bytes written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    data = audio_to_bytes(audio, fmt, sample_rate)
    with open(path, "wb") as f:
        f.write(data)

    size_kb = len(data) / 1024
    duration_s = len(audio) / sample_rate
    logger.info(
        "Audio saved | path=%s | format=%s | size=%.1f KB | duration=%.2f s",
        path, fmt, size_kb, duration_s,
    )
    return len(data)


def load_wav_as_float32(path: str) -> tuple[np.ndarray, int]:
    """
    Load any soundfile-supported audio file and return (float32_array, sample_rate).

    Used when validating an uploaded speaker reference audio file — XTTS-v2
    expects a real wav path, but this helper lets us verify the file is readable
    before passing it to the model.
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    # Downmix stereo/multi-channel to mono by averaging channels.
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    logger.debug("Loaded audio | path=%s | sr=%d | samples=%d", path, sr, len(audio))
    return audio, sr


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def mime_type(fmt: AudioFormat) -> str:
    """Return the MIME type string for a given format."""
    return MIME_TYPES[fmt]


def output_filename(job_id: str, fmt: AudioFormat) -> str:
    """Build a deterministic output filename for a job's audio file."""
    return f"{job_id}.{fmt}"
