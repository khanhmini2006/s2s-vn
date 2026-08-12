"""Helper audio: PCM16 resample + base64 encode/decode cho Realtime protocol."""

from __future__ import annotations

import base64

import numpy as np


def pcm16_to_float(pcm: bytes, sample_rate: int) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def float_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()


def resample_pcm16(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample PCM16 bytes giữa 2 sample rate (dùng scipy)."""
    if from_rate == to_rate:
        return pcm
    from scipy.signal import resample_poly

    x = pcm16_to_float(pcm, from_rate)
    y = resample_poly(x, to_rate, from_rate)
    return float_to_pcm16(y)


def b64_to_pcm16(b64: str) -> bytes:
    return base64.b64decode(b64)


def pcm16_to_b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")
