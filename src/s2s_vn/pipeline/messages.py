"""Messages truyền giữa các handler trong pipeline."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum


class AudioMode(Enum):
    """Cách một đoạn audio được xử lý."""

    FINAL = "final"
    PROGRESSIVE = "progressive"


@dataclass
class VADAudio:
    """Đoạn speech do VAD cắt ra."""

    audio: bytes  # PCM16
    mode: AudioMode
    turn_id: int
    turn_revision: int = 0
    sample_rate: int = 16000


@dataclass
class Transcription:
    """Kết quả STT hoàn chỉnh."""

    text: str
    language_code: str
    turn_id: int
    turn_revision: int = 0


@dataclass
class PartialTranscription:
    """Kết quả STT tạm thời (live)."""

    text: str
    language_code: str
    turn_id: int


@dataclass
class GenerateResponseRequest:
    """Text đẩy vào LLM."""

    text: str
    language_code: str
    turn_id: int
    turn_revision: int = 0
    use_tools: bool = True
    # tool calling: client trả output sau function_call
    tool_call_id: str | None = None
    tool_output: str | None = None


@dataclass
class LLMResponseChunk:
    """Token/text tạm thời từ LLM."""

    text_delta: str
    turn_id: int


@dataclass
class TokenUsage:
    """Token sử dụng của LLM."""

    input_tokens: int
    output_tokens: int
    turn_id: int


@dataclass
class EndOfResponse:
    """LLM kết thúc một response."""

    turn_id: int


@dataclass
class TTSInput:
    """Text sạch đẩy vào TTS."""

    text: str
    turn_id: int


@dataclass
class AudioOutput:
    """Audio đầu ra từ TTS."""

    audio: bytes  # PCM16
    sample_rate: int
    turn_id: int


# Sentinel bytes trên audio queue
PIPELINE_END = b"END"
AUDIO_RESPONSE_DONE = b"__RESPONSE_DONE__"


@dataclass
class ResponseDone:
    """Báo TTS đã gen xong 1 turn. Mang turn_id để service không finish nhầm turn khác."""
    turn_id: int


@dataclass
class StopListening:
    """Báo VAD ngừng nghe mic (TTS đang nói)."""


class CancelScope:
    """Barge-in: hủy response đang chạy bằng generation counter.

    Chia sẻ giữa các handler. Mỗi lần cancel() → generation += 1.
    Handler đang gen (TTS/LM) giữ generation lúc bắt đầu; nếu thấy
    current khác generation → turn đã bị hủy, dừng ngay.
    """

    def __init__(self) -> None:
        self._generation = 0
        self._lock = threading.Lock()

    def cancel(self) -> None:
        with self._lock:
            self._generation += 1

    def current(self) -> int:
        with self._lock:
            return self._generation

    def reset(self) -> None:
        with self._lock:
            self._generation = 0
