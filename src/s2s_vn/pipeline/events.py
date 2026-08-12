"""Events đồng bộ hoá giữa pipeline và lớp transport (Realtime)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpeechStartedEvent:
    turn_id: int


@dataclass
class SpeechStoppedEvent:
    turn_id: int


@dataclass
class PartialTranscriptionEvent:
    text: str
    turn_id: int


@dataclass
class TranscriptionCompletedEvent:
    text: str
    language_code: str
    turn_id: int


@dataclass
class AssistantTextEvent:
    text: str
    turn_id: int


@dataclass
class FunctionCallEvent:
    """LLM yêu cầu gọi function."""

    name: str
    arguments: str  # JSON string
    call_id: str
    turn_id: int


@dataclass
class TokenUsageEvent:
    input_tokens: int
    output_tokens: int
    turn_id: int


@dataclass
class ResponseFailedEvent:
    reason: str
    turn_id: int
