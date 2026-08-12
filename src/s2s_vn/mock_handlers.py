"""Handler giả cho test (skeleton): chứng minh luồng message qua 6 stage.

MockTranscriptionNotifier đã tách sang STT/transcription_notifier.py (part pipeline
chính) — file này chỉ còn mock dùng trong tests/test_pipeline.py.
"""

from __future__ import annotations

import queue

from .base_handler import BaseHandler
from .pipeline.events import AssistantTextEvent, SpeechStartedEvent, SpeechStoppedEvent
from .pipeline.messages import (
    AUDIO_RESPONSE_DONE,
    PIPELINE_END,
    GenerateResponseRequest,
    LLMResponseChunk,
    TTSInput,
    Transcription,
    VADAudio,
)


class MockVADHandler(BaseHandler):
    """bytes → VADAudio. Giả: mỗi chunk là 1 utterance hoàn chỉnh."""

    def __init__(self, input_queue, output_queue, name="vad", chunk_size=512,
                 sample_rate=16000, text_out: queue.Queue | None = None):
        super().__init__(input_queue, output_queue, name)
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.text_out = text_out
        self.turn_id = 0

    def process(self, item):
        if item == PIPELINE_END:
            return None
        if not isinstance(item, bytes):
            return None
        self.turn_id += 1
        tid = self.turn_id
        if self.text_out:
            self.text_out.put(SpeechStartedEvent(turn_id=tid))
            self.text_out.put(SpeechStoppedEvent(turn_id=tid))
        return VADAudio(audio=item, mode="final", turn_id=tid, sample_rate=self.sample_rate)


class MockSTTHandler(BaseHandler):
    """VADAudio → Transcription."""

    def __init__(self, input_queue, output_queue, name="stt"):
        super().__init__(input_queue, output_queue, name)

    def process(self, item):
        if isinstance(item, VADAudio):
            return Transcription(
                text="giả: tiếng việt được nhận diện",
                language_code="vi",
                turn_id=item.turn_id,
                turn_revision=item.turn_revision,
            )
        return None


class MockLLMHandler(BaseHandler):
    """GenerateResponseRequest → LLMResponseChunk (giả 1 chunk)."""

    def __init__(self, input_queue, output_queue, name="llm"):
        super().__init__(input_queue, output_queue, name)

    def process(self, item):
        if isinstance(item, GenerateResponseRequest):
            return LLMResponseChunk(
                text_delta=f"Trả lời giả cho: {item.text}",
                turn_id=item.turn_id,
            )
        return None


class MockLMOutputProcessor(BaseHandler):
    """LLMResponseChunk → TTSInput; emit AssistantTextEvent."""

    def __init__(self, input_queue, output_queue, text_out: queue.Queue | None = None,
                 name="lm-processor"):
        super().__init__(input_queue, output_queue, name)
        self.text_out = text_out

    def process(self, item):
        if isinstance(item, LLMResponseChunk):
            if self.text_out:
                self.text_out.put(AssistantTextEvent(
                    text=item.text_delta, turn_id=item.turn_id))
            return TTSInput(text=item.text_delta, turn_id=item.turn_id)
        return None


class MockTTSHandler(BaseHandler):
    """TTSInput → AudioOutput (bytes giả)."""

    def __init__(self, input_queue, output_queue, name="tts", chunk_size=512,
                 sample_rate=16000):
        super().__init__(input_queue, output_queue, name)
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate

    def process(self, item):
        if isinstance(item, TTSInput):
            # audio giả: 0.5s PCM16 im lặng, 1 chunk
            n = self.sample_rate // 2
            return b"\x00\x00" * n
        if item == AUDIO_RESPONSE_DONE:
            return item
        return None
