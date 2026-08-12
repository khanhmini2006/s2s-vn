"""Transcription → GenerateResponseRequest (tên theo repo gốc STT/transcription_notifier.py).

Part của pipeline chính: đẩy Transcription qua thành GenerateResponseRequest cho LLM.
"""

from __future__ import annotations

import queue

from ..base_handler import BaseHandler
from ..pipeline.events import TranscriptionCompletedEvent
from ..pipeline.messages import GenerateResponseRequest, Transcription


class MockTranscriptionNotifier(BaseHandler):
    """Transcription → GenerateResponseRequest; emit TranscriptionCompletedEvent."""

    def __init__(self, input_queue, output_queue, text_out: queue.Queue | None = None,
                 name="transcription-notifier"):
        super().__init__(input_queue, output_queue, name)
        self.text_out = text_out

    def process(self, item):
        if isinstance(item, Transcription):
            if self.text_out:
                self.text_out.put(TranscriptionCompletedEvent(
                    text=item.text, language_code=item.language_code, turn_id=item.turn_id))
            return GenerateResponseRequest(
                text=item.text, language_code=item.language_code, turn_id=item.turn_id)
        return None
