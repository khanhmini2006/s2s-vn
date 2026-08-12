"""Test luồng message của mock handlers (không load model)."""

import queue
import threading

from s2s_vn.mock_handlers import (
    MockLLMHandler,
    MockLMOutputProcessor,
    MockSTTHandler,
    MockTTSHandler,
    MockVADHandler,
)
from s2s_vn.STT.transcription_notifier import MockTranscriptionNotifier
from s2s_vn.pipeline.messages import (
    AUDIO_RESPONSE_DONE,
    AudioOutput,
    GenerateResponseRequest,
    LLMResponseChunk,
    PIPELINE_END,
    TTSInput,
    Transcription,
    VADAudio,
)


def build_mock_chain():
    """Dựng chuỗi 6 mock handler nối bằng queue."""
    q = {
        "audio_in": queue.Queue(),
        "spoken": queue.Queue(),
        "stt": queue.Queue(),
        "prompt": queue.Queue(),
        "lm": queue.Queue(),
        "lm_proc": queue.Queue(),
        "audio_out": queue.Queue(),
        "text_out": queue.Queue(),
    }
    vad = MockVADHandler(q["audio_in"], q["spoken"], text_out=q["text_out"])
    stt = MockSTTHandler(q["spoken"], q["stt"])
    notifier = MockTranscriptionNotifier(q["stt"], q["prompt"], text_out=q["text_out"])
    llm = MockLLMHandler(q["prompt"], q["lm"])
    proc = MockLMOutputProcessor(q["lm"], q["lm_proc"], text_out=q["text_out"])
    tts = MockTTSHandler(q["lm_proc"], q["audio_out"])
    return q, [vad, stt, notifier, llm, proc, tts]


def run_handlers(handlers):
    threads = [threading.Thread(target=h.run, daemon=True) for h in handlers]
    for t in threads:
        t.start()
    return threads


def test_mock_full_flow():
    """Mock: audio in → events → audio out."""
    q, handlers = build_mock_chain()
    threads = run_handlers(handlers)
    try:
        q["audio_in"].put(b"\x00" * 1024)
        # chờ audio out
        out = q["audio_out"].get(timeout=2.0)
        assert isinstance(out, bytes) and len(out) > 0

        events = []
        while not q["text_out"].empty():
            events.append(q["text_out"].get())
        kinds = {type(e).__name__ for e in events}
        assert "SpeechStartedEvent" in kinds
        assert "SpeechStoppedEvent" in kinds
        assert "TranscriptionCompletedEvent" in kinds
        assert "AssistantTextEvent" in kinds
    finally:
        for t in threads:
            t.join(timeout=1.0)


def test_mock_message_types():
    """Mock: mỗi stage trả đúng type."""
    q, handlers = build_mock_chain()
    threads = run_handlers(handlers)
    try:
        q["audio_in"].put(b"\x00" * 1024)
        out = q["audio_out"].get(timeout=2.0)
        assert isinstance(out, bytes)
        # check AssistantTextEvent
        got = None
        while not q["text_out"].empty():
            e = q["text_out"].get()
            if type(e).__name__ == "AssistantTextEvent":
                got = e
        assert got is not None
        assert got.text.startswith("Trả lời giả cho:")
    finally:
        for t in threads:
            t.join(timeout=1.0)
