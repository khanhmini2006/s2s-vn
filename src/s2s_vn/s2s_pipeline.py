"""Pipeline: dựng chuỗi 6 handler + queue + ThreadManager.

Hiện dùng mock handlers để chứng minh luồng message.
Handler thật thay từng bước (VAD → STT → LLM → TTS).
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field

from .LLM.lm_output_processor import LMOutputProcessor
from .STT.transcription_notifier import MockTranscriptionNotifier
from .backend_registry import LLMConfig, get_llm_handler, get_stt_handler, get_tts_handler
from .VAD.vad_handler import SileroVADHandler
from .base_handler import BaseHandler
from .pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END, CancelScope
from .utils.thread_manager import ThreadManager


@dataclass
class PipelineConfig:
    chunk_size: int = 512  # mẫu PCM16
    sample_rate: int = 16000
    # STT
    stt_name: str = "phowhisper-medium"
    stt_beam_size: int = 1  # PhoWhisper: 1 = greedy; 3-5 = chất lượng hơn, chậm hơn
    stt_compute_type: str = "int8_float16"  # int8_float16 | float16 | int8 | float32
    # LLM
    llm_backend: str = "openai"  # openai | hf-router | gemini | local
    llm_model: str = "gpt-4.1-mini"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_device: str = "cuda"
    llm_system_prompt: str | None = None  # None → default của handler
    # TTS (VieNeu v3 Turbo)
    tts_name: str = "vieneu"
    tts_voice: str = "Trúc Ly"
    tts_streaming: bool = True
    tts_denoise: bool = True
    tts_backend: str = "onnx"  # onnx (streaming, RTF<1) | pytorch (batch) | auto
    tts_style: str = "tu_nhien"  # tu_nhien | tin_tuc | doc_truyen
    tts_temperature: float = 0.8  # sampling độ "sáng tạo" giọng
    tts_max_chars: int = 256  # cỡ chunk streaming
    # realtime
    enable_live_transcription: bool = True
    live_transcription_update_interval: float = 0.1
    speculative_reopen_ms: int = 1000
    # VAD
    vad_threshold: float = 0.6
    min_silence_ms: int = 300  # im bao lâu mới chốt câu (quá nhỏ → cắt nửa câu)
    min_speech_ms: int = 500  # utterance ngắn hơn bị bỏ (noise/ambient)
    speech_pad_ms: int = 500


@dataclass
class PipelineQueues:
    """Tất cả queue + event một pipeline cần."""

    audio_in: queue.Queue = field(default_factory=queue.Queue)
    spoken_prompt: queue.Queue = field(default_factory=queue.Queue)
    stt_output: queue.Queue = field(default_factory=queue.Queue)
    text_prompt: queue.Queue = field(default_factory=queue.Queue)
    lm_response: queue.Queue = field(default_factory=queue.Queue)
    lm_processed: queue.Queue = field(default_factory=queue.Queue)
    audio_out: queue.Queue = field(default_factory=queue.Queue)
    text_out: queue.Queue = field(default_factory=queue.Queue)  # events → transport
    # events điều khiển
    should_listen: threading.Event = field(default_factory=threading.Event)
    stop_event: threading.Event = field(default_factory=threading.Event)
    cancel_scope: CancelScope = field(default_factory=CancelScope)


def _make_handlers(q: PipelineQueues, cfg: PipelineConfig) -> list[BaseHandler]:
    """Dựng chuỗi 6 handler: VAD + STT + LLM + TTS thật."""
    vad = SileroVADHandler(
        q.audio_in, q.spoken_prompt, text_out=q.text_out,
        sample_rate=cfg.sample_rate,
        threshold=cfg.vad_threshold,
        min_silence_ms=cfg.min_silence_ms,
        min_speech_ms=cfg.min_speech_ms,
        speech_pad_ms=cfg.speech_pad_ms,
        cancel_scope=q.cancel_scope,
    )
    stt = get_stt_handler(
        q.spoken_prompt, q.stt_output, cfg,
        text_out=q.text_out,
        cancel_scope=q.cancel_scope,
    )
    notifier = MockTranscriptionNotifier(q.stt_output, q.text_prompt, text_out=q.text_out)
    llm = get_llm_handler(
        q.text_prompt, q.lm_response,
        LLMConfig(
            backend=cfg.llm_backend,
            model_name=cfg.llm_model,
            api_key=cfg.llm_api_key,
            base_url=cfg.llm_base_url,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
            device=cfg.llm_device,
            system_prompt=cfg.llm_system_prompt,
        ),
        text_out=q.text_out,
        cancel_scope=q.cancel_scope,
    )
    processor = LMOutputProcessor(q.lm_response, q.lm_processed, text_out=q.text_out,
                                  cancel_scope=q.cancel_scope)
    tts = get_tts_handler(
        q.lm_processed, q.audio_out, cfg,
        cancel_scope=q.cancel_scope,
    )
    return [vad, stt, notifier, llm, processor, tts]


class S2SPipeline:
    """Đóng gói toàn bộ: queues + handlers + thread manager."""

    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        self.queues = PipelineQueues()
        self.queues.should_listen.set()
        self.handlers = _make_handlers(self.queues, self.cfg)
        self.threads = ThreadManager(self.handlers)

    def start(self) -> None:
        self.threads.start()

    def warmup_all(self, timeout_s: float = 180.0) -> None:
        """Khởi động toàn bộ model handler (gọi warmup của từng handler có warmup).

        Dùng khi WS connect — để session.created đến sau khi model sẵn sàng,
        câu nói đầu không phải chờ tải model.
        """
        import time

        deadline = time.monotonic() + timeout_s
        for h in self.handlers:
            warmup = getattr(h, "warmup", None)
            if warmup is None:
                continue
            if time.monotonic() > deadline:
                break
            try:
                warmup()
            except Exception as e:
                # model fail → lazy warmup vẫn chạy khi process đầu tiên
                import logging
                logging.getLogger("s2s.pipeline").warning(
                    f"[{h.name}] warmup lỗi (sẽ thử lại khi có data): {e!r}")

    def stop(self) -> None:
        self.threads.stop()

    def send_audio(self, pcm_chunk: bytes) -> None:
        """Đẩy PCM16 chunk vào pipeline (từ mic hoặc websocket)."""
        self.queues.audio_in.put(pcm_chunk)

    def stop_listening(self) -> None:
        """Tắt mic (TTS đang nói). Đẩy sentinel báo hết response."""
        self.queues.should_listen.clear()
        self.queues.audio_out.put(AUDIO_RESPONSE_DONE)

    def finish_turn(self) -> None:
        """Báo hết audio vào (test/demo). VAD flush utterance còn dở."""
        self.queues.audio_in.put(PIPELINE_END)

    def wait_audio_out(self, timeout: float = 5.0) -> object | None:
        try:
            return self.queues.audio_out.get(timeout=timeout)
        except queue.Empty:
            return None


def main() -> None:
    """Demo: chạy pipeline giả, đẩy audio, in output."""

    import time

    p = S2SPipeline()
    p.start()
    try:
        for i in range(3):
            p.send_audio(b"\x00" * 1024)  # 1 chunk PCM16 im lặng
            time.sleep(0.05)
        for _ in range(10):
            out = p.wait_audio_out(timeout=1.0)
            if out is None:
                print("[main] hết output (timeout)")
                break
            if isinstance(out, bytes) and out == AUDIO_RESPONSE_DONE:
                print("[main] nhận AUDIO_RESPONSE_DONE")
                continue
            print(f"[main] nhận output: {type(out).__name__}, len={len(out) if hasattr(out, '__len__') else out}")
    finally:
        p.stop()
        print(f"[main] threads còn sống: {p.threads.alive_count}")


if __name__ == "__main__":
    main()
