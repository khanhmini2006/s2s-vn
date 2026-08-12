"""TTS handler dùng Piper TTS (VITS + ONNX runtime, giọng huongly tiếng Việt).

LƯU Ý LICENSE: package piper-tts (repo OHF-voice/piper1-gpl) phát hành theo
GPL-3.0-or-later — copyleft mã nguồn, khác các ràng buộc chỉ-giới-hạn-mục-đích-dùng
của MIT (vieneu) hay CC-BY-NC-4.0 (mms-vie). Dự án này khai Apache-2.0 — có xung
đột license tiềm ẩn khi tích hợp trực tiếp (import cùng process). Người dùng đã
được cảnh báo và chấp nhận rủi ro này khi quyết định tích hợp.
Giọng huongly.onnx không nằm trong kho chính thức rhasspy/piper-voices — nguồn
gốc/license riêng của checkpoint voice chưa xác minh.

TTSInput → AudioOutput (PCM16). Piper.synthesize() sinh audio THEO TỪNG CÂU
(sentence-level streaming thật, không phải tự cắt chunk giả lập như MMS-TTS) —
mỗi AudioChunk trả về đẩy ngay 1 AudioOutput, TTFA thấp hơn khi văn bản có nhiều câu.
Sample rate đọc từ từng AudioChunk (huongly = 22050Hz) — resample về pipeline
rate nếu khác, dùng scipy.signal.resample_poly (nhất quán vieneu/mms).
"""

from __future__ import annotations

import queue
import threading

import numpy as np

from ..base_handler import BaseHandler
from ..pipeline.messages import AudioOutput, ResponseDone, TTSInput


class PiperTTSHandler(BaseHandler):
    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        name: str = "tts",
        model_path: str = "",
        config_path: str = "",
        output_sample_rate: int = 16000,
        use_cuda: bool = False,
        cancel_scope=None,
    ):
        super().__init__(input_queue, output_queue, name)
        self.model_path = model_path
        self.config_path = config_path
        self.output_sample_rate = output_sample_rate
        self.use_cuda = use_cuda
        self.cancel_scope = cancel_scope
        self._voice = None
        # warmup() có thể bị gọi từ 2 thread cùng lúc (warmup_all() nền +
        # process() tự lazy-warmup) — cùng lý do đã áp dụng cho MMSTTSHandler.
        self._warmup_lock = threading.Lock()

    def _cancelled(self, gen: int) -> bool:
        return self.cancel_scope is not None and self.cancel_scope.current() != gen

    def warmup(self) -> None:
        with self._warmup_lock:
            if self._voice is not None:  # thread khác đã warmup xong trong lúc chờ lock
                return
            from piper import PiperVoice

            self._voice = PiperVoice.load(
                self.model_path, config_path=self.config_path, use_cuda=self.use_cuda,
            )

    def process(self, item):
        if isinstance(item, AudioOutput):  # pass-through (resampled audio khác nguồn)
            return item
        if not isinstance(item, TTSInput):
            return None
        if self._voice is None:
            self.warmup()

        from piper.config import SynthesisConfig

        gen = self.cancel_scope.current() if self.cancel_scope else 0
        syn_config = SynthesisConfig()

        for chunk in self._voice.synthesize(item.text, syn_config=syn_config):
            if self._cancelled(gen):
                break
            audio = chunk.audio_float_array
            if chunk.sample_rate != self.output_sample_rate:
                from scipy.signal import resample_poly

                audio = resample_poly(audio, self.output_sample_rate, chunk.sample_rate)
            pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
            self.output_queue.put(AudioOutput(
                audio=pcm16,
                sample_rate=self.output_sample_rate,
                turn_id=item.turn_id,
            ))
        return ResponseDone(turn_id=item.turn_id)
