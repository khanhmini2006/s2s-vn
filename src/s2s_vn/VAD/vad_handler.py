"""VAD handler dùng Silero VAD v5.

Nhận bytes PCM16 16kHz → tích luỹ buffer → phát hiện start/stop speech
→ cắt utterance hoàn chỉnh (VADAudio, mode=final) đẩy vào STT.
"""

from __future__ import annotations

import queue
import threading

import numpy as np

from ..base_handler import BaseHandler
from ..pipeline.events import SpeechStartedEvent, SpeechStoppedEvent
from ..pipeline.messages import PIPELINE_END, AudioMode, VADAudio


class SileroVADHandler(BaseHandler):
    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        name: str = "vad",
        text_out: queue.Queue | None = None,
        sample_rate: int = 16000,
        threshold: float = 0.7,
        min_silence_ms: int = 64,
        min_speech_ms: int = 500,
        speech_pad_ms: int = 500,
        device: str = "cpu",
        cancel_scope=None,
    ):
        super().__init__(input_queue, output_queue, name)
        self.text_out = text_out
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.speech_pad_ms = speech_pad_ms
        self.device = device
        self.cancel_scope = cancel_scope

        self.buffer = bytearray()
        self.speech_acc = bytearray()  # audio speech tích luỹ (chưa có pad)
        self.speech_started = False
        self.silence_ms = 0
        self.turn_id = 0
        self._session = None
        self._state = None
        self._frame_samples = 512

    def warmup(self) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        # Silero VAD v5 ONNX từ HF (ổn định hơn torch.hub/GitHub)
        ckpt = hf_hub_download(
            "onnx-community/silero-vad",
            "onnx/model.onnx",
        )
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(ckpt, sess_opts, providers=providers)
        # Silero ONNX: state = (state, context) tensor 2x1x128; frame phải có context 64 mẫu
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._frame_samples = 512  # mẫu/khung

    def _vad_frame(self, pcm16: bytes) -> bool:
        """Trả True nếu frame chứa speech."""
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        x = audio.reshape(1, -1)  # rank 2: (batch, samples)
        prob, state_out = self._session.run(
            None,
            {
                "input": x,
                "state": self._state,
                "sr": np.array(self.sample_rate, dtype=np.int64),
            },
        )
        self._state = state_out
        return float(np.asarray(prob).reshape(-1)[0]) >= self.threshold

    def process(self, item):
        if item == PIPELINE_END:
            # flush phần còn lại (buffer + speech đang tích luỹ)
            if self.buffer:
                # xử lý buffer dư như 1 frame im lặng
                del self.buffer[:]
            if self.speech_started:
                self._emit_utterance()
            return None
        if not isinstance(item, bytes):
            return None
        if self._session is None:
            self.warmup()

        self.buffer.extend(item)
        # phân tích theo frame 512 mẫu (1024 bytes PCM16)
        frame_bytes = self._frame_samples * 2
        if len(self.buffer) < frame_bytes:
            return None

        # lấy 1 frame
        frame = bytes(self.buffer[:frame_bytes])
        del self.buffer[:frame_bytes]

        is_speech = self._vad_frame(frame)
        if is_speech:
            self.silence_ms = 0
            if not self.speech_started:
                self.speech_started = True
                self.turn_id += 1
                # barge-in: turn mới → cancel response cũ (nếu có)
                if self.cancel_scope is not None:
                    self.cancel_scope.cancel()
                self._emit(SpeechStartedEvent(turn_id=self.turn_id))
                self._frame_count = 0
            self.speech_acc.extend(frame)
            # live transcription: mỗi ~1s phát 1 đoạn progressive
            self._frame_count = getattr(self, "_frame_count", 0) + 1
            if self._frame_count % (self.sample_rate // self._frame_samples) == 0 \
                    and len(self.speech_acc) > 0:
                self._emit_progressive()
        else:
            if self.speech_started:
                # im lặng trong utterance — vẫn giữ để có pad tự nhiên
                self.speech_acc.extend(frame)
            self.silence_ms += (frame_bytes / 2) / self.sample_rate * 1000

        if self.speech_started and self.silence_ms >= self.min_silence_ms:
            self._emit_utterance()
        return None

    def _emit_progressive(self) -> None:
        """Emit VADAudio progressive (chưa hoàn chỉnh) cho live transcription."""
        if len(self.speech_acc) == 0:
            return
        self.output_queue.put(
            VADAudio(audio=bytes(self.speech_acc), mode=AudioMode.PROGRESSIVE,
                     turn_id=self.turn_id, sample_rate=self.sample_rate)
        )

    def _emit_utterance(self) -> None:
        if len(self.speech_acc) == 0:
            return
        # filter utterance quá ngắn (noise/ambient) — tính theo ms
        speech_ms = len(self.speech_acc) / 2 / self.sample_rate * 1000
        if speech_ms < self.min_speech_ms:
            self.speech_acc.clear()
            self.speech_started = False
            self.silence_ms = 0
            return
        audio = bytes(self.speech_acc)
        self.speech_acc.clear()
        self._emit(SpeechStoppedEvent(turn_id=self.turn_id))
        self.speech_started = False
        self.silence_ms = 0
        self.output_queue.put(
            VADAudio(audio=audio, mode=AudioMode.FINAL, turn_id=self.turn_id,
                     sample_rate=self.sample_rate)
        )

    def _emit(self, event) -> None:
        if self.text_out:
            self.text_out.put(event)
