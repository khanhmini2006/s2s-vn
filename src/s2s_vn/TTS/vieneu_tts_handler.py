"""TTS handler dùng VieNeu-TTS v3 Turbo (tiếng Việt).

TTSInput → AudioOutput (PCM16). Streaming: sinh chunk, resample 48k→16k,
đẩy từng chunk ra output ngay (TTFA < 1s).
"""

from __future__ import annotations

import queue

import numpy as np

from ..base_handler import BaseHandler
from ..pipeline.messages import AudioOutput, ResponseDone, TTSInput

CROSSFADE_SAMPLES = 160  # 10ms @16k — crossfade nối chunk TTS, loại click ở biên


class VieNeuTTSHandler(BaseHandler):
    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        name: str = "tts",
        voice: str = "Trúc Ly",
        output_sample_rate: int = 16000,
        streaming: bool = True,
        denoise: bool = True,
        backend: str = "onnx",  # onnx (streaming) | pytorch (batch) | auto
        style: str = "tu_nhien",  # tu_nhien | tin_tuc | doc_truyen
        temperature: float = 0.8,
        max_chars: int = 256,
        cancel_scope=None,
    ):
        super().__init__(input_queue, output_queue, name)
        self.voice = voice
        self.output_sample_rate = output_sample_rate
        self.streaming = streaming
        self.denoise = denoise
        self.backend = backend
        self.style = style
        self.temperature = temperature
        self.max_chars = max_chars
        self.cancel_scope = cancel_scope
        self._tts = None
        self._resample_ratio = None

    def _cancelled(self, gen: int) -> bool:
        return self.cancel_scope is not None and self.cancel_scope.current() != gen

    def warmup(self) -> None:
        from vieneu import Vieneu

        self._tts = Vieneu(mode="v3turbo", backend=self.backend)
        # trả chunk dạng float array tần số 48k
        self._src_rate = self._tts.sample_rate
        self._resample_ratio = self.output_sample_rate / self._src_rate

    def _to_pcm16(self, chunk: np.ndarray) -> bytes:
        """float array (48k) → PCM16 bytes (16k)."""
        from scipy.signal import resample_poly

        x = np.asarray(chunk, dtype=np.float32)
        if self._resample_ratio != 1.0:
            x = resample_poly(x, self.output_sample_rate, self._src_rate)
        return (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()

    def _pcm16_to_float(self, b: bytes) -> np.ndarray:
        return np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0

    def _float_to_pcm16(self, x: np.ndarray) -> bytes:
        return (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()

    def _crossfade(self, tail: np.ndarray, head: np.ndarray) -> np.ndarray:
        """Fade-out tail (chunk trước) chồng fade-in head (chunk sau) → nối liền."""
        n = min(len(tail), len(head))
        fade_out = np.linspace(1, 0, n, dtype=np.float32)
        fade_in = 1 - fade_out
        return np.concatenate([tail[:n] * fade_out + head[:n] * fade_in, head[n:]])

    def process(self, item):
        if isinstance(item, AudioOutput):  # pass-through (resampled audio khác nguồn)
            return item
        if not isinstance(item, TTSInput):
            return None
        if self._tts is None:
            self.warmup()

        gen = self.cancel_scope.current() if self.cancel_scope else 0
        
        # Check if self.voice is a custom voice
        import pathlib
        current_dir = pathlib.Path(__file__).parent
        static_voices_dir = current_dir.parent / "api" / "static" / "voices"
        custom_voice_path = static_voices_dir / f"{self.voice}.wav"
        
        ref_audio_arg = None
        voice_arg = self.voice
        if custom_voice_path.exists():
            ref_audio_arg = str(custom_voice_path)
            voice_arg = None

        if self.streaming:
            # put từng chunk vào queue NGAY (streaming thật, TTFA thấp).
            # Crossfade 10ms ở biên chunk: VieNeu cắt chunk ở dấu câu → nối thẳng
            # gây click/rẹt (phase lệch + micro-gap). Giữ tail chunk trước, fade
            # chồng với head chunk sau → nối liền, hết rẹt.
            stream_kwargs = {"ref_audio": ref_audio_arg, "voice": voice_arg,
                             "style": self.style, "temperature": self.temperature,
                             "max_chars": self.max_chars, "denoise": self.denoise}
            pending_tail = None  # float32 16k — phần cuối chunk trước chưa gửi
            for chunk in self._tts.infer_stream(item.text, **stream_kwargs):
                if self._cancelled(gen):
                    break
                samples = self._pcm16_to_float(self._to_pcm16(chunk))
                if len(samples) == 0:
                    continue
                if pending_tail is not None:
                    samples = self._crossfade(pending_tail, samples)
                # giữ 10ms cuối cho chunk sau; phần còn lại gửi ngay
                keep = samples[-CROSSFADE_SAMPLES:] if len(samples) > CROSSFADE_SAMPLES \
                    else samples.copy()
                to_send = samples[:-CROSSFADE_SAMPLES] if len(samples) > CROSSFADE_SAMPLES \
                    else samples
                pending_tail = keep
                if len(to_send) > 0:
                    self.output_queue.put(AudioOutput(
                        audio=self._float_to_pcm16(to_send),
                        sample_rate=self.output_sample_rate,
                        turn_id=item.turn_id,
                    ))
            # flush tail cuối (không còn chunk sau để crossfade)
            if pending_tail is not None and len(pending_tail) > 0 and not self._cancelled(gen):
                self.output_queue.put(AudioOutput(
                    audio=self._float_to_pcm16(pending_tail),
                    sample_rate=self.output_sample_rate,
                    turn_id=item.turn_id,
                ))
        else:
            if not self._cancelled(gen):
                audio = self._tts.infer(item.text, ref_audio=ref_audio_arg, voice=voice_arg)
                self.output_queue.put(AudioOutput(
                    audio=self._to_pcm16(audio),
                    sample_rate=self.output_sample_rate,
                    turn_id=item.turn_id,
                ))
        # báo hết response cho turn này
        return ResponseDone(turn_id=item.turn_id)
