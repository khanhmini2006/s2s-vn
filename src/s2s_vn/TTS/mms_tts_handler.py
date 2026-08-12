"""TTS handler dùng Facebook MMS-TTS (VITS, tiếng Việt: facebook/mms-tts-vie).

LƯU Ý LICENSE: model facebook/mms-tts-vie phát hành theo CC-BY-NC-4.0
(Creative Commons Attribution-NonCommercial) — CHỈ dùng phi thương mại.
Khác MIT của VieNeu. Không dùng backend này nếu dự án có yếu tố thương mại.

TTSInput → AudioOutput (PCM16). Model sinh audio 1 lần (không autoregressive,
RTF~0.01 trên GPU) — tự cắt chunk giả lập streaming để hành vi khớp AudioOutput
hiện có trong pipeline, theo cách facebookmms_handler.py (repo gốc
huggingface/speech-to-speech) làm.

Không hỗ trợ voice cloning/nhiều giọng — chỉ 1 giọng cố định do checkpoint quyết
định. Sample rate đọc từ model.config.sampling_rate (facebook/mms-tts-vie =
16000Hz, khớp pipeline — không resample thực tế), tự resample nếu model khác
có sample rate lệch, theo đúng cách facebookmms_handler.py gốc làm (không
hardcode giả định).
"""

from __future__ import annotations

import queue
import threading

import numpy as np

from ..base_handler import BaseHandler
from ..pipeline.messages import AudioOutput, ResponseDone, TTSInput

CHUNK_MS = 200  # độ dài mỗi audio chunk giả lập streaming


class MMSTTSHandler(BaseHandler):
    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        name: str = "tts",
        model_name: str = "facebook/mms-tts-vie",
        output_sample_rate: int = 16000,
        device: str = "cuda",
        cancel_scope=None,
    ):
        super().__init__(input_queue, output_queue, name)
        self.model_name = model_name
        self.output_sample_rate = output_sample_rate
        self.device = device
        self.cancel_scope = cancel_scope
        self._model = None
        self._tokenizer = None
        self._model_sample_rate = None
        # warmup() có thể bị gọi từ 2 thread cùng lúc: thread warmup_all() nền
        # và thread run() của chính handler này (process() tự lazy-warmup nếu
        # chưa sẵn sàng). transformers 5.x lazy-import không thread-safe — 2
        # thread cùng import lần đầu → ImportError/treo (đã tái hiện thực tế).
        self._warmup_lock = threading.Lock()

    def _cancelled(self, gen: int) -> bool:
        return self.cancel_scope is not None and self.cancel_scope.current() != gen

    def warmup(self) -> None:
        with self._warmup_lock:
            if self._model is not None:  # thread khác đã warmup xong trong lúc chờ lock
                return
            import torch
            from transformers import AutoTokenizer, VitsModel

            device = self.device if torch.cuda.is_available() else "cpu"
            self.device = device
            model = VitsModel.from_pretrained(self.model_name).to(device)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._tokenizer = tokenizer
            # đọc sample rate THẬT từ model (không hardcode) — theo cách repo gốc
            # facebookmms_handler.py làm: resample dựa vào self.model.config.sampling_rate,
            # không giả định cố định. facebook/mms-tts-vie = 16000Hz nhưng handler này
            # nhận model_name tùy ý qua registry, phải tổng quát cho checkpoint khác.
            self._model_sample_rate = model.config.sampling_rate
            self._model = model  # gán cuối cùng — báo "sẵn sàng" cho check ngoài lock

    def process(self, item):
        if isinstance(item, AudioOutput):  # pass-through (resampled audio khác nguồn)
            return item
        if not isinstance(item, TTSInput):
            return None
        if self._model is None:
            self.warmup()

        import torch

        gen = self.cancel_scope.current() if self.cancel_scope else 0

        inputs = self._tokenizer(item.text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(self.device).long()
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            output = self._model(input_ids=input_ids, attention_mask=attention_mask)

        audio = output.waveform.cpu().numpy().squeeze()
        if self._model_sample_rate != self.output_sample_rate:
            from scipy.signal import resample_poly

            audio = resample_poly(audio, self.output_sample_rate, self._model_sample_rate)
        pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()

        chunk_bytes = int(self.output_sample_rate * CHUNK_MS / 1000) * 2  # PCM16 = 2 bytes/sample
        for i in range(0, len(pcm16), chunk_bytes):
            if self._cancelled(gen):
                break
            self.output_queue.put(AudioOutput(
                audio=pcm16[i:i + chunk_bytes],
                sample_rate=self.output_sample_rate,
                turn_id=item.turn_id,
            ))
        return ResponseDone(turn_id=item.turn_id)
