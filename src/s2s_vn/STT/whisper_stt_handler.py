"""STT handler dùng faster-whisper với PhoWhisper-medium CT2 (tiếng Việt)."""

from __future__ import annotations

import queue

import numpy as np

from ..base_handler import BaseHandler
from ..pipeline.events import PartialTranscriptionEvent
from ..pipeline.messages import AudioMode, VADAudio, Transcription


class WhisperSTTHandler(BaseHandler):
    """VADAudio → Transcription.

    Dùng faster-whisper (CTranslate2). Model: PhoWhisper-medium CT2
    (fine-tune Việt trên Whisper medium), default compute_type int8_float16.
    """

    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        name: str = "stt",
        model_name: str = "quocphu/PhoWhisper-ct2-FasterWhisper",
        model_subfolder: str = "PhoWhisper-medium-ct2-fasterWhisper",
        device: str = "auto",  # auto → CUDA nếu có
        compute_type: str = "int8_float16",
        beam_size: int = 1,  # 1 = greedy, nhanh hơn 3-5× vs beam 5
        language: str = "vi",
        text_out: queue.Queue | None = None,
        enable_live_transcription: bool = True,
        cancel_scope=None,
    ):
        super().__init__(input_queue, output_queue, name)
        self.model_name = model_name
        self.model_subfolder = model_subfolder
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.language = language
        self.text_out = text_out
        self.enable_live_transcription = enable_live_transcription
        self.cancel_scope = cancel_scope
        self._model = None

    def warmup(self) -> None:
        import os

        from huggingface_hub import snapshot_download
        from faster_whisper import WhisperModel

        repo_dir = snapshot_download(
            self.model_name,
            allow_patterns=f"{self.model_subfolder}/*",
        )
        model_path = os.path.join(repo_dir, self.model_subfolder)
        self._model = WhisperModel(
            model_path,
            device=self.device,
            compute_type=self.compute_type,
        )

    def process(self, item):
        if not isinstance(item, VADAudio):
            return None
        if self._model is None:
            self.warmup()

        gen_start = self.cancel_scope.current() if self.cancel_scope else 0

        # live transcription: progressive → PartialTranscriptionEvent (không vào LLM)
        if item.mode == AudioMode.PROGRESSIVE:
            if not self.enable_live_transcription:
                return None
            text = self._transcribe(item.audio, gen_start)
            if text and self.text_out:
                self.text_out.put(PartialTranscriptionEvent(
                    text=text, turn_id=item.turn_id))
            return None

        # final: Transcription → LLM
        text = self._transcribe(item.audio, gen_start)
        if not text:
            return None
        return Transcription(
            text=text,
            language_code=self.language,
            turn_id=item.turn_id,
            turn_revision=item.turn_revision,
        )

    def _transcribe(self, pcm16: bytes, gen_start: int) -> str:
        if not pcm16:
            return ""
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,  # VAD đã cắt trước
            no_speech_threshold=0.6,  # bỏ segment không speech (chống hallucination)
            log_prob_threshold=-1.0,  # bỏ segment log-prob thấp (text ảo)
            condition_on_previous_text=False,  # không lây text cũ → giảm lặp
        )
        
        # Iterate and check cancel scope
        text_parts = []
        for seg in segments:
            if self.cancel_scope and self.cancel_scope.current() != gen_start:
                break
            text_parts.append(seg.text)
            
        return "".join(text_parts).strip()
