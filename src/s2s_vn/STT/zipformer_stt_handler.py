"""STT handler dùng Zipformer-30M-RNNT (sherpa-onnx) — tiếng Việt, cực nhẹ + nhanh.

Model: hynt/Zipformer-30M-RNNT-6000h (giải Nhất VLSP 2025, license cc-by-nc-nd —
chỉ dùng nghiên cứu/cá nhân, KHÔNG thương mại).
"""

from __future__ import annotations

import queue

import numpy as np

from ..base_handler import BaseHandler
from ..pipeline.events import PartialTranscriptionEvent
from ..pipeline.messages import AudioMode, Transcription, VADAudio


class ZipformerSTTHandler(BaseHandler):
    """VADAudio → Transcription.

    Zipformer-RNNT qua sherpa-onnx (encoder/decoder/joiner ONNX int8 + bpe tokens).
    """

    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        name: str = "stt",
        model_name: str = "hynt/Zipformer-30M-RNNT-6000h",
        beam_size: int = 1,  # 1 = greedy_search (nhanh); >1 = modified_beam_search
        num_threads: int = 4,
        sample_rate: int = 16000,
        language: str = "vi",
        text_out: queue.Queue | None = None,
        enable_live_transcription: bool = True,
        cancel_scope=None,
    ):
        super().__init__(input_queue, output_queue, name)
        self.model_name = model_name
        self.beam_size = beam_size
        self.num_threads = num_threads
        self.sample_rate = sample_rate
        self.language = language
        self.text_out = text_out
        self.enable_live_transcription = enable_live_transcription
        self.cancel_scope = cancel_scope
        self._rec = None

    def warmup(self) -> None:
        import os

        import sherpa_onnx
        from huggingface_hub import hf_hub_download

        enc = hf_hub_download(self.model_name, "encoder-epoch-20-avg-10.int8.onnx")
        dec = hf_hub_download(self.model_name, "decoder-epoch-20-avg-10.int8.onnx")
        join = hf_hub_download(self.model_name, "joiner-epoch-20-avg-10.int8.onnx")
        bpe = hf_hub_download(self.model_name, "bpe.model")

        # tokens.txt từ bpe.model (symbol<tab>id) — lưu bên cạnh bpe
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=bpe)
        tok_path = os.path.join(os.path.dirname(bpe), "tokens.txt")
        with open(tok_path, "w", encoding="utf-8") as f:
            for i in range(sp.get_piece_size()):
                f.write(f"{sp.id_to_piece(i)}\t{i}\n")

        decoding = ("modified_beam_search" if self.beam_size > 1 else "greedy_search")
        self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=tok_path, encoder=enc, decoder=dec, joiner=join,
            num_threads=self.num_threads,
            decoding_method=decoding)

    def process(self, item):
        if not isinstance(item, VADAudio):
            return None
        if self._rec is None:
            self.warmup()

        gen_start = self.cancel_scope.current() if self.cancel_scope else 0

        # live transcription: progressive → PartialTranscriptionEvent (không vào LLM)
        if item.mode == AudioMode.PROGRESSIVE:
            if not self.enable_live_transcription:
                return None
            text = self._decode(item.audio, gen_start)
            if text and self.text_out:
                self.text_out.put(PartialTranscriptionEvent(
                    text=text, turn_id=item.turn_id))
            return None

        # final: Transcription → LLM
        text = self._decode(item.audio, gen_start)
        if not text:
            return None
        return Transcription(
            text=text,
            language_code=self.language,
            turn_id=item.turn_id,
            turn_revision=item.turn_revision,
        )

    def _decode(self, pcm16: bytes, gen_start: int) -> str:
        if not pcm16:
            return ""
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        stream = self._rec.create_stream()
        stream.accept_waveform(self.sample_rate, audio)
        self._rec.decode_stream(stream)
        if self.cancel_scope and self.cancel_scope.current() != gen_start:
            return ""
        # BPE pieces viết hoa → lower cho khớp convention (LLM/UI)
        return stream.result.text.lower().strip()
