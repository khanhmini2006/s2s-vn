"""LMOutputProcessor: gom LLM text_delta thành TTSInput đầy đủ.

Nhận LLMResponseChunk → tích luỹ text theo turn_id. Khi EndOfResponse
(turn kết thúc) → emit TTSInput với toàn bộ câu + AssistantTextEvent cho
toàn bộ text (để client hiển thị dần hoặc 1 lần).
"""

from __future__ import annotations

import queue

from ..base_handler import BaseHandler
from ..pipeline.events import AssistantTextEvent
from ..pipeline.messages import (
    EndOfResponse,
    GenerateResponseRequest,
    LLMResponseChunk,
    TTSInput,
)


def _strip_thinking(text: str) -> str:
    """Bỏ <think>...</think> block (model reasoning) — không đọc qua TTS."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _clean_text(text: str) -> str:
    """Làm sạch text trước TTS: bỏ markdown + emoji (TTS không đọc "sao sao"/emoji)."""
    import re
    # markdown: ** * # ` ~ _
    text = re.sub(r"[*_#`~]", "", text)
    # emoji + variation selectors + ZWJ
    text = re.sub(r"[\U0001F000-\U0001FAFF☀-➿️‍]", "", text)
    return text


class LMOutputProcessor(BaseHandler):
    def __init__(self, input_queue: queue.Queue, output_queue: queue.Queue,
                 text_out: queue.Queue | None = None, name: str = "lm-processor",
                 cancel_scope=None):
        super().__init__(input_queue, output_queue, name)
        self.text_out = text_out
        self.cancel_scope = cancel_scope
        # turn_id → text đang tích luỹ
        self._buffers: dict[int, str] = {}
        # turn_id → generation lúc turn bắt đầu (cho barge-in check)
        self._gens: dict[int, int] = {}
        # turn nào đang được TTS (chờ EndOfResponse)
        self._pending_tts: set[int] = set()

    def process(self, item):
        if isinstance(item, LLMResponseChunk):
            tid = item.turn_id
            # gen ghi nhận ở chunk đầu tiên của turn (barge-in baseline)
            self._gens.setdefault(tid, self._gen_current())
            self._buffers[tid] = self._buffers.get(tid, "") + item.text_delta
            if self.text_out:
                self.text_out.put(AssistantTextEvent(text=item.text_delta, turn_id=tid))
            return None

        if isinstance(item, EndOfResponse):
            tid = item.turn_id
            text = self._buffers.pop(tid, "")
            gen = self._gens.pop(tid, None)  # pop để không leak
            # nếu turn bị cancel (barge-in) → không đưa vào TTS
            if gen is not None and self.cancel_scope is not None \
                    and self.cancel_scope.current() != gen:
                return None
            text = _clean_text(_strip_thinking(text))
            if text:
                return TTSInput(text=text, turn_id=tid)
            return None

        if isinstance(item, GenerateResponseRequest):
            # turn mới bắt đầu — xoá buffer cũ (nếu chưa flushed)
            self._buffers.pop(item.turn_id, None)
            return None
        return None

    def _gen_current(self) -> int:
        return self.cancel_scope.current() if self.cancel_scope else 0

    def _turn_gen(self, tid: int) -> int:
        return self._gens.get(tid, 0)
