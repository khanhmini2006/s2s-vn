"""RealtimeService: dịch OpenAI Realtime protocol ↔ pipeline.

Nhận client events (dict) → gọi pipeline. Nhận pipeline events/audio
(từ text_out + audio_out queue) → tạo server events.

Session state: item_id, response_id, conversation_id, audio buffer.
Sample rate: protocol = pipeline = 16kHz (không resample thừa, như repo gốc).
"""

from __future__ import annotations

import base64
import json
import queue
import threading
import uuid

from ...pipeline.events import (
    AssistantTextEvent,
    FunctionCallEvent,
    PartialTranscriptionEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TokenUsageEvent,
    TranscriptionCompletedEvent,
)
from ...pipeline.messages import (
    PIPELINE_END,
    AudioOutput,
    GenerateResponseRequest,
    ResponseDone,
)
from ...s2s_pipeline import PipelineConfig, S2SPipeline
from .audio_utils import pcm16_to_b64, resample_pcm16

PROTOCOL_RATE = 16000
PIPELINE_RATE = 16000

# Voice OpenAI → VieNeu voice mapping (mặc định; có thể override qua session.update)
VOICE_MAP = {
    "alloy": "Trúc Ly",
    "ash": "Trúc Ly",
    "ballad": "Trúc Ly",
    "coral": "Đoan Trang",
    "echo": "Minh Đức",
    "sage": "Ngọc Linh",
    "shimmer": "Mai Anh",
    "verse": "Xuân Vĩnh",
    "marin": "Trúc Ly",
    "cedar": "Minh Triết",
}


def _uuid(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


class RealtimeService:
    """Một session Realtime = một pipeline. Không thread-safe: gọi từ 1 loop."""

    def __init__(self, cfg: PipelineConfig, on_event=None):
        self.cfg = cfg
        self.pipeline = S2SPipeline(cfg)
        self.pipeline.start()
        # pre-warm model nền (Qwen3-8B/STT/TTS) — KHÔNG block event loop;
        # xong → emit server.model_ready để demo bật nút Bắt đầu
        self._warmup_done = threading.Event()
        threading.Thread(target=self._warmup_models, daemon=True).start()
        self._drain_cb = on_event  # on_event(dict)

        # session state
        self.session_id = _uuid("sess_")
        self.conversation_id = _uuid("conv_")
        self.response_id = None
        self.model = cfg.llm_model
        self.instructions = None
        self.voice = cfg.tts_voice
        self.turn_detection = {"type": "server_vad", "create_response": True,
                               "interrupt_response": True}
        self.output_modalities = ["audio"]
        self._seq = 0

        # track
        self._turn_item_id = None
        self._turn_id_to_item = {}  # pipeline turn_id → protocol item_id
        self._response_status = None
        self._buffered_deltas = []  # gom transcript deltas cho response.done
        self._usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self._output_item_id = None
        self._current_response_turn = None
        self._current_vad_turn = 0
        self._fc = None  # FunctionCallEvent cuối (tool calling)

        # drain thread đưa pipeline events → asyncio queue (do server setup)
        self._drain_thread = None
        self._stop_drain = threading.Event()
        self._process_task = None
        self._loop = None
        self._pending = None
        self._process_event = None

    # --- lifecycle ---
    def _warmup_models(self) -> None:
        """Warmup model nền; báo server.model_ready khi xong (không block asyncio)."""
        self.pipeline.warmup_all()
        # Warmup embedding RAG (multilingual-e5-small): không warmup → lần query
        # đầu tiên (tool call) phải tải model ~12s, chặn luôn vòng LLM.
        try:
            from s2s_vn.api.rag_service import rag_service
            rag_service.search("warmup", top_k=1)
        except Exception as e:
            print(f"[warmup] RAG embedding lỗi (sẽ tải lazy khi tool call): {e!r}", flush=True)
        self._warmup_done.set()
        try:
            self._emit("server.model_ready")
        except Exception:
            pass

    def start_drain(self, on_event=None) -> None:
        """Đưa pipeline text_out/audio_out → on_event(dict).

        Drain thread gọi _handle_* trực tiếp. Known issue: shared state
        (response_id, _current_response_turn) mutate từ drain thread trong
        khi asyncio chạy handle_event — race lý thuyết nhưng hiếm trigger
        vì VAD events và client events hiếm đồng thời. TODO: route qua
        asyncio queue nếu cần thread-safe chặt.
        """
        if on_event is not None:
            self._drain_cb = on_event

        def _drain():
            import traceback
            while not self._stop_drain.is_set():
                got = False
                try:
                    ev = self.pipeline.queues.text_out.get(timeout=0.05)
                    self._handle_pipeline_event(ev)
                    got = True
                except queue.Empty:
                    pass
                except Exception:
                    traceback.print_exc()
                    continue
                try:
                    ao = self.pipeline.queues.audio_out.get(timeout=0.05)
                    self._handle_audio(ao)
                    got = True
                except queue.Empty:
                    pass
                except Exception:
                    traceback.print_exc()
                    continue
                if not got:
                    self._stop_drain.wait(0.05)
        self._drain_thread = threading.Thread(target=_drain, daemon=True)
        self._drain_thread.start()

    def stop(self) -> None:
        self._stop_drain.set()
        self.pipeline.stop()

    # --- emit server event ---
    def _emit(self, type_: str, **fields) -> None:
        ev = {"type": type_, "event_id": _uuid("evt_"), **fields}
        if self._drain_cb:
            self._drain_cb(ev)

    # --- client events ---
    def handle_event(self, client_event: dict) -> None:
        """Nhận client event JSON (đã parse). Dispatch theo type."""
        try:
            etype = client_event.get("type")
            handler = getattr(self, f"_on_{etype.replace('.', '_')}", None)
            if handler is None:
                self._emit_error(f"unknown client event type: {etype}",
                                 client_event.get("event_id"))
                return
            handler(client_event)
        except Exception as e:
            self._emit_error(f"{e!r}", client_event.get("event_id"))

    # --- session ---
    def _on_session_update(self, ev: dict) -> None:
        sess = ev.get("session", {})
        if "instructions" in sess:
            self.instructions = sess["instructions"]
            llm = self._llm_handler()
            if self.instructions and hasattr(llm, "system_prompt"):
                llm.system_prompt = self.instructions
        if "model" in sess and sess["model"] != self.model:
            self._emit_error("cannot change model during session", ev.get("event_id"))
            return
        ad = sess.get("audio", {})
        if "output" in ad:
            out = ad["output"]
            if "voice" in out:
                v = out["voice"]
                if isinstance(v, str) and v in VOICE_MAP:
                    self.voice = VOICE_MAP[v]
                elif isinstance(v, str):
                    self.voice = v  # user tự đặt tên voice VieNeu
        td = (ad.get("input", {}) or {}).get("turn_detection")
        if td is not None:
            self.turn_detection = td
        if "output_modalities" in sess:
            self.output_modalities = sess["output_modalities"]
        if "tools" in sess:
            llm = self._llm_handler()
            if hasattr(llm, "set_tools"):
                llm.set_tools(sess["tools"])
        # trả full session config
        self._emit("session.updated", session=self._session_object())

    def _llm_handler(self):
        return self.pipeline.handlers[3]  # vad, stt, notifier, llm, proc, tts

    # --- input audio ---
    def _on_input_audio_buffer_append(self, ev: dict) -> None:
        audio_b64 = ev.get("audio", "")
        if not audio_b64:
            self._emit_error("missing audio", ev.get("event_id"))
            return
        try:
            pcm24 = base64.b64decode(audio_b64)
        except Exception:
            self._emit_error("invalid base64 audio", ev.get("event_id"))
            return
        pcm16 = resample_pcm16(pcm24, PROTOCOL_RATE, PIPELINE_RATE)
        # chunk 1024 bytes (512 mẫu 16k)
        for i in range(0, len(pcm16), 1024):
            self.pipeline.send_audio(pcm16[i:i + 1024])

    def _on_input_audio_buffer_commit(self, ev: dict) -> None:
        self.pipeline.finish_turn()

    def _on_input_audio_buffer_clear(self, ev: dict) -> None:
        self._emit("input_audio_buffer.cleared")

    def _on_pipeline_update(self, ev: dict) -> None:
        """Custom event: cập nhật VAD params runtime (không trong OpenAI spec)."""
        vad = ev.get("vad", {})
        vad_handler = self.pipeline.handlers[0]
        if "min_silence_ms" in vad:
            vad_handler.min_silence_ms = vad["min_silence_ms"]
        if "threshold" in vad:
            vad_handler.threshold = vad["threshold"]
        if "speech_pad_ms" in vad:
            vad_handler.speech_pad_ms = vad["speech_pad_ms"]

    # --- response ---
    def _on_response_create(self, ev: dict) -> None:
        # VAD mode: pipeline tự trigger. Ở đây hỗ trợ trường hợp turn_detection null
        # hoặc client muốn generate thêm: feed text prompt trực tiếp.
        self.pipeline.finish_turn()

    def _on_response_cancel(self, ev: dict) -> None:
        self._cancel_current_response("client_cancelled")

    # --- conversation items ---
    def _on_conversation_item_create(self, ev: dict) -> None:
        item = ev.get("item", {})
        itype = item.get("type")
        if itype == "message":
            content = item.get("content", [])
            texts = [c.get("text", "") for c in content if c.get("type") == "input_text"]
            text = "".join(texts).strip()
            if text:
                tid = self._next_turn_id()
                self.pipeline.queues.text_prompt.put(
                    GenerateResponseRequest(text=text, language_code="vi", turn_id=tid))
                self._turn_id_to_item[tid] = _uuid("item_")
        elif itype == "function_call_output":
            # tool output → đưa lại LLM để tiếp tục vòng lặp
            call_id = item.get("call_id", "")
            output = item.get("output", "")
            tid = self._next_turn_id()
            self.pipeline.queues.text_prompt.put(GenerateResponseRequest(
                text="", language_code="vi", turn_id=tid,
                tool_call_id=call_id, tool_output=str(output),
            ))
            self._turn_id_to_item[tid] = _uuid("item_")

    # --- helpers ---
    def _next_turn_id(self) -> int:
        self._seq += 1
        return self._seq

    def _cancel_current_response(self, reason: str) -> None:
        # barge-in: cancel pipeline (TTS/LM dừng), response kết thúc cancelled
        self.pipeline.queues.cancel_scope.cancel()
        self._emit("response.done", response=self._response_object("cancelled", reason))
        self.response_id = None

    def _emit_error(self, message: str, client_event_id=None) -> None:
        err = {"type": "invalid_request_error", "code": None, "message": message,
               "param": None}
        if client_event_id:
            err["event_id"] = client_event_id
        self._emit("error", error=err)

    # --- pipeline event → server event ---
    def _handle_pipeline_event(self, ev) -> None:
        import time as _time

        if isinstance(ev, SpeechStartedEvent):
            tid = ev.turn_id
            self._current_vad_turn = tid
            item_id = _uuid("item_")
            self._turn_id_to_item[tid] = item_id
            self._turn_item_id = item_id
            self._timing = {"speech_started": _time.monotonic()}
            self._emit("input_audio_buffer.speech_started", item_id=item_id,
                       audio_start_ms=0)
            # barge-in: nếu đang có response → cancel (turn_detection interrupt)
            if self.response_id and self.turn_detection.get("interrupt_response", True):
                self._cancel_current_response("turn_detected")

        elif isinstance(ev, SpeechStoppedEvent):
            item_id = self._turn_id_to_item.get(ev.turn_id, _uuid("item_"))
            if hasattr(self, '_timing'):
                self._timing["speech_stopped"] = _time.monotonic()
            self._emit("input_audio_buffer.speech_stopped", item_id=item_id,
                       audio_end_ms=0)
            self._emit("input_audio_buffer.committed", item_id=item_id)
            self._emit("conversation.item.created",
                       item=self._user_audio_item(item_id, transcript=None))

        elif isinstance(ev, TranscriptionCompletedEvent):
            item_id = self._turn_id_to_item.get(ev.turn_id, _uuid("item_"))
            if ev.turn_id < self._current_vad_turn:
                return
            if hasattr(self, '_timing'):
                self._timing["stt_done"] = _time.monotonic()
            self._emit(
                "conversation.item.input_audio_transcription.completed",
                item_id=item_id, content_index=0,
                transcript=ev.text,
                usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )

        elif isinstance(ev, PartialTranscriptionEvent):
            item_id = self._turn_id_to_item.get(ev.turn_id, _uuid("item_"))
            if ev.turn_id < self._current_vad_turn:
                return
            self._emit(
                "conversation.item.input_audio_transcription.delta",
                item_id=item_id, content_index=0, delta=ev.text,
            )

        elif isinstance(ev, AssistantTextEvent):
            # response output transcript delta — cần response đang mở
            if self.response_id is None:
                self._start_response(ev.turn_id)
            self._emit("response.output_audio_transcript.delta",
                       response_id=self.response_id, item_id=self._current_output_item_id(),
                       output_index=0, content_index=0, delta=ev.text)
            self._buffered_deltas.append(ev.text)

        elif isinstance(ev, FunctionCallEvent):
            # function_call item + arguments done
            self._fc = ev
            if ev.name == "search_knowledge":
                # server-side tool: tự tra cứu kho tài liệu → đẩy output cho LLM tiếp tục
                # (không cần client trả function_call_output)
                self._execute_knowledge_tool(ev)
            if self.response_id is None:
                # truyền turn_id để ResponseDone(turn) khớp — nếu không, response không bao giờ done
                self._start_response(ev.turn_id)
            call_item_id = _uuid("item_")
            self._emit("response.output_item.added", response_id=self.response_id,
                       output_index=0, item={
                           "id": call_item_id, "object": "realtime.item",
                           "type": "function_call", "status": "in_progress",
                           "call_id": ev.call_id, "name": ev.name,
                           "arguments": "",
                       })
            self._emit("response.function_call_arguments.delta",
                       response_id=self.response_id, item_id=call_item_id,
                       output_index=0, call_id=ev.call_id, delta=ev.arguments)
            self._emit("response.function_call_arguments.done",
                       response_id=self.response_id, item_id=call_item_id,
                       output_index=0, call_id=ev.call_id, name=ev.name,
                       arguments=ev.arguments)
            self._emit("conversation.item.created", item={
                "id": call_item_id, "object": "realtime.item",
                "type": "function_call", "status": "completed",
                "call_id": ev.call_id, "name": ev.name,
                "arguments": ev.arguments,
            })
            self._emit("response.output_item.done", response_id=self.response_id,
                       output_index=0, item={
                           "id": call_item_id, "object": "realtime.item",
                           "type": "function_call", "status": "completed",
                           "call_id": ev.call_id, "name": ev.name,
                           "arguments": ev.arguments,
                       })

        elif isinstance(ev, TokenUsageEvent):
            self._usage = {"input_tokens": ev.input_tokens,
                           "output_tokens": ev.output_tokens,
                           "total_tokens": ev.input_tokens + ev.output_tokens}

        elif type(ev).__name__ == "ResponseFailedEvent":
            self._emit("error", error={"type": "server_error", "code": None,
                                       "message": ev.reason, "param": None})

    def _execute_knowledge_tool(self, ev: FunctionCallEvent) -> None:
        """Tự chạy tool search_knowledge: query ChromaDB → tool_output → LLM.

        Dùng chính turn_id của event để ResponseDone(turn) khớp response đang mở.
        """
        import json as _json

        query = ""
        try:
            query = (_json.loads(ev.arguments) or {}).get("query", "")
        except Exception:
            query = ""
        try:
            from s2s_vn.api.rag_service import rag_service
            chunks = rag_service.search(query, top_k=3) if query else []
        except Exception:
            chunks = []
        if chunks:
            output = "\n\n".join(chunks)
        else:
            output = "Không tìm thấy tài liệu nào phù hợp trong kho tri thức."
        self.pipeline.queues.text_prompt.put(GenerateResponseRequest(
            text="", language_code="vi", turn_id=ev.turn_id,
            tool_call_id=ev.call_id, tool_output=output,
        ))

    def _handle_audio(self, item) -> None:
        if isinstance(item, AudioOutput):
            if self._current_response_turn is not None and item.turn_id != self._current_response_turn:
                return
            if self.response_id is None:
                self._start_response(item.turn_id)
            # timing: first audio delta
            if hasattr(self, '_timing') and "first_audio" not in self._timing:
                import time as _time
                self._timing["first_audio"] = _time.monotonic()
                t = self._timing
                if "speech_stopped" in t:
                    latency = (t["first_audio"] - t["speech_stopped"]) * 1000
                    print(f"[latency] speech_stopped→first_audio: {latency:.0f}ms", flush=True)
                if "stt_done" in t and "speech_stopped" in t:
                    stt_lat = (t["stt_done"] - t["speech_stopped"]) * 1000
                    print(f"[latency] speech_stopped→stt_done: {stt_lat:.0f}ms", flush=True)
            pcm24 = resample_pcm16(item.audio, PIPELINE_RATE, PROTOCOL_RATE)
            self._emit("response.output_audio.delta",
                       response_id=self.response_id,
                       item_id=self._current_output_item_id(),
                       output_index=0, content_index=0,
                       delta=pcm16_to_b64(pcm24))
        elif isinstance(item, ResponseDone):
            if self._current_response_turn == item.turn_id:
                self._finish_response("completed")

    # --- response lifecycle ---
    def _start_response(self, turn_id: int | None = None) -> None:
        self.response_id = _uuid("resp_")
        self._output_item_id = _uuid("item_")
        self._current_response_turn = turn_id
        self._buffered_deltas = []
        self._emit("response.created", response=self._response_object("in_progress"))
        self._emit("response.output_item.added", response_id=self.response_id,
                   output_index=0, item=self._output_item("in_progress"))
        self._emit("response.content_part.added", response_id=self.response_id,
                   item_id=self._output_item_id, output_index=0, content_index=0,
                   part={"type": "audio"})

    def _finish_response(self, status: str) -> None:
        if self.response_id is None:
            return
        # timing: total response latency
        if hasattr(self, '_timing') and "speech_stopped" in self._timing:
            import time as _time
            total = (_time.monotonic() - self._timing["speech_stopped"]) * 1000
            print(f"[latency] total response: {total:.0f}ms", flush=True)
        rid = self.response_id
        iid = self._output_item_id
        transcript = "".join(self._buffered_deltas)
        self._emit("response.output_audio_transcript.done", response_id=rid,
                   item_id=iid, output_index=0, content_index=0,
                   transcript=transcript)
        self._emit("response.output_audio.done", response_id=rid, item_id=iid,
                   output_index=0, content_index=0)
        self._emit("response.content_part.done", response_id=rid, item_id=iid,
                   output_index=0, content_index=0,
                   part={"type": "audio", "transcript": transcript})
        self._emit("response.output_item.done", response_id=rid, output_index=0,
                   item=self._output_item("completed", transcript=transcript))
        self._emit("response.done", response=self._response_object(status))
        self.response_id = None
        self._current_response_turn = None

    def _current_output_item_id(self) -> str:
        return getattr(self, "_output_item_id", _uuid("item_"))

    # --- object builders ---
    def _session_object(self) -> dict:
        return {
            "type": "realtime",
            "id": self.session_id,
            "object": "realtime.session",
            "model": self.model,
            "instructions": self.instructions,
            "output_modalities": self.output_modalities,
            "max_output_tokens": "inf",
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": PROTOCOL_RATE},
                    "transcription": {"model": self.cfg.stt_name,
                                      "language": "vi"},
                    "turn_detection": self.turn_detection,
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": PROTOCOL_RATE},
                    "voice": self._voice_name(),
                },
            },
        }

    def _voice_name(self) -> str:
        rev = {v: k for k, v in VOICE_MAP.items()}
        return rev.get(self.voice, self.voice)

    def _user_audio_item(self, item_id: str, transcript) -> dict:
        content = [{"type": "input_audio", "audio": "",
                    "transcript": transcript}]
        return {"type": "message", "id": item_id, "object": "realtime.item",
                "status": "completed", "role": "user", "content": content}

    def _output_item(self, status: str, transcript: str = "") -> dict:
        content = [{"type": "output_audio", "audio": "", "transcript": transcript}]
        return {"type": "message", "id": self._output_item_id,
                "object": "realtime.item", "status": status, "role": "assistant",
                "content": content}

    def _response_object(self, status: str, reason=None) -> dict:
        return {
            "id": self.response_id,
            "object": "realtime.response",
            "status": status,
            "status_details": {"type": status, "reason": reason, "error": None},
            "conversation_id": self.conversation_id,
            "output": [],
            "output_modalities": self.output_modalities,
            "max_output_tokens": "inf",
            "usage": getattr(self, "_usage", {"input_tokens": 0, "output_tokens": 0,
                                              "total_tokens": 0}),
        }
