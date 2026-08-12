"""LLM handler dùng OpenAI-compatible protocol (/v1/chat/completions).

Hỗ trợ mọi provider nói OpenAI-compatible: OpenAI, HF Inference Providers,
vLLM, llama.cpp, Ollama, Gemini API (generativelanguage.googleapis.com/v1beta/openai).

Truyền base_url + api_key + model_name. Streaming theo từng text delta.
"""

from __future__ import annotations

import queue

from ..base_handler import BaseHandler
from ..pipeline.events import AssistantTextEvent, FunctionCallEvent, TokenUsageEvent
from ..pipeline.messages import (
    EndOfResponse,
    GenerateResponseRequest,
    LLMResponseChunk,
)


class OpenAICompatibleLLMHandler(BaseHandler):
    """GenerateResponseRequest → LLMResponseChunk/EndOfResponse."""

    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        name: str = "llm",
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        model_name: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stream: bool = True,
        text_out: queue.Queue | None = None,
        cancel_scope=None,
        system_prompt: str | None = None,
    ):
        super().__init__(input_queue, output_queue, name)
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.text_out = text_out
        self.cancel_scope = cancel_scope
        from .prompts import OPENAI_COMPATIBLE_PROMPT
        self.system_prompt = system_prompt or OPENAI_COMPATIBLE_PROMPT
        self._client = None
        # tool calling
        self.tools: list[dict] = []
        self._history: list[dict] = []  # messages conversation (system+user+assistant+tool)
        self.max_history = 20  # giới hạn messages tránh memory leak
        self._call_id_counter = 0

    def set_tools(self, tools: list[dict]) -> None:
        self.tools = tools or []

    def warmup(self) -> None:
        from openai import OpenAI

        if not self.api_key:
            raise RuntimeError(
                f"LLM backend cần api_key (provider={self.base_url}). "
                "Set qua config hoặc env."
            )
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _build_messages(self, item: GenerateResponseRequest) -> list[dict]:
        """Dựng messages: history + turn hiện tại (text hoặc tool_output)."""

        # RAG qua tool search_knowledge (server tự xử lý) — không inject trực tiếp
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self._history)
        if item.tool_call_id and item.tool_output is not None:
            messages.append({
                "role": "tool",
                "tool_call_id": item.tool_call_id,
                "content": item.tool_output,
            })
        else:
            messages.append({"role": "user", "content": item.text})
        return messages

    def process(self, item):
        if not isinstance(item, GenerateResponseRequest):
            return None
        if self._client is None:
            self.warmup()

        # lưu user text vào history (tool round thì không — Gemini yêu cầu
        # function call turn đứng ngay sau user/function response turn)
        if not (item.tool_call_id and item.tool_output is not None):
            self._history.append({"role": "user", "content": item.text})

        messages = self._build_messages(item)
        kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if self.tools:
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = "auto"
        if self.stream:
            return self._handle_stream(item, kwargs)
        return self._handle_non_stream(item, kwargs)

    def _handle_stream(self, item, kwargs):
        outputs = []
        usage = {"input": 0, "output": 0}
        tool_calls_acc = {}  # index → {name, args}
        gen_start = self.cancel_scope.current() if self.cancel_scope else 0
        stream = self._client.chat.completions.create(stream=True, **kwargs)
        for chunk in stream:
            if not chunk.choices:
                u = getattr(chunk, "usage", None)
                if u:
                    usage["input"] = getattr(u, "prompt_tokens", 0)
                    usage["output"] = getattr(u, "completion_tokens", 0)
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    idx = tc.index
                    acc = tool_calls_acc.setdefault(
                        idx, {"name": "", "arguments": "", "id": tc.id or "",
                              "thought_signature": None})
                    if tc.function.name:
                        acc["name"] += tc.function.name
                    if tc.function.arguments:
                        acc["arguments"] += tc.function.arguments
                    # Gemini 3.x: thought_signature nằm trong extra_content.google
                    ec = getattr(tc, "extra_content", None)
                    if isinstance(ec, dict) and isinstance(ec.get("google"), dict):
                        ts = ec["google"].get("thought_signature")
                        if ts:
                            acc["thought_signature"] = ts
                continue
            text = getattr(delta, "content", None)
            if text:
                # barge-in: dừng stream nếu cancel_scope thay đổi
                if self.cancel_scope and self.cancel_scope.current() != gen_start:
                    stream.close()
                    break
                outputs.append(LLMResponseChunk(text_delta=text, turn_id=item.turn_id))
        # finish reason / tool calls
        if tool_calls_acc:
            for idx, acc in tool_calls_acc.items():
                call_id = acc["id"] or f"call_{item.turn_id}_{idx}"
                self._call_id_counter += 1
                ev = FunctionCallEvent(
                    name=acc["name"], arguments=acc["arguments"],
                    call_id=call_id, turn_id=item.turn_id)
                if self.text_out:
                    self.text_out.put(ev)
                # lưu history: assistant message với tool_call
                # Gemini 3.x yêu cầu thought_signature (từ extra_content) khi gửi lại
                tc_dict = {
                    "id": call_id, "type": "function",
                    "function": {"name": acc["name"], "arguments": acc["arguments"]},
                }
                if acc.get("thought_signature"):
                    tc_dict["extra_content"] = {
                        "google": {"thought_signature": acc["thought_signature"]}}
                self._history.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tc_dict],
                })
            outputs.append(EndOfResponse(turn_id=item.turn_id))
            if self.text_out:
                self.text_out.put(TokenUsageEvent(
                    input_tokens=usage["input"], output_tokens=usage["output"],
                    turn_id=item.turn_id))
            return outputs
        # text-only response → lưu history
        full_text = "".join(o.text_delta for o in outputs)
        self._history.append({"role": "assistant", "content": full_text})
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]
        outputs.append(EndOfResponse(turn_id=item.turn_id))
        if self.text_out:
            self.text_out.put(TokenUsageEvent(
                input_tokens=usage["input"], output_tokens=usage["output"],
                turn_id=item.turn_id))
        return outputs

    def _handle_non_stream(self, item, kwargs):
        resp = self._client.chat.completions.create(stream=False, **kwargs)
        text = resp.choices[0].message.content or ""
        outputs = [
            LLMResponseChunk(text_delta=text, turn_id=item.turn_id),
            EndOfResponse(turn_id=item.turn_id),
        ]
        if self.text_out:
            usage = resp.usage
            self.text_out.put(TokenUsageEvent(
                input_tokens=getattr(usage, "prompt_tokens", 0),
                output_tokens=getattr(usage, "completion_tokens", 0),
                turn_id=item.turn_id))
        return outputs
