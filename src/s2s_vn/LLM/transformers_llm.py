"""LLM handler local qua Transformers (CUDA/CPU).

Dùng AutoModelForCausalLM + chat template. Streaming text, tool calling
giản lược: parse tool_calls từ response nếu model hỗ trợ.
"""

from __future__ import annotations

import queue

from ..base_handler import BaseHandler
from ..pipeline.events import FunctionCallEvent, TokenUsageEvent
from ..pipeline.messages import (
    EndOfResponse,
    GenerateResponseRequest,
    LLMResponseChunk,
)


class TransformersLLMHandler(BaseHandler):
    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        name: str = "llm",
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        device: str = "cuda",
        torch_dtype: str = "auto",
        max_tokens: int = 80,
        temperature: float = 0.6,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.1,
        load_in_4bit: bool = True,
        enable_thinking: bool = False,  # voice: tắt <think> blocks
        cancel_scope=None,
        text_out: queue.Queue | None = None,
        system_prompt: str | None = None,
    ):
        super().__init__(input_queue, output_queue, name)
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.load_in_4bit = load_in_4bit
        self.enable_thinking = enable_thinking
        self.cancel_scope = cancel_scope
        self.text_out = text_out
        from .prompts import TRANSFORMERS_PROMPT
        self.system_prompt = system_prompt or TRANSFORMERS_PROMPT
        self._model = None
        self._tokenizer = None
        self._history: list[dict] = []
        self.max_history = 20  # giới hạn messages tránh memory leak
        self._call_id_counter = 0
        self._tools: list[dict] = []

    def set_tools(self, tools: list[dict]) -> None:
        self._tools = tools or []

    def warmup(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        load_kwargs = {}
        if "AWQ" in self.model_name:
            # model đã lượng tử AWQ sẵn (Qwen3-8B-AWQ) — dùng AwqConfig, không BitsAndBytes
            from transformers import AwqConfig

            load_kwargs["quantization_config"] = AwqConfig(bits=4, backend="autoawq")
            load_kwargs["device_map"] = "auto"
        elif self.load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = getattr(torch, str(self.torch_dtype)) \
                if self.torch_dtype != "auto" else "auto"
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **load_kwargs)
        if self.device == "cuda" and "device_map" not in load_kwargs:
            self._model.to("cuda")
        self._model.eval()

    def _build_messages(self, item: GenerateResponseRequest) -> list[dict]:
        # RAG qua tool search_knowledge (server tự xử lý) — không inject trực tiếp
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self._history)
        if item.tool_call_id and item.tool_output is not None:
            messages.append({"role": "tool", "tool_call_id": item.tool_call_id,
                             "content": item.tool_output})
        else:
            messages.append({"role": "user", "content": item.text})
        return messages

    def process(self, item):
        if not isinstance(item, GenerateResponseRequest):
            return None
        if self._model is None:
            self.warmup()
        return self._generate(item)

    def _generate(self, item: GenerateResponseRequest):
        import threading

        import torch
        from transformers import TextIteratorStreamer

        messages = self._build_messages(item)
        # append user text vào history (nếu không phải tool round)
        if not (item.tool_call_id and item.tool_output is not None):
            self._history.append({"role": "user", "content": item.text})

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            tools=self._tools if self._tools else None)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        gen_kwargs = dict(
            max_new_tokens=self.max_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature if self.temperature > 0 else None,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        # Một số Qwen3-8B (không phải 2507) không nhận enable_thinking kwarg và có
        # <think> mặc định — lọc think block ở dưới thay vì dựa vào kwarg.
        if self.temperature <= 0:
            gen_kwargs.pop("temperature")

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True,
                                        skip_special_tokens=True)
        gen_thread = threading.Thread(
            target=self._model.generate,
            kwargs={**inputs, **gen_kwargs, "streamer": streamer},
            daemon=True,
        )
        gen_thread.start()

        gen_start = self.cancel_scope.current() if self.cancel_scope else 0
        outputs = []
        parts = []
        in_think = False
        for piece in streamer:
            if self._cancelled(gen_start):
                break
            parts.append(piece)
            # bỏ <think> block (Qwen3-8B thinking) + <tool_call> tags — không emit ra transcript
            if "<think" in piece:
                in_think = True
                continue
            if "</think" in piece:
                in_think = False
                continue
            if in_think or "<tool_call" in piece or "</tool_call" in piece:
                continue
            outputs.append(LLMResponseChunk(text_delta=piece, turn_id=item.turn_id))

        full = "".join(parts).strip()
        # bỏ <think> block trước khi parse tool + lưu history (voice không cần suy nghĩ)
        import re as _re
        full = _re.sub(r"<think>.*?</think>", "", full, flags=_re.DOTALL).strip()
        # parse tool_calls từ output (Qwen3 gen <tool_call> tags)
        had_tool_call = False
        if self._tools:
            had_tool_call = self._parse_and_emit_tool_calls(full, item.turn_id)
        # nếu là tool round → append tool message vào history trước assistant
        if item.tool_call_id and item.tool_output is not None:
            self._history.append({"role": "tool", "tool_call_id": item.tool_call_id,
                                  "content": item.tool_output})
        if had_tool_call:
            # turn chỉ gọi tool — không emit text (Qwen3 lẫn JSON tool call vào text)
            outputs = [o for o in outputs if not isinstance(o, LLMResponseChunk)]
            if full:
                self._history.append({"role": "assistant", "content": full})
        elif full:
            self._history.append({"role": "assistant", "content": full})
        # truncate history cũ (giữ messages gần nhất)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]
        outputs.append(EndOfResponse(turn_id=item.turn_id))
        if self.text_out:
            self.text_out.put(TokenUsageEvent(
                input_tokens=len(inputs["input_ids"][0]),
                output_tokens=len(full),
                turn_id=item.turn_id))
        return outputs

    def _cancelled(self, gen_start: int) -> bool:
        if self.cancel_scope is None:
            return False
        return self.cancel_scope.current() != gen_start

    def _parse_and_emit_tool_calls(self, text: str, turn_id: int) -> bool:
        """Parse <tool_call> tags từ Qwen3 output → emit FunctionCallEvent.

        Trả True nếu có ≥1 tool call (để caller bỏ text rác của turn).
        """
        import json as _json
        import re as _re

        # greedy `.*}` — JSON lồng nhau (arguments chứa `}`) làm non-greedy vỡ
        pattern = r"<tool_call>\s*(\{.*\})\s*"
        found = False
        for m in _re.finditer(pattern, text, _re.DOTALL):
            try:
                call = _json.loads(m.group(1))
                name = call.get("name", "")
                args = _json.dumps(call.get("arguments", {}), ensure_ascii=False)
                self._call_id_counter += 1
                call_id = f"call_{self._call_id_counter}"
                if self.text_out:
                    self.text_out.put(FunctionCallEvent(
                        name=name, arguments=args,
                        call_id=call_id, turn_id=turn_id))
                self._history.append({
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": call_id, "type": "function",
                                    "function": {"name": name, "arguments": args}}],
                })
                found = True
            except Exception:
                continue
        return found
