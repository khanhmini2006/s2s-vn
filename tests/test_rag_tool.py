"""Test RAG qua Tool Calling (không load model): tool đăng ký cho LLM + server tự xử lý."""

import json
import queue

import pytest

from s2s_vn.LLM.llm_openai_compatible import OpenAICompatibleLLMHandler
from s2s_vn.backend_registry import LLMConfig, TOOL_SEARCH_KNOWLEDGE, get_llm_handler
from s2s_vn.LLM.transformers_llm import TransformersLLMHandler
from s2s_vn.pipeline.events import FunctionCallEvent
from s2s_vn.pipeline.messages import GenerateResponseRequest
from s2s_vn.s2s_pipeline import PipelineConfig


# --- Tool registration qua registry ---

def test_llm_handlers_register_search_tool():
    q = queue.Queue()
    openai_h = get_llm_handler(q, q, LLMConfig(backend="local", base_url="http://x/v1"))
    assert openai_h.tools == [TOOL_SEARCH_KNOWLEDGE]

    tf_h = get_llm_handler(q, q, LLMConfig(backend="transformers"))
    assert tf_h._tools == [TOOL_SEARCH_KNOWLEDGE]


def test_tool_schema_is_openai_compatible():
    assert TOOL_SEARCH_KNOWLEDGE["type"] == "function"
    fn = TOOL_SEARCH_KNOWLEDGE["function"]
    assert fn["name"] == "search_knowledge"
    assert "query" in fn["parameters"]["properties"]


# --- Server-side tool execution trong RealtimeService ---

def test_server_executes_search_knowledge_tool(monkeypatch):
    from s2s_vn.api.openai_realtime.realtime_service import RealtimeService

    class FakeRag:
        def search(self, query, top_k=3):
            return ["Chính sách bảo hành: 12 tháng."]

    import s2s_vn.api.rag_service as mod
    monkeypatch.setattr(mod, "rag_service", FakeRag())

    events = []
    svc = RealtimeService(PipelineConfig(), on_event=lambda ev: events.append(ev))
    # spy put — LLM handler thread consume text_prompt liên tục nên không get_nowait được
    captured = []
    monkeypatch.setattr(svc.pipeline.queues.text_prompt, "put", captured.append)
    try:
        svc._handle_pipeline_event(FunctionCallEvent(
            name="search_knowledge", arguments='{"query": "chính sách bảo hành"}',
            call_id="call_1", turn_id=7))

        # server tự đẩy tool_output vào pipeline → LLM tiếp tục (không cần client)
        req = captured[0]
        assert isinstance(req, GenerateResponseRequest)
        assert req.tool_call_id == "call_1"
        assert "12 tháng" in req.tool_output
        assert "Chính sách bảo hành" in req.tool_output

        # vẫn emit function_call cho client xem (transparency)
        assert any(e["type"] == "response.function_call_arguments.done" for e in events)
    finally:
        svc.stop()


def test_server_handles_empty_knowledge_base(monkeypatch):
    from s2s_vn.api.openai_realtime.realtime_service import RealtimeService

    class FakeRag:
        def search(self, query, top_k=3):
            return []

    import s2s_vn.api.rag_service as mod
    monkeypatch.setattr(mod, "rag_service", FakeRag())

    svc = RealtimeService(PipelineConfig(), on_event=lambda ev: None)
    captured = []
    monkeypatch.setattr(svc.pipeline.queues.text_prompt, "put", captured.append)
    try:
        svc._handle_pipeline_event(FunctionCallEvent(
            name="search_knowledge", arguments='{"query": "gì đó"}',
            call_id="call_2", turn_id=8))
        req = captured[0]
        assert req.tool_output and "không" in req.tool_output.lower()
    finally:
        svc.stop()


def test_other_tools_still_forwarded_to_client(monkeypatch):
    """Tool không phải search_knowledge → không tự xử lý, chỉ emit cho client."""
    from s2s_vn.api.openai_realtime.realtime_service import RealtimeService

    svc = RealtimeService(PipelineConfig(), on_event=lambda ev: None)
    captured = []
    monkeypatch.setattr(svc.pipeline.queues.text_prompt, "put", captured.append)
    try:
        svc._handle_pipeline_event(FunctionCallEvent(
            name="get_weather", arguments='{"city": "Hanoi"}',
            call_id="call_3", turn_id=9))
        assert captured == []  # không tự đẩy tool_output
        assert svc._fc is not None  # giữ để client loop
    finally:
        svc.stop()
