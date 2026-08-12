"""Test RAG inject vào LLM (cả backend openai-compatible + transformers, không load model)."""

import queue

import pytest

from s2s_vn.LLM.llm_openai_compatible import OpenAICompatibleLLMHandler
from s2s_vn.LLM.transformers_llm import TransformersLLMHandler
from s2s_vn.pipeline.messages import GenerateResponseRequest


@pytest.fixture
def fake_rag(monkeypatch):
    """Giả rag_service: search trả 1 chunk cố định."""
    import s2s_vn.api.rag_service as mod

    class Fake:
        def search(self, query, top_k=3):
            return ["S2S-VN dùng VieNeu cho TTS."]

    monkeypatch.setattr(mod, "rag_service", Fake())
    return mod.rag_service


def make_req(text="s2s-vn dùng gì cho TTS?", tool_output=None):
    return GenerateResponseRequest(text=text, language_code="vi",
                                   turn_id=1,
                                   tool_call_id="c1" if tool_output else None,
                                   tool_output=tool_output)


def test_transformers_llm_no_direct_inject(fake_rag):
    """RAG đã chuyển sang tool search_knowledge — không inject trực tiếp vào prompt."""
    h = TransformersLLMHandler(queue.Queue(), queue.Queue())
    messages = h._build_messages(make_req())
    assert "Thông tin tham khảo" not in messages[0]["content"]


def test_transformers_llm_skips_rag_for_tool_output(fake_rag):
    h = TransformersLLMHandler(queue.Queue(), queue.Queue())
    messages = h._build_messages(make_req(tool_output="ok"))
    assert "Thông tin tham khảo" not in messages[0]["content"]


def test_openai_compatible_no_direct_inject(fake_rag):
    h = OpenAICompatibleLLMHandler(queue.Queue(), queue.Queue(), api_key="sk-test")
    messages = h._build_messages(make_req())
    assert "Thông tin tham khảo" not in messages[0]["content"]


def test_openai_compatible_skips_rag_for_tool_output(fake_rag):
    h = OpenAICompatibleLLMHandler(queue.Queue(), queue.Queue(), api_key="sk-test")
    messages = h._build_messages(make_req(tool_output="ok"))
    assert "Thông tin tham khảo" not in messages[0]["content"]
