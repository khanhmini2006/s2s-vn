"""Test endpoint /v1/test/pipeline — self-check luồng STT/LLM/RAG/TTS.

KHÔNG load model thật trong pytest (nặng/OOM) — monkeypatch từng tầng.
"""

from fastapi.testclient import TestClient

from s2s_vn.api.openai_realtime.server import create_app
from s2s_vn.s2s_pipeline import PipelineConfig


def test_pipeline_test_endpoint_returns_all_stages(monkeypatch):
    import s2s_vn.api.openai_realtime.server as mod

    monkeypatch.setattr(mod, "_pipeline_test_stt",
                        lambda cfg: {"ok": True, "detail": "xin chào bạn khỏe không"})
    monkeypatch.setattr(mod, "_pipeline_test_llm",
                        lambda cfg: {"ok": True, "detail": "Chào bạn!"})
    monkeypatch.setattr(mod, "_pipeline_test_rag",
                        lambda cfg: {"ok": True, "detail": "3 kết quả"})
    monkeypatch.setattr(mod, "_pipeline_test_tts",
                        lambda cfg: {"ok": True, "detail": "5.2s, 6 chunks"})

    c = TestClient(create_app(PipelineConfig()))
    r = c.post("/v1/test/pipeline")
    assert r.status_code == 200
    data = r.json()
    # đủ 4 tầng, mỗi tầng có ok + detail
    assert set(data) == {"stt", "llm", "rag", "tts"}
    for stage in ("stt", "llm", "rag", "tts"):
        assert data[stage]["ok"] is True
        assert data[stage]["detail"]


def test_pipeline_test_reports_failure(monkeypatch):
    import s2s_vn.api.openai_realtime.server as mod

    monkeypatch.setattr(mod, "_pipeline_test_stt",
                        lambda cfg: {"ok": False, "detail": "model không nhận diện"})
    monkeypatch.setattr(mod, "_pipeline_test_llm",
                        lambda cfg: {"ok": True, "detail": "ok"})
    monkeypatch.setattr(mod, "_pipeline_test_rag",
                        lambda cfg: {"ok": True, "detail": "0 kết quả"})
    monkeypatch.setattr(mod, "_pipeline_test_tts",
                        lambda cfg: {"ok": True, "detail": "ok"})

    c = TestClient(create_app(PipelineConfig()))
    r = c.post("/v1/test/pipeline")
    data = r.json()
    assert data["stt"]["ok"] is False
    assert "không nhận diện" in data["stt"]["detail"]
