"""Test persist config: POST /v1/config ghi file → khởi động sau đọc lại."""

import json

import pytest
from fastapi.testclient import TestClient

from s2s_vn.api.openai_realtime.server import create_app, load_config, save_config
from s2s_vn.s2s_pipeline import PipelineConfig


@pytest.fixture
def cfg_file(tmp_path):
    return str(tmp_path / "config.json")


def test_save_load_roundtrip(cfg_file):
    c = PipelineConfig(llm_backend="gemini", llm_model="gemini-3.6-flash",
                       llm_api_key="sk-test", tts_voice="Minh Đức",
                       stt_name="zipformer-vi-6000h", enable_live_transcription=False)
    save_config(c, cfg_file)
    loaded = load_config(cfg_file)
    assert loaded is not None
    assert loaded.llm_backend == "gemini"
    assert loaded.llm_model == "gemini-3.6-flash"
    assert loaded.llm_api_key == "sk-test"  # key được persist (file local, chmod 600)
    assert loaded.tts_voice == "Minh Đức"
    assert loaded.stt_name == "zipformer-vi-6000h"
    assert loaded.enable_live_transcription is False


def test_load_missing_returns_none(tmp_path):
    assert load_config(str(tmp_path / "none.json")) is None


def test_load_ignores_unknown_fields(cfg_file):
    with open(cfg_file, "w") as f:
        json.dump({"llm_backend": "openai", "khong_ton_tai": 123,
                   "llm_model": "gpt-4.1-mini"}, f)
    loaded = load_config(cfg_file)
    assert loaded is not None
    assert loaded.llm_backend == "openai"
    assert loaded.llm_model == "gpt-4.1-mini"


def test_post_config_writes_file(cfg_file, monkeypatch):
    import s2s_vn.api.openai_realtime.server as mod
    monkeypatch.setattr(mod, "CONFIG_FILE", cfg_file)
    app = create_app(PipelineConfig())
    c = TestClient(app)
    r = c.post("/v1/config", json={"llm": {"backend": "transformers", "model": "Qwen/Qwen3-8B"}})
    assert r.status_code == 200
    data = json.load(open(cfg_file))
    # save_config ghi dạng GROUP theo component (khớp GET/POST /v1/config)
    assert data["llm"]["backend"] == "transformers"
    assert data["llm"]["model"] == "Qwen/Qwen3-8B"
    # load lại phải khôi phục đúng (group → PipelineConfig phẳng)
    loaded = load_config(cfg_file)
    assert loaded is not None
    assert loaded.llm_backend == "transformers"
    assert loaded.llm_model == "Qwen/Qwen3-8B"


def test_create_app_uses_saved_file(cfg_file):
    save_config(PipelineConfig(llm_backend="gemini", llm_model="gemini-3.6-flash"), cfg_file)
    # create_app(None) → đọc từ file persist
    import s2s_vn.api.openai_realtime.server as mod
    mod.CONFIG_FILE = cfg_file
    app = create_app(None)
    c = app.state.pipeline_config
    assert c.llm_backend == "gemini"
    assert c.llm_model == "gemini-3.6-flash"
