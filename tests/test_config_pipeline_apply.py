"""Verify param settings THỰC SỰ xuống handler khi build pipeline.

POST /v1/config đổi hết param mọi thành phần → S2SPipeline(cfg) build →
inspect từng handler xem attribute khớp config (model chỉ nạp ở warmup,
__init__ nhẹ — test không load model thật).
"""

import pytest
from fastapi.testclient import TestClient

from s2s_vn.api.openai_realtime.server import create_app
from s2s_vn.s2s_pipeline import PipelineConfig, S2SPipeline


@pytest.fixture
def no_config_file(tmp_path, monkeypatch):
    import s2s_vn.api.openai_realtime.server as mod
    monkeypatch.setattr(mod, "CONFIG_FILE", str(tmp_path / "config.json"))


def test_all_params_reach_handlers(no_config_file):
    app = create_app(PipelineConfig())
    c = TestClient(app)
    r = c.post("/v1/config", json={
        "llm": {"backend": "transformers", "model": "Qwen/Qwen3-8B",
                "temperature": 0.3, "max_tokens": 512, "device": "cuda",
                "system_prompt": "Bạn là trợ lý."},
        "stt": {"name": "zipformer-vi-6000h", "beam_size": 2,
                "compute_type": "float16"},
        "tts": {"voice": "Minh Đức", "streaming": False, "denoise": False,
                "backend": "pytorch", "style": "tin_tuc",
                "temperature": 0.6, "max_chars": 128},
        "vad": {"threshold": 0.4, "min_silence_ms": 400,
                "min_speech_ms": 600, "speech_pad_ms": 700},
    })
    assert r.status_code == 200

    cfg = app.state.pipeline_config
    pipe = S2SPipeline(cfg)
    vad, stt, _notifier, llm, _proc, tts = pipe.handlers

    # --- VAD (index 0): cả 4 param ---
    assert vad.threshold == 0.4
    assert vad.min_silence_ms == 400
    assert vad.min_speech_ms == 600
    assert vad.speech_pad_ms == 700

    # --- STT (index 1): model + beam ---
    assert stt.model_name == "hynt/Zipformer-30M-RNNT-6000h"
    assert stt.beam_size == 2

    # --- LLM (index 3): model + mọi param ---
    assert llm.model_name == "Qwen/Qwen3-8B"
    assert llm.temperature == 0.3
    assert llm.max_tokens == 512
    assert llm.device == "cuda"
    assert llm.system_prompt == "Bạn là trợ lý."

    # --- TTS (index 5): voice + mọi param ---
    assert tts.voice == "Minh Đức"
    assert tts.streaming is False
    assert tts.denoise is False
    assert tts.backend == "pytorch"
    assert tts.style == "tin_tuc"
    assert tts.temperature == 0.6
    assert tts.max_chars == 128
