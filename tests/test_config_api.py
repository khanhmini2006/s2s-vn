"""Test REST config: STT backend + model params qua GET/POST /v1/config."""

from fastapi.testclient import TestClient

from s2s_vn.api.openai_realtime.server import create_app
from s2s_vn.backend_registry import STT_BACKENDS, TTS_BACKENDS
from s2s_vn.s2s_pipeline import PipelineConfig


def client():
    return TestClient(create_app(PipelineConfig()))


def test_get_config_returns_stt_name_and_options():
    r = client().get("/v1/config")
    assert r.status_code == 200
    body = r.json()["stt"]
    assert body["name"] == "phowhisper-medium"
    # options = lựa chọn từ registry (nguồn sự thật), mỗi entry có name + label
    assert {o["name"] for o in body["options"]} == set(STT_BACKENDS)
    for o in body["options"]:
        assert o["label"]


def test_get_config_returns_model_params():
    r = client().get("/v1/config")
    assert r.status_code == 200
    body = r.json()
    # các param chỉnh được trên UI, đều phải xuất hiện
    assert body["llm"]["temperature"] == 0.0
    assert body["llm"]["max_tokens"] == 1024
    assert body["stt"]["beam_size"] == 1
    assert body["stt"]["compute_type"] == "int8_float16"
    assert body["tts"]["streaming"] is True
    assert body["vad"]["min_speech_ms"] == 500
    assert body["vad"]["speech_pad_ms"] == 500
    assert body["live_transcription"]["enabled"] is True


def test_get_config_returns_llm_model_options():
    r = client().get("/v1/config")
    llm = r.json()["llm"]
    options = llm["model_options"]
    # map theo backend — mỗi backend có danh sách model khuyến nghị
    assert isinstance(options, dict)
    assert "gemini-2.5-flash" in options["gemini"]
    assert "gpt-4.1-mini" in options["openai"]
    assert "Qwen/Qwen3-4B-Instruct-2507" in options["transformers"]
    # backend đang dùng phải có model hiện tại trong danh sách
    assert llm["model"] in options[llm["backend"]]


def test_get_config_returns_tts_voice_options():
    r = client().get("/v1/config")
    body = r.json()["tts"]
    # voice_options = danh sách giọng chọn được (preset + cloned), có nhãn thật
    assert body["denoise"] is True
    options = body["voice_options"]
    assert {o["name"] for o in options} >= {"Trúc Ly", "Minh Đức"}
    for o in options:
        assert o["name"]


def test_get_config_returns_tts_name_and_options():
    r = client().get("/v1/config")
    body = r.json()["tts"]
    assert body["name"] == "vieneu"
    # options = lựa chọn backend TTS từ registry (nguồn sự thật), mỗi entry có name + label
    assert {o["name"] for o in body["options"]} == set(TTS_BACKENDS)
    for o in body["options"]:
        assert o["label"]


def test_post_config_changes_tts_name():
    c = client()
    r = c.post("/v1/config", json={"tts": {"name": "mms-vie"}})
    assert r.status_code == 200
    assert c.get("/v1/config").json()["tts"]["name"] == "mms-vie"


def test_post_config_rejects_unknown_tts_name():
    r = client().post("/v1/config", json={"tts": {"name": "bogus"}})
    assert r.status_code == 400


def test_post_config_changes_model_params():
    c = client()
    r = c.post("/v1/config", json={
        "llm": {"temperature": 0.4, "max_tokens": 256},
        "stt": {"beam_size": 3, "compute_type": "float16"},
        "tts": {"streaming": False, "denoise": False},
        "vad": {"min_speech_ms": 300, "speech_pad_ms": 400},
        "live_transcription": {"enabled": False},
    })
    assert r.status_code == 200
    body = c.get("/v1/config").json()
    assert body["llm"]["temperature"] == 0.4
    assert body["llm"]["max_tokens"] == 256
    assert body["stt"]["beam_size"] == 3
    assert body["stt"]["compute_type"] == "float16"
    assert body["tts"]["streaming"] is False
    assert body["tts"]["denoise"] is False
    assert body["vad"]["min_speech_ms"] == 300
    assert body["vad"]["speech_pad_ms"] == 400
    assert body["live_transcription"]["enabled"] is False


def test_post_config_changes_tts_voice():
    c = client()
    r = c.post("/v1/config", json={"tts": {"voice": "Minh Đức"}})
    assert r.status_code == 200
    assert c.get("/v1/config").json()["tts"]["voice"] == "Minh Đức"


def test_post_config_rejects_unknown_voice():
    r = client().post("/v1/config", json={"tts": {"voice": "Không tồn tại"}})
    assert r.status_code == 400


def test_post_config_rejects_invalid_denoise():
    r = client().post("/v1/config", json={"tts": {"denoise": "yes"}})
    assert r.status_code == 400


def test_post_config_system_prompt_roundtrip():
    c = client()
    r = c.post("/v1/config", json={"llm": {"system_prompt": "Bạn là trợ lý bán hàng."}})
    assert r.status_code == 200
    assert c.get("/v1/config").json()["llm"]["system_prompt"] == "Bạn là trợ lý bán hàng."


def test_post_config_rejects_non_string_prompt():
    r = client().post("/v1/config", json={"llm": {"system_prompt": 123}})
    assert r.status_code == 400


def test_post_config_accepts_null_prompt():
    """Settings UI gửi null khi textarea rỗng — không được lỗi 400."""
    c = client()
    r = c.post("/v1/config", json={"llm": {"system_prompt": None}})
    assert r.status_code == 200


def test_config_api_key_roundtrip_and_not_echoed():
    c = client()
    # GET ban đầu: chưa có key (không có env trong test)
    assert c.get("/v1/config").json()["llm"]["api_key_set"] is False
    # POST đặt key
    r = c.post("/v1/config", json={"llm": {"api_key": "sk-test-123"}})
    assert r.status_code == 200
    body = c.get("/v1/config").json()["llm"]
    assert body["api_key_set"] is True
    assert "api_key" not in body  # không echo key ra GET


def test_post_config_rejects_non_string_key():
    r = client().post("/v1/config", json={"llm": {"api_key": 123}})
    assert r.status_code == 400


def test_post_config_changes_stt_name():
    c = client()
    r = c.post("/v1/config", json={"stt": {"name": "zipformer-vi-6000h"}})
    assert r.status_code == 200
    r = c.get("/v1/config")
    assert r.json()["stt"]["name"] == "zipformer-vi-6000h"


def test_post_config_rejects_unknown_stt():
    r = client().post("/v1/config", json={"stt": {"name": "bogus"}})
    assert r.status_code == 400


def test_post_config_rejects_non_dict_stt():
    r = client().post("/v1/config", json={"stt": "bogus"})
    assert r.status_code == 400


def test_post_config_rejects_invalid_param_values():
    c = client()
    # temperature ngoài khoảng
    assert c.post("/v1/config", json={"llm": {"temperature": 5}}).status_code == 400
    # beam_size không phải số nguyên ≥ 1
    assert c.post("/v1/config", json={"stt": {"beam_size": 0}}).status_code == 400
    assert c.post("/v1/config", json={"stt": {"beam_size": "x"}}).status_code == 400
    # compute_type ngoài danh sách
    assert c.post("/v1/config", json={"stt": {"compute_type": "fp8"}}).status_code == 400
    # streaming không phải bool
    assert c.post("/v1/config", json={"tts": {"streaming": "yes"}}).status_code == 400
    # max_tokens không phải int ≥ 1
    assert c.post("/v1/config", json={"llm": {"max_tokens": 0}}).status_code == 400
