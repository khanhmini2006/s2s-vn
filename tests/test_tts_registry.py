"""Test registry TTS: chọn đúng handler theo tên backend."""

import queue

import pytest

from s2s_vn.backend_registry import TTS_BACKENDS, get_tts_handler
from s2s_vn.s2s_pipeline import PipelineConfig


def make_cfg(tts_name):
    return PipelineConfig(tts_name=tts_name)


def test_default_backend_is_vieneu():
    q = queue.Queue()
    h = get_tts_handler(q, q, make_cfg("vieneu"))
    assert type(h).__name__ == "VieNeuTTSHandler"


def test_mms_backend_selected():
    q = queue.Queue()
    h = get_tts_handler(q, q, make_cfg("mms-vie"))
    assert type(h).__name__ == "MMSTTSHandler"
    assert h.model_name == "facebook/mms-tts-vie"


def test_piper_backend_selected():
    q = queue.Queue()
    h = get_tts_handler(q, q, make_cfg("piper-vie"))
    assert type(h).__name__ == "PiperTTSHandler"
    assert h.model_path.endswith("huongly.onnx")
    assert h.config_path.endswith("huongly.onnx.json")


def test_unknown_backend_raises():
    q = queue.Queue()
    with pytest.raises(ValueError):
        get_tts_handler(q, q, make_cfg("not-a-backend"))


def test_registry_lists_backends_with_metadata():
    """TTS_BACKENDS là nguồn sự thật: đủ model, có label."""
    assert set(TTS_BACKENDS) >= {"vieneu", "mms-vie", "piper-vie"}
    vn = TTS_BACKENDS["vieneu"]
    assert vn.label and vn.model_name is None
    mms = TTS_BACKENDS["mms-vie"]
    assert mms.label and mms.model_name == "facebook/mms-tts-vie"
    piper = TTS_BACKENDS["piper-vie"]
    assert piper.label and piper.model_name.endswith("huongly.onnx")
