"""Test registry STT: chọn đúng handler theo tên backend."""

import queue

import pytest

from s2s_vn.backend_registry import STT_BACKENDS, get_stt_handler
from s2s_vn.s2s_pipeline import PipelineConfig


def make_cfg(stt_name):
    return PipelineConfig(stt_name=stt_name)


def test_default_backend_is_whisper():
    q = queue.Queue()
    h = get_stt_handler(q, q, make_cfg("phowhisper-medium"))
    assert type(h).__name__ == "WhisperSTTHandler"


def test_zipformer_backend_selected():
    q = queue.Queue()
    h = get_stt_handler(q, q, make_cfg("zipformer-vi-6000h"))
    assert type(h).__name__ == "ZipformerSTTHandler"


def test_unknown_backend_raises():
    q = queue.Queue()
    with pytest.raises(ValueError):
        get_stt_handler(q, q, make_cfg("not-a-backend"))


def test_registry_lists_backends_with_metadata():
    """STT_BACKENDS là nguồn sự thật: đủ model, có label + model_name."""
    assert set(STT_BACKENDS) >= {"phowhisper-medium", "zipformer-vi-6000h"}
    pw = STT_BACKENDS["phowhisper-medium"]
    assert pw.label and pw.model_name == "quocphu/PhoWhisper-ct2-FasterWhisper"
    assert pw.subfolder == "PhoWhisper-medium-ct2-fasterWhisper"
    zf = STT_BACKENDS["zipformer-vi-6000h"]
    assert zf.label
    assert zf.model_name == "hynt/Zipformer-30M-RNNT-6000h"
    assert zf.subfolder is None

