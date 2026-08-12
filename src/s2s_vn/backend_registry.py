"""Registry: chọn handler theo tên backend (thay vì if/elif rải rác)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base_handler import BaseHandler
from .LLM.llm_openai_compatible import OpenAICompatibleLLMHandler
from .LLM.transformers_llm import TransformersLLMHandler
from .STT.whisper_stt_handler import WhisperSTTHandler
from .STT.zipformer_stt_handler import ZipformerSTTHandler
from .TTS.mms_tts_handler import MMSTTSHandler
from .TTS.piper_tts_handler import PiperTTSHandler
from .TTS.vieneu_tts_handler import VieNeuTTSHandler

_PIPER_VOICE_DIR = Path(__file__).parent / "api" / "static" / "voices_piper"


@dataclass
class LLMConfig:
    backend: str = "openai"  # openai | hf-router | local | gemini | transformers
    model_name: str = "gpt-4.1-mini"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    device: str = "cuda"
    system_prompt: str | None = None  # None → default của handler


@dataclass
class STTConfig:
    """Đăng ký một STT backend — nguồn sự thật cho config API + UI."""

    name: str
    label: str  # hiển thị trong settings UI
    model_name: str
    subfolder: str | None = None  # chỉ faster-whisper (repo CT2) cần


# Đăng ký STT backend. Thêm model mới = thêm 1 entry (handler chọn theo family
# trong get_stt_handler: phowhisper-medium → WhisperSTTHandler, zipformer → ZipformerSTTHandler).
STT_BACKENDS: dict[str, STTConfig] = {
    "phowhisper-medium": STTConfig(
        name="phowhisper-medium",
        label="PhoWhisper-medium (faster-whisper CT2)",
        model_name="quocphu/PhoWhisper-ct2-FasterWhisper",
        subfolder="PhoWhisper-medium-ct2-fasterWhisper",
    ),
    "zipformer-vi-6000h": STTConfig(
        name="zipformer-vi-6000h",
        label="Zipformer-30M-RNNT (sherpa-onnx, VLSP2025)",
        model_name="hynt/Zipformer-30M-RNNT-6000h",
    ),
}


@dataclass
class TTSConfig:
    """Đăng ký một TTS backend — nguồn sự thật cho config API + UI."""

    name: str
    label: str
    model_name: str | None = None  # None → handler tự quyết (vieneu không cần model_name)


# Đăng ký TTS backend. Thêm model mới = thêm 1 entry (handler chọn theo tên
# trong get_tts_handler: vieneu → VieNeuTTSHandler, mms-vie → MMSTTSHandler).
TTS_BACKENDS: dict[str, TTSConfig] = {
    "vieneu": TTSConfig(
        name="vieneu",
        label="VieNeu-TTS v3 Turbo (voice clone, MIT)",
    ),
    "mms-vie": TTSConfig(
        name="mms-vie",
        label="Facebook MMS-TTS (VITS, CC-BY-NC-4.0 — chỉ phi thương mại)",
        model_name="facebook/mms-tts-vie",
    ),
    "piper-vie": TTSConfig(
        name="piper-vie",
        label="Piper TTS - giọng Huongly (ONNX, GPL-3.0-or-later)",
        model_name=str(_PIPER_VOICE_DIR / "huongly.onnx"),
    ),
}


# Model khuyến nghị theo backend — nguồn sự thật cho UI (settings model select)
LLM_MODEL_OPTIONS: dict[str, list[str]] = {
    "transformers": [
        "Qwen/Qwen3-8B",                    # đang dùng (4bit, tool calling ổn)
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-8B-Instruct-2507",
        "Qwen/Qwen3-1.7B-Instruct-2507",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",
    ],
    "openai": ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o",
               "gpt-4.1-nano", "gpt-5-mini"],
    "hf-router": ["Qwen/Qwen3-8B-Instruct-2507", "Qwen/Qwen3-4B-Instruct-2507",
                  "Qwen/Qwen3-14B", "Qwen/Qwen3-32B",
                  "meta-llama/Llama-3.1-8B-Instruct", "google/gemma-3-12b-it",
                  "deepseek-ai/DeepSeek-V3.2", "moonshotai/Kimi-K2-Instruct"],
    "gemini": ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite",
               "gemini-3.1-pro-preview", "gemini-2.5-flash", "gemini-2.5-pro",
               "gemini-2.0-flash"],
    "local": [],  # model phụ thuộc server vLLM/Ollama — gõ tay
}


# Tool RAG — LLM gọi khi cần tra cứu kho tài liệu (server tự xử lý trong realtime_service)
TOOL_SEARCH_KNOWLEDGE: dict = {
    "type": "function",
    "function": {
        "name": "search_knowledge",
        "description": "Tìm kiếm trong kho tài liệu nội bộ của công ty (đã upload qua tab Knowledge). "
                       "Dùng khi câu hỏi liên quan tài liệu, chính sách, báo cáo, số liệu nội bộ.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Câu hỏi hoặc từ khóa cần tra cứu"},
            },
            "required": ["query"],
        },
    },
}


# Provider → (base_url, api_key_env)
PROVIDERS: dict[str, tuple[str, str | None]] = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "hf-router": ("https://router.huggingface.co/v1", "HF_TOKEN"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    # "local" → base_url do người dùng set (vLLM/llama.cpp/Ollama), không key
}


def get_llm_handler(
    input_queue,
    output_queue,
    cfg: LLMConfig,
    text_out=None,
    cancel_scope=None,
) -> BaseHandler:
    """Trả LLM handler theo cfg.backend.

    - openai / hf-router / gemini: OpenAI-compatible remote
    - local: OpenAI-compatible tới base_url do cfg chỉ định (vLLM/llama.cpp/Ollama)
    """
    if cfg.backend in ("openai", "hf-router", "gemini"):
        base_url, key_env = PROVIDERS[cfg.backend]
        api_key = cfg.api_key or (key_env and __import__("os").environ.get(key_env))
        h = OpenAICompatibleLLMHandler(
            input_queue, output_queue,
            base_url=cfg.base_url or base_url,
            api_key=api_key,
            model_name=cfg.model_name,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            system_prompt=cfg.system_prompt,
            text_out=text_out,
            cancel_scope=cancel_scope,
        )
        h.set_tools([TOOL_SEARCH_KNOWLEDGE])
        return h
    if cfg.backend == "local":
        # base_url user set, ví dụ http://localhost:8000/v1 (vLLM) hay
        # http://localhost:11434/v1 (Ollama), http://localhost:8080/v1 (llama.cpp)
        h = OpenAICompatibleLLMHandler(
            input_queue, output_queue,
            base_url=cfg.base_url or "http://localhost:8000/v1",
            api_key=cfg.api_key or "sk-local",
            model_name=cfg.model_name,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            system_prompt=cfg.system_prompt,
            text_out=text_out,
            cancel_scope=cancel_scope,
        )
        h.set_tools([TOOL_SEARCH_KNOWLEDGE])
        return h
    if cfg.backend == "transformers":
        # local Transformers (CUDA/CPU) — không cần server, chia GPU pipeline
        h = TransformersLLMHandler(
            input_queue, output_queue,
            model_name=cfg.model_name,
            device=cfg.device,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            system_prompt=cfg.system_prompt,
            cancel_scope=cancel_scope,
            text_out=text_out,
        )
        h.set_tools([TOOL_SEARCH_KNOWLEDGE])
        return h
    raise ValueError(f"LLM backend không hỗ trợ: {cfg.backend}")


def get_stt_handler(
    input_queue,
    output_queue,
    cfg,
    text_out=None,
    cancel_scope=None,
) -> BaseHandler:
    """Trả STT handler theo cfg.stt_name (cfg: PipelineConfig — không import
    để tránh circular import pipeline.s2s_pipeline ↔ registry).

    Backend đăng ký trong STT_BACKENDS (thêm model = thêm 1 entry).
    """
    backend = STT_BACKENDS.get(cfg.stt_name)
    if backend is None:
        raise ValueError(f"STT backend không hỗ trợ: {cfg.stt_name}")
    if backend.name == "phowhisper-medium":
        return WhisperSTTHandler(
            input_queue, output_queue,
            model_name=backend.model_name,
            model_subfolder=backend.subfolder,
            beam_size=cfg.stt_beam_size,
            compute_type=cfg.stt_compute_type,
            text_out=text_out,
            enable_live_transcription=cfg.enable_live_transcription,
            cancel_scope=cancel_scope,
        )
    if backend.name == "zipformer-vi-6000h":
        return ZipformerSTTHandler(
            input_queue, output_queue,
            model_name=backend.model_name,
            beam_size=cfg.stt_beam_size,
            text_out=text_out,
            enable_live_transcription=cfg.enable_live_transcription,
            cancel_scope=cancel_scope,
        )
    # family mới (thêm backend STT): thêm 1 nhánh ở đây
    raise ValueError(f"STT backend chưa có handler: {backend.name}")


def get_tts_handler(
    input_queue,
    output_queue,
    cfg,
    cancel_scope=None,
) -> BaseHandler:
    """Trả TTS handler theo cfg.tts_name (cfg: PipelineConfig — không import
    để tránh circular import pipeline.s2s_pipeline ↔ registry).

    Backend đăng ký trong TTS_BACKENDS (thêm model = thêm 1 entry).
    """
    backend = TTS_BACKENDS.get(cfg.tts_name)
    if backend is None:
        raise ValueError(f"TTS backend không hỗ trợ: {cfg.tts_name}")
    if backend.name == "vieneu":
        return VieNeuTTSHandler(
            input_queue, output_queue,
            voice=cfg.tts_voice,
            output_sample_rate=cfg.sample_rate,
            streaming=cfg.tts_streaming,
            denoise=cfg.tts_denoise,
            backend=cfg.tts_backend,
            style=cfg.tts_style,
            temperature=cfg.tts_temperature,
            max_chars=cfg.tts_max_chars,
            cancel_scope=cancel_scope,
        )
    if backend.name == "mms-vie":
        return MMSTTSHandler(
            input_queue, output_queue,
            model_name=backend.model_name,
            output_sample_rate=cfg.sample_rate,
            cancel_scope=cancel_scope,
        )
    if backend.name == "piper-vie":
        return PiperTTSHandler(
            input_queue, output_queue,
            model_path=backend.model_name,
            config_path=backend.model_name + ".json",
            output_sample_rate=cfg.sample_rate,
            cancel_scope=cancel_scope,
        )
    # family mới (thêm backend TTS): thêm 1 nhánh ở đây
    raise ValueError(f"TTS backend chưa có handler: {backend.name}")
