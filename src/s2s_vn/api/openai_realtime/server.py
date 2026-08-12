"""FastAPI server: WS endpoint OpenAI Realtime-compatible."""

import dataclasses
import json
import logging
import os
import queue
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket

from ...backend_registry import STT_BACKENDS, TTS_BACKENDS
from ...s2s_pipeline import PipelineConfig
from .webrtc_router import setup_webrtc
from .websocket_router import setup_realtime

from typing import Optional

# --- Config persist: bấm Lưu settings → ghi file → phiên chạy sau tự dùng lại ---
CONFIG_FILE = os.environ.get("S2S_CONFIG_FILE", "config.json")
_CONFIG_FIELDS = [f.name for f in dataclasses.fields(PipelineConfig)]

# File config dạng GROUP theo component (khớp GET/POST /v1/config): group.key → field PipelineConfig
_GROUP_MAP: dict[str, dict[str, str]] = {
    "audio": {"chunk_size": "chunk_size", "sample_rate": "sample_rate"},
    "stt": {"name": "stt_name", "beam_size": "stt_beam_size",
            "compute_type": "stt_compute_type"},
    "llm": {"backend": "llm_backend", "model": "llm_model", "api_key": "llm_api_key",
            "base_url": "llm_base_url", "temperature": "llm_temperature",
            "max_tokens": "llm_max_tokens", "device": "llm_device",
            "system_prompt": "llm_system_prompt"},
    "tts": {"name": "tts_name", "voice": "tts_voice", "streaming": "tts_streaming",
            "denoise": "tts_denoise", "backend": "tts_backend", "style": "tts_style",
            "temperature": "tts_temperature", "max_chars": "tts_max_chars"},
    "vad": {"threshold": "vad_threshold", "min_silence_ms": "min_silence_ms",
            "min_speech_ms": "min_speech_ms", "speech_pad_ms": "speech_pad_ms"},
    "live_transcription": {"enabled": "enable_live_transcription",
                           "update_interval": "live_transcription_update_interval",
                           "speculative_reopen_ms": "speculative_reopen_ms"},
}


def save_config(cfg: PipelineConfig, path: str = CONFIG_FILE) -> None:
    """Ghi PipelineConfig ra JSON dạng GROUP theo component (chứa luôn API key, chmod 600)."""
    data = {}
    for group, mapping in _GROUP_MAP.items():
        data[group] = {key: getattr(cfg, field) for key, field in mapping.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def load_config(path: str = CONFIG_FILE) -> Optional[PipelineConfig]:
    """Đọc config persist; không tồn tại/hỏng → None (dùng default).

    Hỗ trợ 2 dạng:
    - GROUP: {"llm": {"backend": ...}, "stt": {"name": ...}, ...} (khớp GET/POST /v1/config)
    - PHẲNG: {"llm_backend": ..., "stt_name": ...} (dạng cũ — backward compatible)
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        flat: dict = {}
        for group, mapping in _GROUP_MAP.items():
            g = data.get(group)
            if isinstance(g, dict):
                for key, field in mapping.items():
                    if key in g:
                        flat[field] = g[key]
        # dạng phẳng cũ: field trực tiếp thuộc PipelineConfig
        for k, v in data.items():
            if k in _CONFIG_FIELDS and k not in flat:
                flat[k] = v
        return PipelineConfig(**flat) if flat else None
    except Exception:
        return None


# --- Test luồng: self-check STT/LLM/RAG/TTS (panel trên demo UI) ---
# Chỉ 1 test chạy 1 lúc: mỗi lần test tải model thật (Qwen3-8B 4bit ~6GB VRAM) —
# chạy song song 2 test = 2 bản model = OOM (RTX 5070 Ti 16GB).
_test_lock = threading.Lock()

def _clear_vram():
    """Trả VRAM về OS sau khi model test bị GC (torch cache giữ lại nếu không)."""
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

def _pipeline_test_stt(c):
    """STT decode audio mẫu có sẵn → text (nếu model tải + nhận diện được)."""
    import numpy as np
    import scipy.signal

    from ...backend_registry import get_stt_handler
    from ...pipeline.messages import VADAudio
    try:
        pcm24 = open(Path(__file__).parent.parent.parent.parent.parent / "scripts" / "test_input_24k.pcm", "rb").read()
        a16 = scipy.signal.resample_poly(
            np.frombuffer(pcm24, dtype=np.int16).astype(np.float32) / 32768.0, 16000, 24000)
        pcm16 = (a16 * 32767).astype(np.int16).tobytes()
        iq, oq = queue.Queue(), queue.Queue()
        h = get_stt_handler(iq, oq, c)
        out = h.process(VADAudio(audio=pcm16, mode="final", turn_id=1, sample_rate=16000))
        if out and out.text:
            return {"ok": True, "detail": out.text}
        return {"ok": False, "detail": "STT trả text rỗng"}
    except Exception as e:
        return {"ok": False, "detail": f"STT lỗi: {e!r}"}

def _pipeline_test_llm(c):
    """LLM ping 1 câu ngắn → text (model phải tải + trả lời)."""
    from ...backend_registry import LLMConfig, get_llm_handler
    from ...pipeline.messages import (EndOfResponse, GenerateResponseRequest,
                                     LLMResponseChunk)
    h = None
    try:
        iq, oq = queue.Queue(), queue.Queue()
        h = get_llm_handler(iq, oq, LLMConfig(
            backend=c.llm_backend, model_name=c.llm_model,
            api_key=c.llm_api_key, base_url=c.llm_base_url,
            temperature=0.0, max_tokens=256, device=c.llm_device,
            system_prompt=c.llm_system_prompt))
        outs = h.process(GenerateResponseRequest(text="Xin chào", language_code="vi", turn_id=1))
        text = "".join(o.text_delta for o in outs if isinstance(o, LLMResponseChunk))
        if text:
            return {"ok": True, "detail": text[:80]}
        return {"ok": False, "detail": "LLM trả text rỗng (thiếu key? model không tải?)"}
    except Exception as e:
        return {"ok": False, "detail": f"LLM lỗi: {e!r}"}
    finally:
        # model test là bản riêng (8B 4bit ~6GB) — phải trả VRAM, không thì
        # mỗi lần bấm Test luồng tích lũy thêm 1 bản model → OOM
        del h
        _clear_vram()

def _pipeline_test_rag(c):
    """RAG: search thử — chạy được là pass (kho rỗng vẫn pass)."""
    try:
        from ...api.rag_service import rag_service
        results = rag_service.search("test", top_k=1)
        return {"ok": True, "detail": f"{len(results)} kết quả (kho tri thức)"}
    except Exception as e:
        return {"ok": False, "detail": f"RAG lỗi: {e!r}"}

def _pipeline_test_tts(c):
    """TTS sinh 1 câu ngắn → audio chunks > 0."""
    from ...backend_registry import get_tts_handler
    from ...pipeline.messages import TTSInput
    try:
        iq, oq = queue.Queue(), queue.Queue()
        h = get_tts_handler(iq, oq, c)
        out = h.process(TTSInput(text="Xin chào, đây là bài kiểm tra giọng nói.", turn_id=1))
        chunks = 0
        while not oq.empty():
            oq.get(); chunks += 1
        if chunks > 0:
            return {"ok": True, "detail": f"{chunks} audio chunks"}
        return {"ok": False, "detail": "TTS không sinh audio"}
    except Exception as e:
        return {"ok": False, "detail": f"TTS lỗi: {e!r}"}


def create_app(cfg: Optional[PipelineConfig] = None) -> FastAPI:
    from pathlib import Path

    from fastapi import Body
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    app = FastAPI(title="s2s-vn", version="0.1.0")
    # cfg → config persist (config.json) → default
    app.state.pipeline_config = cfg or load_config(CONFIG_FILE) or PipelineConfig()
    if not cfg and os.path.exists(CONFIG_FILE):
        logging.getLogger("s2s.api").info(f"Đã nạp config persist từ {CONFIG_FILE}")

    # --- Auth ---
    API_KEY = os.environ.get("S2S_API_KEY", "")  # set để bật auth; trống = không auth (demo)

    def check_auth(header_auth: str | None, query_key: str | None = None) -> None:
        if not API_KEY:
            return  # auth disabled
        token = None
        if header_auth and header_auth.startswith("Bearer "):
            token = header_auth[7:]
        elif query_key:
            token = query_key
        if token != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    setup_realtime(app, app.state.pipeline_config, check_auth=check_auth)
    setup_webrtc(app, app.state.pipeline_config, check_auth=check_auth)

    # --- Metrics state ---
    app.state.metrics = {
        "sessions_total": 0,
        "sessions_active": 0,
        "requests_total": 0,
        "started_at": __import__("time").time(),
    }

    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/v1/health")
    def health_check():
        return {"status": "ok", "uptime_s": round(__import__("time").time() - app.state.metrics["started_at"], 1)}

    @app.get("/v1/usage")
    def usage():
        return dict(app.state.metrics)

    @app.get("/")
    def home():
        # trang chủ — fallback JSON nếu thiếu index.html
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"status": "ok", "endpoint": "ws://<host>/v1/realtime",
                "demo": "/demo", "settings": "/settings", "voice": "/voice",
                "knowledge": "/knowledge"}

    @app.get("/demo")
    def demo():
        return FileResponse(static_dir / "demo.html")

    @app.get("/webrtc")
    def webrtc_page():
        return FileResponse(static_dir / "webrtc.html")

    # --- Voice cloning ---
    import os as _os
    PRESET_VOICES = [
        "Trúc Ly", "Minh Đức", "Ngọc Linh", "Xuân Vĩnh",
        "Đoan Trang", "Mai Anh", "Thục Đoan", "Minh Triết",
        "Thùy Dung", "Quang Sơn", "Ngọc Trân", "Thanh Bình",
        "Phạm Tuyên", "Thái Sơn",
    ]
    voice_dir = static_dir / "voices"
    voice_dir.mkdir(exist_ok=True)

    def _voice_options() -> list[dict]:
        """Danh sách giọng chọn được: preset + giọng đã clone (nguồn sự thật cho config API + UI)."""
        cloned = sorted(f.stem for f in voice_dir.glob("*.wav"))
        return [{"name": v, "label": v} for v in PRESET_VOICES + cloned]

    @app.post("/v1/voice/clone")
    async def clone_voice(request: Request):
        import uuid as _uuid

        body = await request.json()
        voice_name = body.get("name", f"voice_{_uuid.uuid4().hex[:6]}")
        # ref_audio là path hoặc base64 — lưu file rồi trỏ
        import base64 as _b64
        audio_b64 = body.get("audio", "")
        if not audio_b64:
            return {"error": "missing audio"}
        try:
            audio_bytes = _b64.b64decode(audio_b64)
        except Exception:
            return {"error": "invalid base64"}
        voice_path = voice_dir / f"{voice_name}.wav"
        voice_path.write_bytes(audio_bytes)
        return {"voice": voice_name, "path": str(voice_path)}

    @app.get("/v1/voice/list")
    def list_voices():
        voices = [o["name"] for o in _voice_options() if o["name"] not in PRESET_VOICES]
        return {"voices": voices, "preset": PRESET_VOICES}

    # --- config REST endpoints (model config, cần restart pipeline) ---
    COMPUTE_TYPES = ("int8_float16", "float16", "int8", "float32")

    def _cfg_err(msg: str) -> HTTPException:
        return HTTPException(status_code=400, detail=msg)

    from ...LLM.prompts import DEFAULT_SYSTEM_PROMPTS, effective_system_prompt
    from ...backend_registry import LLM_MODEL_OPTIONS

    def _api_key_configured(c) -> bool:
        """Key có sẵn qua config UI hoặc env (theo backend hiện tại). Không echo giá trị."""
        if c.llm_api_key:
            return True
        env_map = {"openai": "OPENAI_API_KEY", "hf-router": "HF_TOKEN",
                   "gemini": "GEMINI_API_KEY"}
        env_name = env_map.get(c.llm_backend)
        return bool(env_name and os.environ.get(env_name))

    @app.get("/v1/config")
    def get_config():
        c = app.state.pipeline_config
        return {
            "llm": {"backend": c.llm_backend, "model": c.llm_model,
                    "device": c.llm_device, "base_url": c.llm_base_url,
                    "temperature": c.llm_temperature, "max_tokens": c.llm_max_tokens,
                    "system_prompt": c.llm_system_prompt,
                    "effective_prompt": effective_system_prompt(c.llm_backend, c.llm_system_prompt),
                    "default_prompts": DEFAULT_SYSTEM_PROMPTS,
                    "api_key_set": _api_key_configured(c),
                    "model_options": LLM_MODEL_OPTIONS},
            "stt": {"name": c.stt_name,
                    "beam_size": c.stt_beam_size,
                    "compute_type": c.stt_compute_type,
                    "options": [{"name": b.name, "label": b.label}
                                for b in STT_BACKENDS.values()]},
            "tts": {"name": c.tts_name, "voice": c.tts_voice, "streaming": c.tts_streaming,
                    "denoise": c.tts_denoise, "backend": c.tts_backend,
                    "style": c.tts_style, "temperature": c.tts_temperature,
                    "max_chars": c.tts_max_chars,
                    "options": [{"name": b.name, "label": b.label}
                                for b in TTS_BACKENDS.values()],
                    "voice_options": _voice_options()},
            "vad": {"min_silence_ms": c.min_silence_ms, "threshold": c.vad_threshold,
                    "min_speech_ms": c.min_speech_ms, "speech_pad_ms": c.speech_pad_ms},
            "live_transcription": {"enabled": c.enable_live_transcription},
        }

    @app.post("/v1/config")
    def update_config(body: dict = Body(...)):
        # cập nhật PipelineConfig (WS connection kế tiếp sẽ dùng config mới)
        c = app.state.pipeline_config
        llm = body.get("llm", {})
        if not isinstance(llm, dict): raise _cfg_err("llm phải là object")
        if "api_key" in llm and llm["api_key"]:
            if not isinstance(llm["api_key"], str):
                raise _cfg_err("llm.api_key phải là chuỗi")
            c.llm_api_key = llm["api_key"].strip()
        if "backend" in llm:
            c.llm_backend = llm["backend"]
            # phòng ngừa: backend remote (openai/hf-router/gemini) bắt buộc có key
            env_map = {"openai": "OPENAI_API_KEY", "hf-router": "HF_TOKEN",
                       "gemini": "GEMINI_API_KEY"}
            if c.llm_backend in env_map:
                key = c.llm_api_key or os.environ.get(env_map[c.llm_backend], "")
                if not key:
                    raise _cfg_err(
                        f"llm.backend={c.llm_backend} cần API key — nhập ở Settings → LLM → API key "
                        f"(hoặc set env {env_map[c.llm_backend]}) trước khi lưu")
        if "model" in llm: c.llm_model = llm["model"]
        if "device" in llm: c.llm_device = llm["device"]
        if "base_url" in llm: c.llm_base_url = llm["base_url"]
        if "temperature" in llm:
            if not isinstance(llm["temperature"], (int, float)) or not 0 <= llm["temperature"] <= 2:
                raise _cfg_err("llm.temperature phải là số trong [0, 2]")
            c.llm_temperature = float(llm["temperature"])
        if "max_tokens" in llm:
            if not isinstance(llm["max_tokens"], int) or llm["max_tokens"] < 1:
                raise _cfg_err("llm.max_tokens phải là số nguyên ≥ 1")
            c.llm_max_tokens = llm["max_tokens"]
        if "system_prompt" in llm and llm["system_prompt"] is not None:
            if not isinstance(llm["system_prompt"], str):
                raise _cfg_err("llm.system_prompt phải là chuỗi")
            c.llm_system_prompt = llm["system_prompt"]
        tts = body.get("tts", {})
        if not isinstance(tts, dict): raise _cfg_err("tts phải là object")
        if "name" in tts and tts["name"] not in TTS_BACKENDS:
            raise _cfg_err(f"TTS backend không hỗ trợ: {tts['name']}")
        if "name" in tts: c.tts_name = tts["name"]
        if "voice" in tts:
            if not any(tts["voice"] == o["name"] for o in _voice_options()):
                raise _cfg_err(f"Giọng không có sẵn: {tts['voice']} — dùng POST /v1/voice/clone để tạo giọng mới")
            c.tts_voice = tts["voice"]
        if "streaming" in tts:
            if not isinstance(tts["streaming"], bool):
                raise _cfg_err("tts.streaming phải là boolean")
            c.tts_streaming = tts["streaming"]
        if "denoise" in tts:
            if not isinstance(tts["denoise"], bool):
                raise _cfg_err("tts.denoise phải là boolean")
            c.tts_denoise = tts["denoise"]
        if "backend" in tts:
            if tts["backend"] not in ("onnx", "pytorch", "auto"):
                raise _cfg_err("tts.backend phải là onnx | pytorch | auto")
            c.tts_backend = tts["backend"]
        if "style" in tts:
            if tts["style"] not in ("tu_nhien", "tin_tuc", "doc_truyen"):
                raise _cfg_err("tts.style phải là tu_nhien | tin_tuc | doc_truyen")
            c.tts_style = tts["style"]
        if "temperature" in tts:
            if not isinstance(tts["temperature"], (int, float)) or not 0.1 <= tts["temperature"] <= 2.0:
                raise _cfg_err("tts.temperature phải là số trong [0.1, 2.0]")
            c.tts_temperature = float(tts["temperature"])
        if "max_chars" in tts:
            if not isinstance(tts["max_chars"], int) or not 32 <= tts["max_chars"] <= 1024:
                raise _cfg_err("tts.max_chars phải là số nguyên trong [32, 1024]")
            c.tts_max_chars = tts["max_chars"]
        vad = body.get("vad", {})
        if not isinstance(vad, dict): raise _cfg_err("vad phải là object")
        if "min_silence_ms" in vad:
            if not isinstance(vad["min_silence_ms"], int) or vad["min_silence_ms"] < 0:
                raise _cfg_err("vad.min_silence_ms phải là số nguyên ≥ 0")
            c.min_silence_ms = vad["min_silence_ms"]
        if "threshold" in vad:
            if not isinstance(vad["threshold"], (int, float)) or not 0 < vad["threshold"] < 1:
                raise _cfg_err("vad.threshold phải là số trong (0, 1)")
            c.vad_threshold = float(vad["threshold"])
        if "min_speech_ms" in vad:
            if not isinstance(vad["min_speech_ms"], int) or vad["min_speech_ms"] < 0:
                raise _cfg_err("vad.min_speech_ms phải là số nguyên ≥ 0")
            c.min_speech_ms = vad["min_speech_ms"]
        if "speech_pad_ms" in vad:
            if not isinstance(vad["speech_pad_ms"], int) or vad["speech_pad_ms"] < 0:
                raise _cfg_err("vad.speech_pad_ms phải là số nguyên ≥ 0")
            c.speech_pad_ms = vad["speech_pad_ms"]
        stt = body.get("stt", {})
        if stt:
            if not isinstance(stt, dict):
                raise _cfg_err("stt phải là object")
            if "name" in stt and stt["name"] not in STT_BACKENDS:
                raise _cfg_err(f"STT backend không hỗ trợ: {stt['name']}")
            if "name" in stt: c.stt_name = stt["name"]
            if "beam_size" in stt:
                if not isinstance(stt["beam_size"], int) or stt["beam_size"] < 1:
                    raise _cfg_err("stt.beam_size phải là số nguyên ≥ 1")
                c.stt_beam_size = stt["beam_size"]
            if "compute_type" in stt:
                if stt["compute_type"] not in COMPUTE_TYPES:
                    raise _cfg_err(f"stt.compute_type phải thuộc {list(COMPUTE_TYPES)}")
                c.stt_compute_type = stt["compute_type"]
        lt = body.get("live_transcription", {})
        if not isinstance(lt, dict): raise _cfg_err("live_transcription phải là object")
        if "enabled" in lt:
            if not isinstance(lt["enabled"], bool):
                raise _cfg_err("live_transcription.enabled phải là boolean")
            c.enable_live_transcription = lt["enabled"]
        # persist: lưu ra file để phiên chạy server sau dùng lại
        try:
            save_config(c, CONFIG_FILE)
        except Exception as e:
            logging.getLogger("s2s.api").warning(f"Không ghi được config persist: {e}")
        return {"status": "updated", "note": "Pipeline mới (WS connection kế tiếp) sẽ dùng config này. Đã lưu cho các phiên sau."}


    @app.post("/v1/test/pipeline")
    async def pipeline_test():
        import asyncio

        # chặn chạy song song: 2 test đồng thời tải 2 bản Qwen3-8B → CUDA OOM
        if not _test_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Đang có test luồng chạy — đợi lần trước xong rồi bấm lại")
        try:
            c = app.state.pipeline_config
            # transformers 5.x lazy-import KHÔNG thread-safe: LLM (transformers), RAG
            # (sentence_transformers) và TTS (mms-vie, VitsModel) đều dùng transformers
            # — nếu ≥2 thread cùng import lần đầu, 1 thread thấy namespace partial
            # (ImportError AutoModelForCausalLM/AutoTokenizer). Pre-import hết ở
            # event-loop thread trước khi spawn thread test (RAG luôn dùng
            # sentence_transformers nên luôn cần pre-import).
            if c.llm_backend == "transformers" or c.tts_name == "mms-vie":
                import transformers  # noqa: F401
                from transformers import AutoModelForCausalLM, AutoTokenizer, VitsModel  # noqa: F401
            # chạy trong thread — không block event loop (model tải lâu)
            stt, llm, rag, tts = await asyncio.gather(
                asyncio.to_thread(_pipeline_test_stt, c),
                asyncio.to_thread(_pipeline_test_llm, c),
                asyncio.to_thread(_pipeline_test_rag, c),
                asyncio.to_thread(_pipeline_test_tts, c),
            )
            return {"stt": stt, "llm": llm, "rag": rag, "tts": tts}
        finally:
            _test_lock.release()

    # --- LLM Proxy: /v1/chat/completions (background agent / RAG tinh chế) ---
    @app.post("/v1/chat/completions")
    async def llm_proxy(body: dict = Body(...)):
        import asyncio
        import queue
        import threading

        import json as _json

        from ...backend_registry import LLMConfig, get_llm_handler
        from ...pipeline.messages import (EndOfResponse, GenerateResponseRequest,
                                         LLMResponseChunk, PIPELINE_END)

        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="messages (list) là bắt buộc")
        stream = bool(body.get("stream", False))

        c = app.state.pipeline_config
        cfg = LLMConfig(
            backend=c.llm_backend, model_name=c.llm_model,
            api_key=c.llm_api_key, base_url=c.llm_base_url,
            temperature=c.llm_temperature, max_tokens=c.llm_max_tokens,
            device=c.llm_device, system_prompt=c.llm_system_prompt,
        )
        iq, oq = queue.Queue(), queue.Queue()
        h = get_llm_handler(iq, oq, cfg)
        # history từ messages (trừ turn cuối làm user text)
        h._history = [{"role": m["role"], "content": m["content"]}
                      for m in messages[:-1] if m.get("role") != "system"]
        system_msgs = [m["content"] for m in messages if m.get("role") == "system"]
        if system_msgs:
            h.system_prompt = system_msgs[-1]
        stop = threading.Event()
        h.stop_event = stop
        t = threading.Thread(target=h.run, daemon=True)
        t.start()

        async def _collect():
            chunks: list[str] = []
            while True:
                try:
                    out = oq.get(timeout=120)
                except queue.Empty:
                    break
                if isinstance(out, LLMResponseChunk):
                    chunks.append(out.text_delta)
                    yield out.text_delta, None
                elif isinstance(out, EndOfResponse):
                    stop.set()
                    break
            stop.set()

        async def _run():
            iq.put(GenerateResponseRequest(
                text=messages[-1].get("content", ""), language_code="vi", turn_id=1))
            iq.put(PIPELINE_END)

        if not stream:
            text = []
            await _run()
            async for delta, _ in _collect():
                text.append(delta)
            return {
                "id": "chatcmpl-proxy",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant",
                                                      "content": "".join(text)},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                          "total_tokens": 0},
            }

        # stream: SSE
        from fastapi.responses import StreamingResponse

        async def _stream_gen():
            await _run()
            async for delta, _ in _collect():
                if delta:
                    payload = {"choices": [{"index": 0,
                                            "delta": {"content": delta},
                                            "finish_reason": None}]}
                    yield f"data: {_json.dumps(payload, ensure_ascii=False)}\n\n"
            yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(_stream_gen(),
                                 media_type="text/event-stream")

    # --- RAG Knowledge Base ---

    @app.post("/v1/knowledge")
    async def upload_knowledge(file: UploadFile = File(...)):
        import uuid
        from s2s_vn.api.rag_service import rag_service
        
        content = await file.read()
        text = ""
        
        if file.filename.endswith(".pdf"):
            import pypdf
            import io
            try:
                pdf = pypdf.PdfReader(io.BytesIO(content))
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            except Exception as e:
                return {"error": f"Failed to parse PDF: {str(e)}"}
        else:
            try:
                text = content.decode("utf-8")
            except Exception as e:
                return {"error": f"Failed to decode text: {str(e)}"}
                
        if not text.strip():
            return {"error": "Empty or unreadable file"}
            
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        chunks = rag_service.add_document(text, doc_id, metadata={"filename": file.filename})
        return {"status": "success", "doc_id": doc_id, "chunks": chunks, "filename": file.filename}

    @app.get("/v1/knowledge")
    def list_knowledge():
        from s2s_vn.api.rag_service import rag_service
        docs = rag_service.list_documents()
        return {"documents": docs, "total": len(docs)}

    @app.get("/v1/knowledge/search")
    def search_knowledge(q: str = ""):
        from s2s_vn.api.rag_service import rag_service
        if not q.strip():
            return {"results": [], "query": q}
        results = rag_service.search(q.strip(), top_k=3)
        return {"results": results, "query": q}

    @app.delete("/v1/knowledge/{doc_id}")
    def delete_knowledge(doc_id: str):
        from s2s_vn.api.rag_service import rag_service
        deleted = rag_service.delete_document(doc_id)
        if deleted == 0:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
        return {"status": "deleted", "doc_id": doc_id, "chunks_deleted": deleted}

    return app


def _validate_llm_key(cfg: PipelineConfig, source: str) -> None:
    """Chặn cấu hình chắc chắn vỡ: backend remote thiếu key (thay vì LLM chết im)."""
    env_map = {"openai": "OPENAI_API_KEY", "hf-router": "HF_TOKEN",
               "gemini": "GEMINI_API_KEY"}
    if cfg.llm_backend in env_map:
        key = cfg.llm_api_key or os.environ.get(env_map[cfg.llm_backend], "")
        if not key:
            raise RuntimeError(
                f"Cấu hình từ {source}: llm_backend={cfg.llm_backend} cần API key — "
                f"đặt env {env_map[cfg.llm_backend]} hoặc nhập llm.api_key qua POST /v1/config. "
                f"Dùng --llm-backend transformers (local, không cần key) để chạy ngay.")


def main(argv: list[str] | None = None) -> None:
    import argparse

    import uvicorn

    from ...backend_registry import PROVIDERS

    parser = argparse.ArgumentParser(description="s2s-vn realtime server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config", default=None,
                        help="File config JSON để nạp (mặc định: config.json). "
                             "Vd: --config config-local.json")
    # STT
    parser.add_argument("--stt-name", default="phowhisper-medium")
    parser.add_argument("--stt-beam-size", type=int, default=1)
    parser.add_argument("--stt-compute-type", default="int8_float16")
    # LLM
    parser.add_argument("--llm-backend", default="openai",
                        choices=["openai", "hf-router", "gemini", "local", "transformers"])
    parser.add_argument("--llm-model", default="gpt-4.1-mini")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-max-tokens", type=int, default=1024)
    parser.add_argument("--llm-device", default="cuda")
    parser.add_argument("--llm-system-prompt", default=None)
    # TTS
    parser.add_argument("--tts-name", default="vieneu", choices=list(TTS_BACKENDS))
    parser.add_argument("--tts-voice", default="Trúc Ly")
    parser.add_argument("--tts-streaming", action="store_true", default=True)
    parser.add_argument("--tts-denoise", action="store_true", default=True)
    parser.add_argument("--tts-backend", default="onnx")
    parser.add_argument("--tts-style", default="tu_nhien")
    parser.add_argument("--tts-temperature", type=float, default=0.8)
    parser.add_argument("--tts-max-chars", type=int, default=256)
    # VAD
    parser.add_argument("--vad-threshold", type=float, default=0.6)
    parser.add_argument("--min-silence-ms", type=int, default=300)
    parser.add_argument("--min-speech-ms", type=int, default=500)
    parser.add_argument("--speech-pad-ms", type=int, default=500)
    # Realtime
    parser.add_argument("--enable-live-transcription", action="store_true", default=True)
    args = parser.parse_args(argv)

    # config ưu tiên hơn args: --config <file> (hoặc config.json mặc định) nếu tồn tại
    cfg_path = args.config or CONFIG_FILE
    cfg = load_config(cfg_path)
    if cfg is not None:
        source = f"file {cfg_path}"
        logging.getLogger("s2s.api").info(f"Đã nạp config từ {cfg_path}")
    else:
        source = "flags CLI"
    if cfg is None:
        cfg = PipelineConfig(
            stt_name=args.stt_name,
            stt_beam_size=args.stt_beam_size,
            stt_compute_type=args.stt_compute_type,
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            llm_base_url=args.llm_base_url,
            llm_temperature=args.llm_temperature,
            llm_max_tokens=args.llm_max_tokens,
            llm_device=args.llm_device,
            llm_system_prompt=args.llm_system_prompt,
            tts_name=args.tts_name,
            tts_voice=args.tts_voice,
            tts_streaming=args.tts_streaming,
            tts_denoise=args.tts_denoise,
            tts_backend=args.tts_backend,
            tts_style=args.tts_style,
            tts_temperature=args.tts_temperature,
            tts_max_chars=args.tts_max_chars,
            vad_threshold=args.vad_threshold,
            min_silence_ms=args.min_silence_ms,
            min_speech_ms=args.min_speech_ms,
            speech_pad_ms=args.speech_pad_ms,
            enable_live_transcription=args.enable_live_transcription,
        )
    # chặn cấu hình chắc chắn vỡ trước khi start (backend remote thiếu key)
    try:
        _validate_llm_key(cfg, source)
    except RuntimeError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
    app = create_app(cfg)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
