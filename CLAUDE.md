# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Tổng quan

Voice-agent pipeline **VAD → STT → LLM → TTS**, tiếng Việt, expose qua OpenAI Realtime-compatible WebSocket (`ws://<host>:8765/v1/realtime`) + WebRTC signaling (`POST /v1/realtime/calls`). Reimplement của [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech). Toàn bộ comment/docstring viết tiếng Việt — giữ nguyên convention này.

Cấu trúc package theo repo gốc: component tách thư mục `VAD/ STT/ LLM/ TTS/`, `s2s_pipeline.py` + `base_handler.py` + `backend_registry.py` ở root, transport trong `api/openai_realtime/`.

## Commands

```bash
# Install (extras: stt, zipformer, tts, tts-mms, tts-piper, realtime, transformers, talk, dev)
pip install -e ".[realtime]"          # server cơ bản
#   .[transformers] → LLM local Qwen3-8B 4bit (transformers + torch + bitsandbytes + accelerate)
#   .[talk]         → client mic/loa (sounddevice + websocket-client)
#   .[zipformer]    → sherpa-onnx
#   .[tts-mms]      → facebook/mms-tts-vie (VITS, CC-BY-NC-4.0 — chỉ phi thương mại)
#   .[tts-piper]    → Piper TTS giọng Huongly (ONNX) — GPL-3.0-or-later

# CLI 3 lệnh (theo repo gốc)
s2s-vn serve                             # server WS + WebRTC (default 0.0.0.0:8765)
s2s-vn talk --url ws://127.0.0.1:8765/v1/realtime   # client mic/loa nói chuyện
s2s-vn local                             # serve + talk in-process (loopback)
# Tương đương serve: PYTHONPATH=src python -m s2s_vn.api.openai_realtime.server
#   --llm-backend openai|hf-router|gemini|local|transformers (cần key env tương ứng)
#   --llm-backend local → --llm-base-url http://localhost:8000/v1 (vLLM/Ollama/llama.cpp)

# Test đơn vị (tests/: pipeline mock + RAG ChromaDB + config + test luồng, ~20s)
pytest

# Test tích hợp (không cần key thật — mock LLM server HTTP)
python scripts/mock_llm_server.py 8081  # OpenAI-compatible mock
s2s-vn serve --port 8765 --llm-backend local --llm-base-url http://127.0.0.1:8081/v1
python scripts/test_realtime_client.py
# Scripts khác: test_barge_in.py, test_tool_calling.py (cần mock_llm_tools.py :8082),
#   test_live_transcription.py, test_e2e_params.py (E2E WS thật inject audio mẫu),
#   test_pipeline_full.py (pipeline thật + mock LLM nhúng), wake_word_client.py

# Lint
ruff check src tests scripts
```

**Web demo** (không cần client): `/demo` (nói chuyện + nút Test luồng + Reload phiên), `/webrtc` (WebRTC). Cấu hình qua CLI flags (xem dưới) hoặc REST `POST /v1/config` + persist `config.json`. Server validate từng field config (sai type/khoảng → 400). API key qua `POST /v1/config` (`llm.api_key`) — GET chỉ trả `api_key_set`, không echo key.

**`s2s-vn serve` flags** (dùng khi chưa có file config; có file → file ưu tiên): `--config <file>` (chọn file config, mặc định `config.json`), `--stt-name/--stt-beam-size/--stt-compute-type`, `--llm-backend/--llm-model/--llm-base-url/--llm-api-key/--llm-temperature/--llm-max-tokens/--llm-device/--llm-system-prompt`, `--tts-name/--tts-voice/--tts-streaming/--tts-denoise/--tts-backend/--tts-style/--tts-temperature/--tts-max-chars`, `--vad-threshold/--min-silence-ms/--min-speech-ms/--speech-pad-ms`, `--enable-live-transcription`. Xem đủ: `s2s-vn serve --help`. Flags khai báo ở `server.py main()`; `cli.py serve` pass-through `sys.argv` còn lại → `server.main(argv)` (không trùng khai báo). Ví dụ: `s2s-vn serve --llm-backend transformers --llm-model Qwen/Qwen3-8B --stt-name zipformer-vi-6000h --tts-name vieneu --tts-voice "Trúc Ly"` (`--tts-name mms-vie` → Facebook MMS-TTS, `--tts-name piper-vie` → Piper TTS giọng Huongly — cả hai bỏ qua `--tts-voice/--tts-style/...` vì không hỗ trợ voice clone).

**RAG = Tool Calling** (chuẩn OpenAI Realtime): mọi LLM handler đăng ký tool `search_knowledge` (`backend_registry.TOOL_SEARCH_KNOWLEDGE`); khi LLM gọi → `realtime_service._execute_knowledge_tool` query ChromaDB → tool_output → LLM tiếp tục. Không inject trực tiếp.

## Kiến trúc

### Pipeline (src/s2s_vn/)

6 handler nối `queue.Queue`, mỗi handler 1 thread (`utils/thread_manager.py`). Dựng ở `s2s_pipeline.py:_make_handlers()`. Component tách thư mục `VAD/ STT/ LLM/ TTS/`:

```
audio_in → [VAD] → [STT] → [Notifier] → [LLM] → [LMOutputProcessor] → [TTS] → audio_out
               VAD/vad_handler (Silero)   STT/whisper|zipformer   STT/transcription_notifier
               LLM/llm_openai_compatible|transformers            LLM/lm_output_processor
               TTS/vieneu_tts_handler|mms_tts_handler|piper_tts_handler
```

- `base_handler.py` (root): vòng lặp `get(timeout=0.1) → process() → put`; exception trong `process` không giết thread. `PIPELINE_END` (b"END") sentinel thoát; `AUDIO_RESPONSE_DONE` trên audio_out.
- `pipeline/messages.py` (VADAudio, Transcription, GenerateResponseRequest, LLMResponseChunk, TTSInput, AudioOutput, CancelScope, ...); `pipeline/events.py` (SpeechStarted, PartialTranscription, AssistantText, FunctionCall, TokenUsage, ...) — handler đẩy vào `text_out`, transport đọc emit server event.
- `CancelScope` (messages.py): barge-in bằng generation counter chia sẻ. Handler gen dài (TTS/LLM) lưu gen lúc bắt đầu; `current() != gen` → turn đã cancel, dừng ngay.
- Model tải qua HF cache lần chạy đầu: Silero ONNX (`onnx-community/silero-vad`), PhoWhisper CT2 (`quocphu/PhoWhisper-ct2-FasterWhisper`), Zipformer (`hynt/Zipformer-30M-RNNT-6000h`), `vieneu.Vieneu(mode="v3turbo")`.

### Transport (src/s2s_vn/api/openai_realtime/)

- `server.py` `create_app()`: auth optional qua env `S2S_API_KEY` (trống = tắt). REST: `/v1/config` (persist), `/v1/test/pipeline` (test luồng, lock serial + cleanup VRAM), `/v1/chat/completions` (LLM proxy), `/v1/voice/*`, `/v1/knowledge*`, static pages. Entry `main(argv=None)` — gọi lại được từ `cli.py serve`. **Lưu ý path**: `static_dir = Path(__file__).parent.parent / "static"` (server nằm trong `openai_realtime/`); test helper scripts path = 5 parent.
- `websocket_router.py`: 1 WS = 1 `Session` = 1 `RealtimeService` (1 pipeline). Drain thread → `asyncio.Queue` → send loop. Idle timeout 300s. **Pre-warm**: `warmup_all()` nền (thread) → emit `server.model_ready` → demo bật nút Bắt đầu. Lưu ý: mỗi session = 1 bản model riêng (nhiều tab đồng thời → OOM).
- `realtime_service.py`: dịch OpenAI Realtime protocol ↔ pipeline. **Protocol = pipeline = 16kHz (không resample thừa — theo repo gốc)** (`audio_utils.resample_pcm16`, scipy). Custom event `pipeline.update` đổi VAD params runtime (không thuộc spec).
- `webrtc_router.py`: signaling `POST /v1/realtime/calls` + aiortc PC (browser Opus 48k → PCM16 16k → pipeline; reply 16k → resample 48k stereo → track; events qua RTCDataChannel). Mỗi call = 1 PC + 1 `RealtimeService`. `attach_drain_audio` patch `_handle_audio` để audio_out → `PipelineOutputTrack`.
- `api/rag_service.py` (giữ ở `api/` — không thuộc protocol): ChromaDB persistent (`./chroma_db`), embedding `intfloat/multilingual-e5-small`, singleton `rag_service`.
- `cli.py` + `talk.py` (root): CLI serve/talk/local; `talk` = client mic/loa (sounddevice 16k = WS 16k, không wake word).

### Handlers (src/s2s_vn/VAD/ STT/ LLM/ TTS/)

- `backend_registry.py` (root): `get_llm_handler()` theo `cfg.backend` — `openai`/`hf-router`/`gemini`/`local` → `OpenAICompatibleLLMHandler` (map provider→base_url/key env trong `PROVIDERS`); `transformers` → `TransformersLLMHandler`. `get_stt_handler` theo `cfg.stt_name` — `phowhisper-medium` → `WhisperSTTHandler` (CT2, default), `zipformer-vi-6000h` → `ZipformerSTTHandler` (sherpa-onnx). `get_tts_handler` theo `cfg.tts_name` — `vieneu` → `VieNeuTTSHandler` (voice clone, MIT, default), `mms-vie` → `MMSTTSHandler` (facebook/mms-tts-vie, VITS, **CC-BY-NC-4.0 — chỉ phi thương mại**, không voice clone), `piper-vie` → `PiperTTSHandler` (Piper ONNX giọng Huongly, **GPL-3.0-or-later**, không voice clone, checkpoint tại `api/static/voices_piper/huongly.onnx`). Không type-hint PipelineConfig trong signature (tránh circular import).
- `STT_BACKENDS` / `TTS_BACKENDS`: nguồn sự thật (GET `/v1/config` options, POST validate). **Thêm model STT/TTS = 1 entry trong registry + 1 branch family trong `get_stt_handler`/`get_tts_handler`**.
- `LLM/llm_openai_compatible.py`: streaming `/v1/chat/completions`, tool calling (`set_tools`, `_history` max 20), `system_prompt` đổi qua `session.update`.
- `LLM/transformers_llm.py`: Qwen3-8B 4bit. Model không nhận `enable_thinking` kwarg (chỉ 2507 trở lên) → filter `<think>` block + turn tool call không emit text JSON lẫn.
- `LLM/lm_output_processor.py`: buffer LLMResponseChunk theo turn_id → EndOfResponse → 1 TTSInput + AssistantTextEvent. `_clean_text` filter markdown/emoji trước TTS.
- `LLM/prompts.py`: TRANSFORMERS_PROMPT / OPENAI_COMPATIBLE_PROMPT + `effective_system_prompt()` — system prompt không tự đổi khi đổi model.
- `STT/transcription_notifier.py` (MockTranscriptionNotifier): Transcription → GenerateResponseRequest — part pipeline chính.

## Gotchas

- **Thứ tự handler hardcode theo index**: `realtime_service._llm_handler()` = `handlers[3]`, `_on_pipeline_update` dùng `handlers[0]` (VAD). Đổi `_make_handlers` phải giữ VAD index 0, LLM index 3.
- Sample rate: pipeline = protocol = 16k (như repo gốc). Nếu đổi `PROTOCOL_RATE` phải sửa luôn static page client + talk.py.
- `test_rag.py` import kiểu `from src.s2s_vn...` (dựa cwd); các file khác `from s2s_vn...` (cần editable install hoặc `PYTHONPATH=src`). `test_pipeline_full.py` hardcode `sys.path.insert(0, "/home/tdkhanh/s2s-vn/src")` — path máy chủ riêng, không portable.
- Qwen2.5-3B trả lời trộn tiếng Trung — dùng Qwen3-8B / Qwen3-4B-Instruct-2507 trở lên cho tiếng Việt.
- Qwen3-8B-AWQ KHÔNG dùng được (gptqmodel không tương thích transformers 5.12); vLLM trên máy này bế tắc (flashinfer JIT cần CUDA toolkit pip đồng bộ).
- `--min-silence-ms` (default 300) quá nhỏ → VAD cắt nửa câu; `min_speech_ms=500` lọc nhiễu.
- **License TTS `mms-vie`**: `facebook/mms-tts-vie` phát hành CC-BY-NC-4.0 (chỉ phi thương mại) — khác MIT của `vieneu` (default). Không dùng `--tts-name mms-vie` nếu dự án/triển khai có yếu tố thương mại. Model cũng không hỗ trợ voice clone (1 giọng cố định), RTF~0.01 (rất nhanh, không streaming autoregressive — `MMSTTSHandler` tự cắt chunk giả lập).
- **License TTS `piper-vie`**: package `piper-tts` (repo `OHF-voice/piper1-gpl`) là **GPL-3.0-or-later** (copyleft mã nguồn — khác các license MIT/CC-BY-NC-4.0 kia chỉ giới hạn *mục đích dùng output*). Dự án này khai Apache-2.0 trong `pyproject.toml` — xung đột license tiềm ẩn khi import trực tiếp cùng process. Đã cân nhắc và chấp nhận rủi ro này khi tích hợp; cân nhắc lại nếu distribute/đóng gói binary. Giọng `huongly.onnx` không nằm trong kho chính thức `rhasspy/piper-voices` — nguồn gốc/license riêng của checkpoint voice chưa xác minh. RTF~0.03 trên CPU (rất nhanh), streaming thật theo câu (không giả lập cắt chunk như `mms-vie`).
- **Race condition warmup TTS đa thread**: mọi `*TTSHandler.warmup()` có thể bị gọi đồng thời từ 2 thread — `RealtimeService._warmup_models()` (nền, gọi `pipeline.warmup_all()`) và chính thread `run()` của handler đó (`process()` tự lazy-warmup nếu `self._model/_voice is None`). Với handler dùng `transformers`/thư viện lazy-import không thread-safe (đã tái hiện thực tế với `MMSTTSHandler` → `ImportError`), bắt buộc có `threading.Lock` bọc `warmup()` (double-checked: kiểm tra lại field đã set sau khi acquire lock). `MMSTTSHandler`/`PiperTTSHandler` đã có; handler TTS mới phải theo pattern này.
- **Test luồng `/v1/test/pipeline`**: lock serial chặn chạy song song (2 test cùng tải Qwen3-8B → CUDA OOM); helper LLM `finally: del h + gc.collect() + torch.cuda.empty_cache()` trả VRAM. Pre-import transformers ở event-loop thread trước khi to_thread (transformers 5.x lazy-import không thread-safe).
- `config.json` persist (chmod 600, env `S2S_CONFIG_FILE`, `--config <file>` chọn file khác) ưu tiên file > args CLI > default. **Validate khi nạp** (`_validate_llm_key` trong main): backend remote (openai/hf-router/gemini) thiếu key → chặn start + cảnh báo rõ (không chết im). Nhiều lần trong phiên config.json bị đổi thủ công sang `openai` thiếu key — validate mới chặn được; khi thấy bị chặn → sửa config hoặc `--llm-backend transformers`.
- Drain thread (start_drain) mutate shared state (`response_id`, `_current_response_turn`) trong khi asyncio loop chạy `handle_event` — race đã biết, TODO trong comment.
