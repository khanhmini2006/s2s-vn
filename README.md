# s2s-vn

Voice-agent pipeline **VAD → STT → LLM → TTS**, tiếng Việt, expose qua OpenAI Realtime-compatible WebSocket + WebRTC.

Reimplement của [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech). Apache-2.0; model component có license riêng (xem bảng Models).

## Kiến trúc

```
Audio bytes → [VAD] → [STT] → [Notifier] → [LLM] → [LMProcessor] → [TTS] → Audio out
                Silero   PhoWhisper  mock?      registry    gom text     VieNeu
              / Zipformer
```

Mỗi handler là 1 thread riêng, nối bằng `queue.Queue` (`s2s_pipeline.py`). Component tách thư mục `VAD/ STT/ LLM/ TTS/` như repo gốc. Mỗi WebSocket/WebRTC connection = 1 pipeline session.

## Cài đặt

```bash
pip install -e ".[realtime]"                # server cơ bản (fastapi + uvicorn + websockets)

# extras theo nhu cầu:
#   .[stt]          → faster-whisper (PhoWhisper)
#   .[zipformer]    → sherpa-onnx (Zipformer-30M)
#   .[tts]          → vieneu
#   .[transformers] → transformers + torch + bitsandbytes + accelerate (LLM local Qwen3-8B 4bit)
#   .[talk]         → sounddevice + websocket-client (client mic/loa)
#   .[dev]          → pytest + ruff
```

## CLI 3 lệnh (theo repo gốc)

```bash
s2s-vn serve                              # chạy Realtime server (WebSocket + WebRTC)
s2s-vn talk --url ws://127.0.0.1:8765/v1/realtime   # client mic/loa nói chuyện
s2s-vn local                              # serve + talk trong 1 process (loopback)
```

`serve` mặc định bind `0.0.0.0:8765`. Tương đương `python -m s2s_vn.api.openai_realtime.server`.

```bash
# LLM remote (cần key env: OPENAI_API_KEY | HF_TOKEN | GEMINI_API_KEY)
s2s-vn serve --llm-backend openai --llm-model gpt-4.1-mini
# LLM local (vLLM/Ollama/llama.cpp OpenAI-compatible)
s2s-vn serve --llm-backend local --llm-base-url http://localhost:8000/v1
# LLM local qua Transformers (Qwen3-8B 4bit, không cần server riêng)
s2s-vn serve --llm-backend transformers --llm-model Qwen/Qwen3-8B
```

## Web demo (không cần client)

Server chạy rồi mở trình duyệt:

| Trang | Mô tả |
|---|---|
| `/demo` | Nói chuyện qua browser mic → WS. Nút **🔍 Test luồng** chẩn đoán STT/LLM/RAG/TTS; nút **🔄 Reload phiên** build lại pipeline sau khi đổi config |
| `/webrtc` | Phiên bản WebRTC (Opus 48k ↔ pipeline) |

**Config:** POST `/v1/config` (JSON) → áp cho WS connection kế tiếp + **persist ra `config.json`** (chmod 600) — phiên chạy server sau tự nạp (ưu tiên file > args CLI > default). GET `/v1/config` xem config hiện tại. Validate từng field (sai type/khoảng → 400). Đổi model = phải tải model mới (lần đầu ~1 phút).

## Cấu hình qua CLI

`s2s-vn serve` hỗ trợ đầy đủ flags (dùng khi chưa có `config.json`; có file thì file ưu tiên):

```bash
# STT
--stt-name phowhisper-medium|zipformer-vi-6000h --stt-beam-size 1 --stt-compute-type int8_float16
# LLM
--llm-backend openai|hf-router|gemini|local|transformers --llm-model MODEL
--llm-base-url URL --llm-api-key KEY --llm-temperature 0.0 --llm-max-tokens 1024 --llm-device cuda
--llm-system-prompt "Bạn là trợ lý..."
# TTS
--tts-voice "Trúc Ly" --tts-backend onnx --tts-style tu_nhien --tts-temperature 0.8 --tts-max-chars 256
# VAD
--vad-threshold 0.6 --min-silence-ms 300 --min-speech-ms 500 --speech-pad-ms 500
```

## Models

| Stage | Model | License |
|---|---|---|
| VAD | Silero VAD v5 (ONNX) | MIT |
| STT | PhoWhisper-medium (faster-whisper CT2) | BSD-3 + MIT |
| STT | Zipformer-30M-RNNT (`zipformer-vi-6000h`, sherpa-onnx — cực nhanh 0.02s/1.8s audio) | cc-by-nc-nd (chỉ nghiên cứu) |
| TTS | VieNeu-TTS v3 Turbo | Apache-2.0 |
| LLM | tuỳ provider (registry) — Qwen3-8B verified local (4bit, tool calling ổn) | Apache-2.0 |

Lưu ý: Qwen2.5-3B trả lời trộn tiếng Trung — dùng Qwen3-8B / Qwen3-4B-Instruct-2507 trở lên. Model tự tải qua HF cache lần chạy đầu.

## RAG (Kho tài liệu)

RAG = **Tool Calling** (chuẩn OpenAI Realtime): LLM handler đăng ký tool `search_knowledge`; khi LLM gọi → server tự query ChromaDB (`./chroma_db`, embedding `intfloat/multilingual-e5-small`) → tool_output → LLM tiếp tục. Client không cần làm gì.

- Upload: `POST /v1/knowledge` (multipart file .txt/.pdf)
- System prompt: `POST /v1/config` với `llm.system_prompt` khuyên model gọi tool khi câu hỏi liên quan tài liệu nội bộ
- LLM Proxy cho background agent: `POST /v1/chat/completions` (non-stream + SSE) dùng backend hiện tại

## API endpoints

| Endpoint | Mô tả |
|---|---|
| `WS /v1/realtime` | OpenAI Realtime protocol (PCM16 16kHz — pipeline = protocol) |
| `POST /v1/realtime/calls` | WebRTC signaling (offer → answer) |
| `GET/POST /v1/config` | Xem / đổi config (persist) |
| `POST /v1/test/pipeline` | Self-check STT/LLM/RAG/TTS (chạy model thật) |
| `POST /v1/chat/completions` | LLM proxy (OpenAI-compatible) |
| `GET/POST/DELETE /v1/knowledge*` | Quản lý kho tài liệu |
| `POST /v1/voice/clone`, `GET /v1/voice/list` | Voice clone |
| `GET /v1/health`, `GET /v1/usage` | Health + metrics |

## Test

```bash
pytest                              # đơn vị: pipeline mock flow + RAG + config + test luồng
python scripts/mock_llm_server.py 8081   # mock LLM OpenAI-compatible (không cần key)
s2s-vn serve --port 8765 --llm-backend local --llm-base-url http://127.0.0.1:8081/v1
python scripts/test_realtime_client.py    # client WS: session → audio mẫu → response.done
```

Scripts khác: `test_barge_in.py`, `test_tool_calling.py` (cần `mock_llm_tools.py :8082`), `test_live_transcription.py`, `test_e2e_params.py` (E2E WS thật), `test_pipeline_full.py`, `wake_word_client.py` (wake word).

## Lưu ý CUDA

- `faster-whisper` (CTranslate2) cần CUDA 12 libs; base torch cu130 cài thêm `nvidia-cublas-cu12`
- LLM transformers 4bit (Qwen3-8B ~6GB VRAM) cần `pip install -e ".[transformers]"`
