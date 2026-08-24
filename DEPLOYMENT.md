# Triển khai (test)

Hướng dẫn ngắn để dựng server + kết nối client cho mục đích **thử nghiệm** (không phải hướng dẫn vận hành production — không có HA/backup/monitoring). Xem `CLAUDE.md` cho chi tiết kiến trúc.

## Yêu cầu máy chủ

| | Tối thiểu |
|---|---|
| OS | Linux (đã test Ubuntu 22.04) |
| Python | ≥ 3.10 |
| GPU | Không bắt buộc nếu dùng LLM remote (openai/gemini/hf-router). Có GPU CUDA (≥8GB VRAM) nếu muốn dùng LLM local (`transformers`, Qwen3-8B 4bit) |
| CUDA toolkit (`nvcc`) | **Không cần** cho cấu hình mặc định. Chỉ cần nếu tự build thêm gói CUDA riêng (vLLM, flash-attn) — nhiều máy không có sẵn, xem Gotcha bên dưới |
| Disk trống | ≥ 5GB (STT+VAD+TTS mặc định); ≥ 15GB nếu dùng thêm LLM local (Qwen3-8B ~6GB) |
| RAM | Đủ chạy Python + model nhỏ; không có yêu cầu đặc biệt cho cấu hình mặc định |

Model tải tự động về HF cache lần chạy đầu (không cần tải thủ công): Silero VAD, STT, TTS đã chọn, và LLM local nếu dùng `transformers`.

## Yêu cầu máy client

**CLI (`s2s-vn talk`)** — nói chuyện qua mic/loa:
- Cài `pip install -e ".[talk]"` (sounddevice, websocket-client)
- Thiết bị âm thanh chuẩn PortAudio (mic + loa nhận diện được bởi hệ điều hành)

**Web (`/demo`, `/webrtc`)** — không cần cài gì:
- Chỉ cần trình duyệt hỗ trợ WebRTC/WebSocket (Chrome/Firefox/Edge bản mới)
- Truy cập `http://<host-server>:8765/demo` hoặc `/webrtc`

## Khởi chạy

```bash
# Cài đặt tối thiểu (LLM remote + TTS piper-vie mặc định)
pip install -e ".[realtime,stt,zipformer,tts-piper]"

# Chạy server (cần API key LLM qua env hoặc file config)
export GEMINI_API_KEY=...   # hoặc OPENAI_API_KEY / HF_TOKEN
s2s-vn serve --llm-backend gemini --stt-name zipformer-vi-6000h --tts-name piper-vie
```

Dùng file config sẵn có làm mẫu tham chiếu mặc định: `config-piper.json` (LLM gemini + TTS `piper-vie`) — `s2s-vn serve --config config-piper.json`.

⚠️ `config-piper.json` chứa sẵn API key Gemini thật (file `chmod 600`, không commit lên git — xem `.gitignore`). Nếu chia sẻ/nhân bản file này, xóa hoặc thay key trước.

Chạy nền lâu dài: dùng `systemd` (service unit gọi `s2s-vn serve`), `tmux`, hoặc `nohup ... &` — repo chưa có Dockerfile/docker-compose sẵn.

## Lưu ý kỹ thuật quan trọng

- **Port mặc định**: `8765` — cả WebSocket (`/v1/realtime`) và WebRTC signaling (`/v1/realtime/calls`) dùng chung port.
- **License TTS — đọc kỹ trước khi chọn/đổi** (`--tts-name`):
  - `piper-vie` (**mặc định**) — GPL-3.0-or-later, copyleft mã nguồn, có xung đột tiềm ẩn với license Apache-2.0 của dự án nếu import trực tiếp cùng process. Đã được chấp nhận rủi ro nội bộ khi tích hợp — **cân nhắc kỹ nếu distribute/đóng gói ra ngoài phạm vi nội bộ**. Giọng `huongly.onnx` không nằm trong kho chính thức, nguồn gốc chưa xác minh.
  - `vieneu` — MIT, dùng thoải mái kể cả thương mại. Đổi sang backend này nếu lo ngại license GPL.
  - `mms-vie` — CC-BY-NC-4.0, **chỉ phi thương mại**.
- **CUDA toolkit thiếu** — máy không có `nvcc` thì backend LLM `local` qua vLLM và `flash-attn` sẽ không cài/build được (JIT compile cần toolkit đồng bộ với torch). Dùng `transformers` (bitsandbytes 4bit, không cần vLLM) hoặc backend remote thay thế.
- **VRAM per-session** — mỗi kết nối WS/WebRTC tạo 1 pipeline riêng với model riêng; nhiều session đồng thời (nhiều tab/client) cộng dồn VRAM, có thể OOM nếu dùng LLM local trên GPU nhỏ.
- **Disk cho HF cache** — kiểm tra dung lượng trống trước khi chạy lần đầu, đặc biệt nếu bật thêm LLM local hoặc nhiều STT/TTS backend cùng lúc (mỗi backend tải model riêng).
