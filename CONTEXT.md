# CONTEXT.md

Ghi lại các quyết định (glossary + ADR ngắn) rút ra từ phiên `/grill-with-docs` khi cập nhật tài liệu triển khai. Không trùng lặp nội dung kỹ thuật đã có trong `CLAUDE.md` (dành cho AI) — file này ghi *quyết định về tài liệu*, không phải kiến trúc code.

## Thuật ngữ

- **Test triển khai** (test deployment): mục đích của `DEPLOYMENT.md` — giúp một người mới đọc xong là chạy được server + client để *thử nghiệm*, không phải hướng dẫn vận hành production lâu dài (không có HA, backup, monitoring, CI/CD).
- **Máy chủ** (server): máy chạy `s2s-vn serve` (pipeline + WS/WebRTC).
- **Máy client**: có 2 loại độc lập — **CLI client** (`s2s-vn talk`, cần mic/loa PortAudio) và **Web client** (`/demo`, `/webrtc`, chỉ cần trình duyệt hỗ trợ WebRTC/WebSocket).

## Quyết định (ADR ngắn)

- **ADR-1 — Vị trí tài liệu**: tạo file mới `DEPLOYMENT.md` ở root, không nhét vào `README.md`. Lý do: tách bạch "giới thiệu/dùng thử nhanh" khỏi "triển khai" tránh README phình to.
- **ADR-2 — Độ sâu**: 1 trang, súc tích. Bỏ hẳn ý định ban đầu về "khuyến nghị cấu hình theo mục đích sử dụng" và "2 mức tài nguyên tối thiểu/đầy đủ" — không cần thiết cho tài liệu test-only, có thể bổ sung sau nếu tài liệu này được nâng cấp thành hướng dẫn production thật.
- **ADR-3 — Phạm vi hạ tầng**: chỉ single-node. Không viết Docker/docker-compose (repo chưa có, đóng gói GPU + HF cache là việc riêng, ngoài phạm vi lần này). Không tính multi-node/load-balancer (kiến trúc hiện tại không có tầng LB).
- **ADR-4 — Cấu hình mặc định khuyến nghị** *(cập nhật)*: LLM remote (`gemini`) + TTS **`piper-vie`** (đổi từ `vieneu` — quyết định lại sau phiên đầu) + STT `zipformer-vi-6000h`. Tham chiếu `config-piper.json` làm mẫu (file này chứa sẵn API key Gemini thật, `chmod 600`, không commit git). **Lưu ý license**: `piper-vie` là GPL-3.0-or-later, có xung đột tiềm ẩn với Apache-2.0 của dự án — chấp nhận rủi ro cho dùng nội bộ/test, cân nhắc lại nếu distribute ra ngoài; `vieneu` (MIT) là lựa chọn thay thế an toàn hơn về license. LLM local (`transformers`, Qwen3-8B) và TTS `mms-vie` liệt kê như tùy chọn khác kèm cảnh báo, không phải mặc định.
- **ADR-5 — Yêu cầu tài nguyên**: 1 mức tối thiểu duy nhất (không phân tầng), dựa trên dữ kiện đo thực tế trên máy dev hiện tại (RTX 5070 Ti 16GB, Ubuntu 22.04.5, không có CUDA toolkit dev `nvcc`, disk 51GB trống tại thời điểm viết).

## Dữ kiện đã xác minh (2026-08-14)

- GPU: RTX 5070 Ti, 16GB VRAM, driver 580.173.02. `nvcc` không có (không CUDA toolkit dev — chặn vLLM/flash-attn, xem `mem:gotchas`).
- OS: Ubuntu 22.04.5 LTS. Python 3.13.12. torch 2.12.0+cu130 (CUDA 13.0 build).
- Disk `/`: 916G tổng, 51G trống (95% đầy) lúc viết tài liệu — cảnh báo rõ trong `DEPLOYMENT.md` vì sát ngưỡng.
- RAM: 62GB, swap 2GB (đã full lúc đo — không phải baseline đáng tin cậy để khuyến nghị mức RAM tối thiểu chính xác, chỉ ghi nhận định tính "đủ dùng").
- Port mặc định: WS `8765` (`/v1/realtime`), WebRTC signaling `POST /v1/realtime/calls` cùng port.
- Model tự tải về (HF cache, lần chạy đầu): Silero VAD (~2MB), Zipformer (~120MB), vieneu (~200-400MB) → tổng nhẹ nếu không dùng LLM local. Thêm Qwen3-8B 4bit (~6GB) nếu dùng `transformers` backend, PhoWhisper (~1.5GB) nếu chọn STT đó thay Zipformer, RAG embedding (~470MB) nếu dùng tool `search_knowledge`.

## Liên kết

- `mem:...` — không có memory nào liên quan trước đó (lần đầu viết CONTEXT.md cho repo này).
- Xem `CLAUDE.md` mục **Gotchas** cho chi tiết kỹ thuật đầy đủ (license TTS, CUDA/vLLM, transformers thread-safety, VRAM per-session).
