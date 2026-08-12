# Design: STT registry + backend wav2vec2 (nguyenvulebinh/wav2vec2-base-vietnamese-250h)

- Ngày: 2026-08-07
- Trạng thái: Approved (user OK "làm đi")
- Repo: s2s-vn (không phải git repository — bỏ bước commit spec)

## Mục tiêu

Thêm backend STT `wav2vec2-base-vietnamese-250h` vào pipeline VAD → STT → LLM → TTS, **giữ nguyên toàn bộ ưu điểm repo hiện tại**:
- PhoWhisper-medium vẫn là default (`stt_name = "phowhisper-medium"`) — không phá vỡ hành vi.
- Live transcription (`transcription.delta` trong lúc nói), barge-in (`CancelScope`), latency tracking, registry pattern, config runtime, mock LLM test scripts — không đổi.

## Yêu cầu

- Chọn backend STT qua `PipelineConfig.stt_name`: `"phowhisper-medium"` | `"wav2vec2-vi-250h"`.
- Backend wav2vec2 giữ live transcription (PROGRESSIVE → `PartialTranscriptionEvent`).
- Đổi backend runtime qua POST `/v1/config` (áp cho WS connection kế tiếp — cùng cơ chế llm/tts/vad hiện tại).
- Model: `nguyenvulebinh/wav2vec2-base-vietnamese-250h` (HF Hub).

## Kiến trúc

Pipeline 6 stage không đổi. Chỉ thay chỗ tạo STT handler trong `_make_handlers()` (s2s_pipeline.py):

```
hardcode WhisperSTTHandler → get_stt_handler(input_queue, output_queue, cfg, text_out, cancel_scope)
```

`get_stt_handler` đặt trong `handlers/registry.py` (cùng file với `get_llm_handler` — pattern đã có). Nhận `cfg: PipelineConfig`, đọc `cfg.stt_name`, `cfg.sample_rate`, `cfg.enable_live_transcription`:
- `"phowhisper-medium"` → `WhisperSTTHandler` (default, params như hiện tại)
- `"wav2vec2-vi-250h"` → `Wav2Vec2STTHandler`
- tên khác → `ValueError` (giống `get_llm_handler`)

## Component mới: handlers/wav2vec2_stt_handler.py

`Wav2Vec2STTHandler` — cùng interface với `WhisperSTTHandler`:

- Input: `VADAudio` (PCM16 16kHz), Output: `Transcription` (final) / `PartialTranscriptionEvent` qua `text_out` (progressive).
- **warmup lazy** (đúng pattern handler khác): 
  - `Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")`
  - `Wav2Vec2Processor.from_pretrained(...)` (tokenizer CTC trong processor)
  - Device: `cuda` nếu `torch.cuda.is_available()`, else CPU.
- **Decode CTC greedy**:
  1. PCM16 → float32: dùng lại `audio_utils.pcm16_to_float` (giữ 16kHz — wav2vec2 cần 16k, khớp pipeline rate).
  2. `model(input_values=torch.from_numpy(x)).logits` → `argmax(-1)`.
  3. `processor.batch_decode(pred_ids)` → text (collapse repeats + bỏ blank do tokenizer CTC).
- `AudioMode.PROGRESSIVE` → decode toàn bộ buffer nhận được → text không rỗng thì put `PartialTranscriptionEvent` vào `text_out`; không đẩy vào `output_queue`.
- `AudioMode.FINAL` → decode → text không rỗng → return `Transcription(text, language_code="vi", turn_id, turn_revision)`.
- `cancel_scope`: lấy `gen_start = cancel_scope.current()` đầu turn; sau decode nếu `current() != gen_start` → bỏ (chống transcript của turn đã bị cancel — giống whisper handler).

## Config + UI

server.py:
- GET `/v1/config`: bỏ hardcode `"model": "quocphu/PhoWhisper-ct2-FasterWhisper"` / subfolder; trả `"stt": {"name": c.stt_name}`.
- POST `/v1/config`: nhận `body.stt.name` → gán `c.stt_name` (validate trong {phowhisper-medium, wav2vec2-vi-250h}; sai → 400).

settings.html (`api/static/`):
- Thêm input `stt-name` (giá trị mặc định đọc từ GET `/v1/config`), gửi kèm trong POST body `stt: {name: ...}`.

## Dependencies

`pyproject.toml` — thêm optional extra (wav2vec2 dùng transformers, khác faster-whisper/CT2):

```toml
wav2vec2 = ["transformers", "torch"]
```

Import transformers/torch chỉ trong `warmup()` (lazy — đúng pattern `transformers_llm.py`), không import top-level để không phá khi thiếu dep.

## Error handling

- Warmup/model load fail → exception trong `process()` → `BaseHandler._process_item` bắt, log `[stt] error processing`, thread không chết (cơ chế sẵn có).
- Không có threshold hallucination như whisper (wav2vec2 không log-prob) — bỏ `no_speech_threshold`/`log_prob_threshold`; VAD đã lọc câu không-speech trước.

## Testing

- `pytest tests/` — test hiện có không đổi (mock flow, RAG).
- Thêm unit test registry: `get_stt_handler` trả đúng class theo tên backend (không load model).
- Script thật `scripts/test_stt_wav2vec2.py`: chạy model trên 1 file/đoạn audio tiếng Việt → in transcript (giống `test_stt.py` hiện tại). Không đưa model nặng (~240MB) vào pytest — chạy riêng.
- Không đụng: scripts mock LLM test (test_realtime_client.py, test_barge_in.py, test_tool_calling.py, ...).

## Ngoài phạm vi

- Không sửa VAD, TTS, LLM, WebRTC, RAG.
- Không thay default STT (PhoWhisper giữ default).
- Không thêm beam search/CTC decoder ngoài greedy (model base 250h; nếu chất lượng chưa đủ, cân nhắc `wav2vec2-base-vi` 800h — quyết định sau khi test thật).
