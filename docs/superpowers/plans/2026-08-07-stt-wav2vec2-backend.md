# STT registry + wav2vec2 backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thêm backend STT `nguyenvulebinh/wav2vec2-base-vietnamese-250h` (Wav2Vec2 CTC greedy) vào pipeline, chọn qua `PipelineConfig.stt_name`, giữ PhoWhisper-medium làm default.

**Architecture:** Thêm `get_stt_handler()` vào `handlers/registry.py` (pattern giống `get_llm_handler`), handler mới `Wav2Vec2STTHandler` cùng interface với `WhisperSTTHandler` (`VADAudio → Transcription` / `PartialTranscriptionEvent`), `_make_handlers()` gọi registry thay vì hardcode. Config runtime qua GET/POST `/v1/config` + `settings.html`.

**Tech Stack:** transformers (`Wav2Vec2ForCTC`, `Wav2Vec2Processor`), torch, FastAPI TestClient, pytest.

## Global Constraints

- Repo **không phải git repository** — BỎ TOÀN BỘ bước commit; dừng sau bước test pass.
- Comment/docstring bằng tiếng Việt (convention repo).
- Model chính xác: `nguyenvulebinh/wav2vec2-base-vietnamese-250h`.
- Sample rate pipeline: 16000 (không đổi).
- Default `stt_name = "phowhisper-medium"` — hành vi hiện tại không đổi.
- Import torch/transformers chỉ trong `warmup()` (lazy — pattern `transformers_llm.py`), không import top-level.
- Không đụng: VAD, TTS, LLM, WebRTC, RAG, mock LLM scripts.
- Test thật (tải model ~240MB) KHÔNG đưa vào pytest — để script riêng.
- Pipeline handler order bị hardcode theo index trong `realtime_service.py` (`handlers[0]`=VAD, `handlers[3]`=LLM) — không đổi thứ tự danh sách handler.

---

### Task 1: Wav2Vec2STTHandler + unit test (mock model) + dependency

**Files:**
- Create: `src/s2s_vn/handlers/wav2vec2_stt_handler.py`
- Create: `tests/test_wav2vec2_stt.py`
- Modify: `pyproject.toml` (thêm optional extra `wav2vec2`)

**Interfaces:**
- Produces: class `Wav2Vec2STTHandler(input_queue, output_queue, name="stt", model_name="nguyenvulebinh/wav2vec2-base-vietnamese-250h", device="auto", sample_rate=16000, language="vi", text_out=None, enable_live_transcription=True, cancel_scope=None)` — subclass `BaseHandler`, `process(item: VADAudio) -> Transcription | None`, progressive put `PartialTranscriptionEvent` vào `text_out`. Attr `_model` (None trước warmup), `_processor`.

- [ ] **Step 1: Write the failing test**

`tests/test_wav2vec2_stt.py`:

```python
"""Test Wav2Vec2STTHandler với model giả (không tải transformers model thật)."""

import queue
from types import SimpleNamespace

import torch

from s2s_vn.handlers.wav2vec2_stt_handler import Wav2Vec2STTHandler
from s2s_vn.pipeline.events import PartialTranscriptionEvent
from s2s_vn.pipeline.messages import AudioMode, Transcription, VADAudio


class FakeModel:
    """Giả Wav2Vec2ForCTC: nhận input_values, trả logits (1, T, V)."""

    def __init__(self, logits):
        self._logits = logits

    def __call__(self, input_values):
        return SimpleNamespace(logits=self._logits)

    def eval(self):
        return self

    def to(self, device):
        return self


def make_handler(enable_live=True):
    iq, oq = queue.Queue(), queue.Queue()
    h = Wav2Vec2STTHandler(iq, oq, device="cpu",
                           enable_live_transcription=enable_live)
    # model giả: vocab=3, T=8; argmax → token 1 lặp → collapse → "xin chào"
    h._model = FakeModel(torch.zeros(1, 8, 3))
    h._processor = SimpleNamespace(decode=lambda ids: "xin chào")
    return h, oq


def test_final_transcription():
    h, oq = make_handler()
    out = h.process(VADAudio(audio=b"\x00" * 3200, mode=AudioMode.FINAL,
                             turn_id=1))
    assert isinstance(out, Transcription)
    assert out.text == "xin chào"
    assert out.turn_id == 1
    assert oq.empty()


def test_progressive_emits_partial_event():
    h, oq = make_handler()
    text_out = queue.Queue()
    h.text_out = text_out
    out = h.process(VADAudio(audio=b"\x00" * 3200, mode=AudioMode.PROGRESSIVE,
                             turn_id=2))
    assert out is None
    ev = text_out.get_nowait()
    assert isinstance(ev, PartialTranscriptionEvent)
    assert ev.text == "xin chào"
    assert oq.empty()


def test_progressive_disabled_returns_none():
    h, oq = make_handler(enable_live=False)
    text_out = queue.Queue()
    h.text_out = text_out
    out = h.process(VADAudio(audio=b"\x00" * 3200, mode=AudioMode.PROGRESSIVE,
                             turn_id=3))
    assert out is None
    assert text_out.empty()


def test_cancelled_turn_returns_empty():
    from s2s_vn.pipeline.messages import CancelScope

    h, oq = make_handler()
    h.cancel_scope = CancelScope()
    h.cancel_scope.cancel()  # current = 1
    # gen_start=0 cũ hơn current → turn đã bị cancel → bỏ
    text = h._decode(b"\x00" * 3200, gen_start=0)
    assert text == ""
```
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wav2vec2_stt.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 's2s_vn.handlers.wav2vec2_stt_handler'`

- [ ] **Step 3: Write the handler**

`src/s2s_vn/handlers/wav2vec2_stt_handler.py`:

```python
"""STT handler dùng Wav2Vec2 CTC (nguyenvulebinh/wav2vec2-base-vietnamese-250h)."""

from __future__ import annotations

import queue

import numpy as np

from ..pipeline.base_handler import BaseHandler
from ..pipeline.events import PartialTranscriptionEvent
from ..pipeline.messages import AudioMode, Transcription, VADAudio


class Wav2Vec2STTHandler(BaseHandler):
    """VADAudio → Transcription.

    Wav2Vec2ForCTC greedy decode. Model: nguyenvulebinh/wav2vec2-base-vietnamese-250h
    (base, 250h tiếng Việt). Không có log-prob như whisper → không lọc
    hallucination bằng threshold; VAD đã cắt câu không-speech trước.
    """

    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        name: str = "stt",
        model_name: str = "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
        device: str = "auto",  # auto → CUDA nếu có
        sample_rate: int = 16000,
        language: str = "vi",
        text_out: queue.Queue | None = None,
        enable_live_transcription: bool = True,
        cancel_scope=None,
    ):
        super().__init__(input_queue, output_queue, name)
        self.model_name = model_name
        self.device = device
        self.sample_rate = sample_rate
        self.language = language
        self.text_out = text_out
        self.enable_live_transcription = enable_live_transcription
        self.cancel_scope = cancel_scope
        self._model = None
        self._processor = None

    def warmup(self) -> None:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self._model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device)

    def process(self, item):
        if not isinstance(item, VADAudio):
            return None
        if self._model is None:
            self.warmup()

        gen_start = self.cancel_scope.current() if self.cancel_scope else 0

        # live transcription: progressive → PartialTranscriptionEvent (không vào LLM)
        if item.mode == AudioMode.PROGRESSIVE:
            if not self.enable_live_transcription:
                return None
            text = self._decode(item.audio, gen_start)
            if text and self.text_out:
                self.text_out.put(PartialTranscriptionEvent(
                    text=text, turn_id=item.turn_id))
            return None

        # final: Transcription → LLM
        text = self._decode(item.audio, gen_start)
        if not text:
            return None
        return Transcription(
            text=text,
            language_code=self.language,
            turn_id=item.turn_id,
            turn_revision=item.turn_revision,
        )

    def _decode(self, pcm16: bytes, gen_start: int) -> str:
        import torch

        if not pcm16:
            return ""
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        input_values = torch.from_numpy(audio).unsqueeze(0)  # (1, T) — Wav2Vec2 cần batch dim
        if self.device == "cuda":
            input_values = input_values.cuda()
        with torch.no_grad():
            logits = self._model(input_values).logits  # (1, T, vocab)
        pred_ids = logits.argmax(dim=-1)[0]
        # barge-in: turn đã cancel → bỏ (giống whisper handler)
        if self.cancel_scope and self.cancel_scope.current() != gen_start:
            return ""
        return self._processor.decode(pred_ids.tolist()).strip()
```

- [ ] **Step 4: Add optional dependency**

`pyproject.toml` — trong `[project.optional-dependencies]`, thêm dòng:

```toml
wav2vec2 = ["transformers", "torch"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_wav2vec2_stt.py tests/test_pipeline.py -q`
Expected: PASS (7 tests: 4 mới + 3 mock flow cũ không đổi)

---

### Task 2: Registry `get_stt_handler` + wiring pipeline

**Files:**
- Modify: `src/s2s_vn/handlers/registry.py` (import Wav2Vec2STTHandler, thêm `get_stt_handler`)
- Modify: `src/s2s_vn/pipeline/s2s_pipeline.py` (`_make_handlers` dùng registry)
- Create: `tests/test_stt_registry.py`

**Interfaces:**
- Consumes: `Wav2Vec2STTHandler` từ Task 1; `WhisperSTTHandler` (có sẵn).
- Produces: `get_stt_handler(input_queue, output_queue, cfg, text_out=None, cancel_scope=None) -> BaseHandler` — đọc `cfg.stt_name`, `cfg.sample_rate`, `cfg.enable_live_transcription`; `ValueError` cho tên lạ. Registry nhận `cfg` không type-hint `PipelineConfig` (tránh circular import `pipeline.s2s_pipeline ↔ registry`); dùng docstring.

- [ ] **Step 1: Write the failing test**

`tests/test_stt_registry.py`:

```python
"""Test registry STT: chọn đúng handler theo tên backend."""

import queue

import pytest

from s2s_vn.handlers.registry import get_stt_handler
from s2s_vn.pipeline.s2s_pipeline import PipelineConfig


def make_cfg(stt_name):
    return PipelineConfig(stt_name=stt_name)


def test_default_backend_is_whisper():
    q = queue.Queue()
    h = get_stt_handler(q, q, make_cfg("phowhisper-medium"))
    assert type(h).__name__ == "WhisperSTTHandler"


def test_wav2vec2_backend_selected():
    q = queue.Queue()
    h = get_stt_handler(q, q, make_cfg("wav2vec2-vi-250h"))
    assert type(h).__name__ == "Wav2Vec2STTHandler"


def test_unknown_backend_raises():
    q = queue.Queue()
    with pytest.raises(ValueError):
        get_stt_handler(q, q, make_cfg("not-a-backend"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stt_registry.py -v`
Expected: FAIL — `ImportError: cannot import name 'get_stt_handler'`

- [ ] **Step 3: Add `get_stt_handler` to registry**

`src/s2s_vn/handlers/registry.py` — thêm import cùng dòng với các import LLM:

```python
from .wav2vec2_stt_handler import Wav2Vec2STTHandler
from .whisper_stt_handler import WhisperSTTHandler
```

Cuối file (sau `get_llm_handler`):

```python
def get_stt_handler(
    input_queue,
    output_queue,
    cfg,
    text_out=None,
    cancel_scope=None,
) -> BaseHandler:
    """Trả STT handler theo cfg.stt_name (cfg: PipelineConfig — không import
    để tránh circular import pipeline.s2s_pipeline ↔ registry).

    - phowhisper-medium: faster-whisper CT2 (default)
    - wav2vec2-vi-250h: Wav2Vec2 CTC greedy (transformers)
    """
    if cfg.stt_name == "phowhisper-medium":
        return WhisperSTTHandler(
            input_queue, output_queue,
            text_out=text_out,
            enable_live_transcription=cfg.enable_live_transcription,
            cancel_scope=cancel_scope,
        )
    if cfg.stt_name == "wav2vec2-vi-250h":
        return Wav2Vec2STTHandler(
            input_queue, output_queue,
            sample_rate=cfg.sample_rate,
            text_out=text_out,
            enable_live_transcription=cfg.enable_live_transcription,
            cancel_scope=cancel_scope,
        )
    raise ValueError(f"STT backend không hỗ trợ: {cfg.stt_name}")
```

- [ ] **Step 4: Wire into pipeline**

`src/s2s_vn/pipeline/s2s_pipeline.py`:
- Đổi import dòng 15: `from ..handlers.registry import LLMConfig, get_llm_handler` → `from ..handlers.registry import LLMConfig, get_llm_handler, get_stt_handler`
- Xoá import `from ..handlers.whisper_stt_handler import WhisperSTTHandler` (dòng 17)
- Thay block tạo STT (hiện là `stt = WhisperSTTHandler(q.spoken_prompt, q.stt_output, text_out=q.text_out, enable_live_transcription=cfg.enable_live_transcription, cancel_scope=q.cancel_scope)`) bằng:

```python
    stt = get_stt_handler(
        q.spoken_prompt, q.stt_output, cfg,
        text_out=q.text_out,
        cancel_scope=q.cancel_scope,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/ -q`
Expected: PASS (7 tests cũ + 7 mới, không load model thật; ~25s do test_rag tải embedding)

---

### Task 3: Config endpoints + settings.html

**Files:**
- Modify: `src/s2s_vn/api/server.py` (GET + POST `/v1/config`)
- Modify: `src/s2s_vn/api/static/settings.html` (input stt-name + JS)
- Create: `tests/test_config_api.py`

**Interfaces:**
- Consumes: `PipelineConfig.stt_name` từ Task 2.
- Produces: GET `/v1/config` trả `"stt": {"name": c.stt_name}`; POST `/v1/config` nhận `body.stt.name`, validate 2 giá trị, sai → HTTP 400.

- [ ] **Step 1: Write the failing test**

`tests/test_config_api.py`:

```python
"""Test REST config: STT backend qua GET/POST /v1/config."""

from fastapi.testclient import TestClient

from s2s_vn.api.server import create_app
from s2s_vn.pipeline.s2s_pipeline import PipelineConfig


def client():
    return TestClient(create_app(PipelineConfig()))


def test_get_config_returns_stt_name():
    r = client().get("/v1/config")
    assert r.status_code == 200
    assert r.json()["stt"]["name"] == "phowhisper-medium"


def test_post_config_changes_stt_name():
    c = client()
    r = c.post("/v1/config", json={"stt": {"name": "wav2vec2-vi-250h"}})
    assert r.status_code == 200
    r = c.get("/v1/config")
    assert r.json()["stt"]["name"] == "wav2vec2-vi-250h"


def test_post_config_rejects_unknown_stt():
    r = client().post("/v1/config", json={"stt": {"name": "bogus"}})
    assert r.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config_api.py -v`
Expected: FAIL — GET trả `stt.model` (hardcode cũ), test assert `stt.name` fail

- [ ] **Step 3: Fix GET /v1/config**

`src/s2s_vn/api/server.py`, trong `get_config()` — thay block `"stt": {...}`:

```python
            "stt": {"name": c.stt_name},
```

- [ ] **Step 4: Fix POST /v1/config**

`src/s2s_vn/api/server.py`, trong `update_config()` — thêm sau block vad:

```python
        stt = body.get("stt", {})
        if "name" in stt:
            if stt["name"] not in ("phowhisper-medium", "wav2vec2-vi-250h"):
                raise HTTPException(status_code=400,
                                    detail=f"STT backend không hỗ trợ: {stt['name']}")
            c.stt_name = stt["name"]
```

- [ ] **Step 5: Update settings.html**

`src/s2s_vn/api/static/settings.html`:

1. Trong card STT, sau field `stt-subfolder` (cuối block `.field` trước `</div>` đóng card):

```html
    <div class="field">
      <label>Backend (phowhisper-medium | wav2vec2-vi-250h)</label>
      <input id="stt-name" value="phowhisper-medium">
    </div>
```

2. Trong `fetch('/v1/config')` handler, trong `if (cfg.stt) {` — thêm:

```js
    document.getElementById('stt-name').value = cfg.stt.name || 'phowhisper-medium';
```

3. Trong payload POST (`stt: {` object) — thêm dòng đầu:

```js
    stt: {
      name: document.getElementById('stt-name').value,
      model: document.getElementById('stt-model').value,
      subfolder: document.getElementById('stt-subfolder').value,
    },
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_config_api.py -q`
Expected: PASS (3 tests)

---

### Task 4: Script thật + cập nhật CLAUDE.md

**Files:**
- Create: `scripts/test_stt_wav2vec2.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `Wav2Vec2STTHandler` từ Task 1.
- Produces: script chạy thật: audio PCM16 16k → in transcript wav2vec2.

- [ ] **Step 1: Write the script**

`scripts/test_stt_wav2vec2.py` (theo pattern `test_stt.py` — sys.path + đo lường thời gian):

```python
"""Test STT handler wav2vec2: đọc test_input_24k.pcm → resample 16k → transcribe.

Model nguyenvulebinh/wav2vec2-base-vietnamese-250h tự tải qua HF cache
lần chạy đầu (~240MB).
"""
import queue
import sys
import time
sys.path.insert(0, "/home/tdkhanh/s2s-vn/src")

import numpy as np
import scipy.signal

from s2s_vn.handlers.wav2vec2_stt_handler import Wav2Vec2STTHandler
from s2s_vn.pipeline.messages import VADAudio

SR = 16000

# 1. Đọc audio mẫu 24k có sẵn → resample 16k → PCM16
pcm24 = open("scripts/test_input_24k.pcm", "rb").read()
a24 = np.frombuffer(pcm24, dtype=np.int16).astype(np.float32) / 32768.0
a16 = scipy.signal.resample_poly(a24, SR, 24000)
pcm16 = (a16 * 32767).astype(np.int16).tobytes()
print(f"[test] pcm16 {len(pcm16)} bytes @16kHz")

# 2. Transcribe bằng wav2vec2
print("[test] transcribe bằng wav2vec2-base-vietnamese-250h ...")
iq, oq = queue.Queue(), queue.Queue()
h = Wav2Vec2STTHandler(iq, oq)
t0 = time.time()
out = h.process(VADAudio(audio=pcm16, mode="final", turn_id=1, sample_rate=SR))
print(f"[test] STT {time.time()-t0:.2f}s")
print(f"[test] kết quả: {out.text if out else None}")
```

- [ ] **Step 2: Run the script with mock LLM server + full pipeline**

Chạy end-to-end (wav2vec2 qua pipeline thật, LLM mock — kiểm tra live transcription + barge-in không vỡ):

```bash
python scripts/mock_llm_server.py 8081 &
PYTHONPATH=src python -m s2s_vn.api.server --llm-backend local \
  --llm-base-url http://127.0.0.1:8081/v1 &
python scripts/test_realtime_client.py
kill %1 %2
```

Expected: transcript + audio reply chạy được. (Nếu muốn test riêng STT: `python scripts/test_stt_wav2vec2.py` — in kết quả transcribe.)

- [ ] **Step 3: Update CLAUDE.md**

`CLAUDE.md`:

1. Dòng install — thêm extra:
```
pip install -e ".[realtime]"          # thêm "[stt,tts,dev]" khi cần model + pytest/ruff
# STT wav2vec2: pip install -e ".[wav2vec2]" (transformers + torch)
```

2. Dòng test script — thêm `test_stt_wav2vec2.py` vào danh sách (sau `test_stt.py`):
```
#   test_live_transcription.py, test_vad_stt.py, test_stt.py, test_llm.py,
```

3. Kiến trúc pipeline — đổi dòng STT:
```
audio_in → [VAD] Silero v5 ONNX → [STT] registry (phowhisper-medium | wav2vec2-vi-250h) → [Notifier] mock
```

4. Section Handlers — sau bullet `registry.py` (LLM), thêm:
```
- `get_stt_handler` (cùng file): chọn STT theo `cfg.stt_name` — `phowhisper-medium` → `WhisperSTTHandler` (faster-whisper CT2, default), `wav2vec2-vi-250h` → `Wav2Vec2STTHandler` (Wav2Vec2ForCTC greedy, transformers, device auto→CUDA). Không type-hint PipelineConfig trong signature (tránh circular import pipeline↔registry).
```

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -q`
Expected: PASS (7 cũ + 7 mới + 3 config = 17 tests; ~25s)

---

## Self-review ghi chú (fix inline)

- [x] Spec coverage: mọi yêu cầu spec có task — registry (T2), handler (T1), config+UI (T3), deps (T1 step 4), testing (T1/T3 unit + T4 script thật). Giữ live transcription, cancel_scope — trong handler code T1.
- [x] Placeholder scan: mọi step có code đầy đủ, không TBD.
- [x] Type consistency: `get_stt_handler(input_queue, output_queue, cfg, text_out=None, cancel_scope=None)` — T2 định nghĩa, T2 step 4 gọi khớp; `Wav2Vec2STTHandler(...)` params nhất quán T1 ↔ T2 step 3 (sample_rate, text_out, enable_live_transcription, cancel_scope).
