"""Test live transcription: audio dài → VAD emit progressive → STT → transcription.delta."""
import sys, time, base64
sys.path.insert(0, "/home/tdkhanh/s2s-vn/src")

import numpy as np, scipy.signal
from s2s_vn.s2s_pipeline import PipelineConfig
from s2s_vn.api.openai_realtime.realtime_service import RealtimeService

# audio 2 câu dài (ghép 2 lần để có >2s nói liên tục)
pcm24 = open("/home/tdkhanh/test_input_24k.pcm", "rb").read()
pcm24_long = pcm24 + pcm24  # ~3.7s

cfg = PipelineConfig(
    llm_backend="local",
    llm_base_url="http://127.0.0.1:8081/v1",
    llm_api_key="sk-test",
    llm_model="mock",
    min_silence_ms=300,
    enable_live_transcription=True,
)
events_seen = []
svc = RealtimeService(cfg, on_event=lambda e: events_seen.append(e["type"]))
svc.start_drain()

for i in range(0, len(pcm24_long), 4096):
    svc.handle_event({"type": "input_audio_buffer.append",
                      "audio": base64.b64encode(pcm24_long[i:i+4096]).decode()})
    time.sleep(0.003)
svc.handle_event({"type": "input_audio_buffer.commit"})

# chờ response.done
deadline = time.time() + 40
while time.time() < deadline:
    if "response.done" in events_seen:
        break
    time.sleep(0.3)

deltas = events_seen.count("conversation.item.input_audio_transcription.delta")
completed = "conversation.item.input_audio_transcription.completed" in events_seen
print(f"[t] delta events: {deltas}")
print(f"[t] transcription.completed: {completed}")
print(f"[t] response.done: {'response.done' in events_seen}")
svc.stop()
print(f"[t] PASS" if deltas > 0 and completed else "[t] FAIL")
