"""Test tool calling: LLM trả tool_call → FunctionCallEvent → client trả output
→ LLM gọi lại → text → TTS audio.

Chạy service + pipeline trực tiếp, không qua WS.
"""
import sys, time
sys.path.insert(0, "/home/tdkhanh/s2s-vn/src")

from s2s_vn.s2s_pipeline import PipelineConfig
from s2s_vn.api.openai_realtime.realtime_service import RealtimeService
from s2s_vn.pipeline.messages import GenerateResponseRequest
from s2s_vn.pipeline.events import FunctionCallEvent

# dùng LLM tools server :8082
cfg = PipelineConfig(
    llm_backend="local",
    llm_base_url="http://127.0.0.1:8082/v1",
    llm_api_key="sk-test",
    llm_model="mock",
    min_silence_ms=300,
    tts_streaming=True,
)

events_seen = []
svc = RealtimeService(cfg, on_event=lambda e: events_seen.append(e["type"]))
svc.start_drain()

# set tools
svc.handle_event({"type": "session.update", "session": {"tools": [
    {"type": "function", "name": "get_weather",
     "description": "Lấy thời tiết", "parameters": {"type": "object",
     "properties": {"city": {"type": "string"}}, "required": ["city"]}},
]}})

# 1. đưa user text → LLM → tool_call
svc.pipeline.queues.text_prompt.put(GenerateResponseRequest(
    text="Hà Nội thời tiết thế nào?", language_code="vi", turn_id=1))

# chờ FunctionCallEvent
deadline = time.time() + 15
fc_event = None
while time.time() < deadline:
    if hasattr(svc, "_fc") and svc._fc:
        fc_event = svc._fc
        break
    # drain events qua text_out — chờ pipeline xử lý
    time.sleep(0.2)
    # lấy FunctionCallEvent từ queue text_out qua events_seen
    if "response.function_call_arguments.done" in events_seen:
        break

print(f"[t] events đã thấy: {events_seen}")

# 2. giả client: trả function_call_output
svc.handle_event({"type": "conversation.item.create", "item": {
    "type": "function_call_output",
    "call_id": "call_abc123",
    "output": "28 độ C, trời nắng",
}})

# 3. chờ response thật (text → TTS → audio)
deadline = time.time() + 30
while time.time() < deadline:
    if "response.done" in events_seen:
        break
    time.sleep(0.3)
print(f"[t] final events: {[e for e in events_seen if 'response' in e or 'function' in e]}")
print(f"[t] có function_call: {'response.function_call_arguments.done' in events_seen}")
print(f"[t] có text sau tool: {'response.output_audio_transcript.delta' in events_seen}")
print(f"[t] có response.done: {'response.done' in events_seen}")
svc.stop()
