"""End-to-end: audio Việt → VAD+STT thật → LLM (mock server) → TTS thật → audio out.

LLM dùng mock OpenAI-compatible server local (không cần key thật).
"""
import sys, time, queue, threading
sys.path.insert(0, "/home/tdkhanh/s2s-vn/src")

import numpy as np
import scipy.signal
from http.server import BaseHTTPRequestHandler, HTTPServer

from s2s_vn.s2s_pipeline import S2SPipeline, PipelineConfig


class MockLLMServer(BaseHTTPRequestHandler):
    """Chat completions stream trả lời tiếng Việt."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        chunks = ["Xin ", "chào bạn", ", ", "tôi là trợ lý", " giọng nói", "."]
        parts = [f'data: {{"choices":[{{"delta":{{"content":"{c}"}}}}]}}\n\n' for c in chunks]
        parts.append('data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":9}}\n\n')
        parts.append("data: [DONE]\n\n")
        body = "".join(parts).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


# mock LLM server
srv = HTTPServer(("127.0.0.1", 0), MockLLMServer)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
print(f"[test] mock LLM @ :{port}")

# gen audio Việt
print("[test] gen audio ...")
from vieneu import Vieneu
v = Vieneu(mode="v3turbo")
a = v.infer("Xin chào, tôi là trợ lý giọng nói tiếng Việt.", voice="Trúc Ly")
a16 = scipy.signal.resample_poly(a, 16000, 48000)
pcm16 = (a16 * 32767).astype(np.int16).tobytes()
print(f"[test] audio {len(pcm16)/32000:.2f}s")

cfg = PipelineConfig(
    min_silence_ms=300,
    llm_backend="local",
    llm_base_url=f"http://127.0.0.1:{port}/v1",
    llm_api_key="sk-test",
    llm_model="mock",
)
p = S2SPipeline(cfg)
p.start()
t_start = time.time()
try:
    for i in range(0, len(pcm16), 1024):
        p.send_audio(pcm16[i:i+1024])
        time.sleep(0.001)
    p.finish_turn()

    # thu audio out (TTS thật → nhiều chunk AudioOutput). Đợi TTS yên 2s (2 turn).
    outs = []
    deadline = time.time() + 60
    last_new = time.time()
    got_any = False
    while time.time() < deadline:
        out = p.wait_audio_out(timeout=0.5)
        if out is None:
            # chỉ break khi đã có audio rồi im 3s (chờ TTS warmup 7s lần đầu)
            if got_any and time.time() - last_new > 3.0:
                break
            continue
        outs.append(out)
        got_any = True
        last_new = time.time()
    print(f"[test] +{time.time()-t_start:.1f}s audio out: {len(outs)} items")
    audio_bytes = sum(getattr(o, "audio", o).__len__() for o in outs
                      if hasattr(o, "audio") or isinstance(o, bytes))
    print(f"[test] tổng audio: {audio_bytes/32000:.2f}s @16k")

    time.sleep(0.3)
    events = []
    while not p.queues.text_out.empty():
        events.append(p.queues.text_out.get())
    print(f"[test] events: {[type(e).__name__ for e in events]}")
    for e in events:
        if type(e).__name__ in ("TranscriptionCompletedEvent", "AssistantTextEvent"):
            print(f"  {type(e).__name__}: {e.text!r}")
        if type(e).__name__ == "TokenUsageEvent":
            print(f"  TokenUsage: in={e.input_tokens} out={e.output_tokens}")
finally:
    p.stop()
    print(f"[test] threads còn sống: {p.threads.alive_count}")
