"""Test LLM OpenAI-compatible handler với mock server (không cần key thật)."""
import sys, time, threading
sys.path.insert(0, "/home/tdkhanh/s2s-vn/src")

import queue
from http.server import BaseHTTPRequestHandler, HTTPServer

from s2s_vn.LLM.llm_openai_compatible import OpenAICompatibleLLMHandler
from s2s_vn.pipeline.messages import GenerateResponseRequest, LLMResponseChunk


class MockLLMServer(BaseHTTPRequestHandler):
    """Chat completions stream: 3 chunk text + usage."""

    def do_POST(self):
        if self.path.endswith("/chat/completions"):
            # đọc body (không dùng)
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            body = (
                'data: {"choices":[{"delta":{"role":"assistant","content":"Xin "}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"chào bạn"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"."}}]}\n\n'
                'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":7}}\n\n'
                'data: [DONE]\n\n'
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())

    def log_message(self, *a):
        pass


server = HTTPServer(("127.0.0.1", 0), MockLLMServer)
port = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()
print(f"[test] mock LLM server @ :{port}")

iq, oq, eq = queue.Queue(), queue.Queue(), queue.Queue()
h = OpenAICompatibleLLMHandler(
    iq, oq,
    base_url=f"http://127.0.0.1:{port}/v1",
    api_key="sk-test",
    model_name="mock",
    text_out=eq,
)

t = threading.Thread(target=h.run, daemon=True)
t.start()
iq.put(GenerateResponseRequest(text="chào", language_code="vi", turn_id=1))
iq.put(None)  # stop

t.join(timeout=10)

print("[test] LLM outputs:")
text = ""
while not oq.empty():
    o = oq.get()
    print(f"  {type(o).__name__}: {getattr(o,'text_delta', getattr(o,'turn_id',o))}")
    if isinstance(o, LLMResponseChunk):
        text += o.text_delta
print(f"[test] ghép text: {text!r}")

print("[test] events:")
while not eq.empty():
    e = eq.get()
    print(f"  {type(e).__name__}: {getattr(e,'text', getattr(e,'output_tokens',e))}")
