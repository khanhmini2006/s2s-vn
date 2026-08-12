"""Test TransformersLLMHandler với Qwen2.5-3B-Instruct local."""
import sys, time, queue, threading
sys.path.insert(0, "/home/tdkhanh/s2s-vn/src")

from s2s_vn.LLM.transformers_llm import TransformersLLMHandler
from s2s_vn.pipeline.messages import GenerateResponseRequest, LLMResponseChunk, EndOfResponse

iq, oq, eq = queue.Queue(), queue.Queue(), queue.Queue()
h = TransformersLLMHandler(iq, oq, text_out=eq,
                           model_name="Qwen/Qwen2.5-3B-Instruct",
                           device="cuda", max_tokens=128, temperature=0.3)

t = threading.Thread(target=h.run, daemon=True)
t.start()

t0 = time.time()
iq.put(GenerateResponseRequest(
    text="Xin chào, bạn tên là gì và hôm nay thời tiết thế nào?", language_code="vi", turn_id=1))

# thu outputs
text = ""
first_delta = None
while True:
    out = oq.get(timeout=60)
    if isinstance(out, LLMResponseChunk):
        if first_delta is None:
            first_delta = time.time() - t0
            print(f"[t] token đầu tiên sau {first_delta*1000:.0f}ms")
        text += out.text_delta
    elif isinstance(out, EndOfResponse):
        break

print(f"[t] Tổng: {time.time()-t0:.1f}s")
print(f"[t] Text: {text!r}")
assert len(text) > 10, "text quá ngắn"
print("[t] PASS")
iq.put(None)
