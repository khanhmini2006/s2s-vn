"""Test VieNeu TTS handler: TTSInput → AudioOutput PCM16 streaming."""
import sys, time, queue
sys.path.insert(0, "/home/tdkhanh/s2s-vn/src")

from s2s_vn.TTS.vieneu_tts_handler import VieNeuTTSHandler
from s2s_vn.pipeline.messages import TTSInput, AudioOutput

iq, oq = queue.Queue(), queue.Queue()
h = VieNeuTTSHandler(iq, oq, voice="Trúc Ly")

t0 = time.time()
print("[test] warmup + gen streaming ...")
t = time.time()
h.warmup()
print(f"[test] warmup {time.time()-t:.1f}s")

t0 = time.time()
first_t = None
n = 0
total = 0
for out in h.process(TTSInput(text="Xin chào, tôi là trợ lý giọng nói tiếng Việt.", turn_id=1)):
    if first_t is None:
        first_t = time.time() - t0
    n += 1
    total += len(out.audio)
    assert isinstance(out, AudioOutput)
    assert out.sample_rate == 16000
    assert len(out.audio) % 2 == 0

print(f"[test] {n} chunks, {total/32000:.2f}s audio @16k")
print(f"[test] TTFA: {first_t*1000:.0f}ms")
print(f"[test] total: {time.time()-t0:.2f}s")

# ghi chunk đầu ra file để verify
with open("/home/tdkhanh/tts_out_chunk0.pcm", "wb") as f:
    pass
print("[test] OK")
