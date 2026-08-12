"""Test Piper TTS handler: TTSInput → AudioOutput PCM16 (streaming theo câu)."""
import sys, time, queue
sys.path.insert(0, "/home/tdkhanh/s2s-vn/src")

from s2s_vn.TTS.piper_tts_handler import PiperTTSHandler
from s2s_vn.pipeline.messages import TTSInput, AudioOutput
from s2s_vn.backend_registry import TTS_BACKENDS

model_path = TTS_BACKENDS["piper-vie"].model_name

iq, oq = queue.Queue(), queue.Queue()
h = PiperTTSHandler(iq, oq, model_path=model_path, config_path=model_path + ".json")

t0 = time.time()
print("[test] warmup ...")
h.warmup()
print(f"[test] warmup {time.time()-t0:.1f}s")

t0 = time.time()
h.process(TTSInput(text="Xin chào, tôi là trợ lý giọng nói tiếng Việt. Rất vui được hỗ trợ bạn hôm nay.", turn_id=1))
n = 0
total = 0
while not oq.empty():
    out = oq.get()
    if not isinstance(out, AudioOutput):
        continue  # ResponseDone cuối cùng
    n += 1
    total += len(out.audio)
    assert out.sample_rate == 16000
    assert len(out.audio) % 2 == 0

print(f"[test] {n} chunks (theo câu), {total/32000:.2f}s audio @16k")
print(f"[test] total gen time: {time.time()-t0:.2f}s")
print("[test] OK")
