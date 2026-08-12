"""Test VAD + STT: VieNeu gen audio → VAD cắt utterance → PhoWhisper transcribe."""
import sys, time
sys.path.insert(0, "/home/tdkhanh/s2s-vn/src")

import numpy as np
import scipy.signal

from s2s_vn.VAD.vad_handler import SileroVADHandler
from s2s_vn.STT.whisper_stt_handler import WhisperSTTHandler
from s2s_vn.pipeline.messages import PIPELINE_END

TEXT = "Xin chào, tôi là trợ lý giọng nói tiếng Việt."
SR = 48000

print("[test] gen audio ...")
from vieneu import Vieneu
v = Vieneu(mode="v3turbo")
a = v.infer(TEXT, voice="Trúc Ly")
a16 = scipy.signal.resample_poly(a, 16000, SR)
pcm16 = (a16 * 32767).astype(np.int16).tobytes()
print(f"[test] audio {len(pcm16)/32000:.2f}s @16kHz, {len(pcm16)} bytes")

import queue, threading

iq, oq, eq = queue.Queue(), queue.Queue(), queue.Queue()
vad = SileroVADHandler(iq, oq, text_out=eq, min_silence_ms=300)
stt = WhisperSTTHandler(oq, queue.Queue())

# chạy VAD trong thread
t_vad = threading.Thread(target=vad.run, daemon=True)
t_vad.start()

# feed audio thành chunk 1024 bytes (512 mẫu)
print("[test] feed audio vào VAD ...")
for i in range(0, len(pcm16), 1024):
    iq.put(pcm16[i:i+1024])
    time.sleep(0.001)
iq.put(PIPELINE_END)  # kết thúc, flush phần còn lại

# chờ VAD xử lý hết
t_vad.join(timeout=10)
print(f"[test] VAD thread dead={not t_vad.is_alive()}")

# transcribe tất cả utterances
n = 0
while not oq.empty():
    utt = oq.get()
    n += 1
    print(f"[test] utterance {n}: {len(utt.audio)/32000:.2f}s, {len(utt.audio)} bytes")
    t0 = time.time()
    tr = stt.process(utt)
    print(f"[test]   STT {time.time()-t0:.2f}s: {tr}")

# events
events = []
while not eq.empty():
    events.append(eq.get())
print(f"[test] events: {[type(e).__name__ for e in events]}")
