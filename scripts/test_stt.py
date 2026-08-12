"""Test STT handler: gen audio Việt bằng VieNeu → transcribe bằng PhoWhisper."""
import sys
import time
sys.path.insert(0, "/home/tdkhanh/s2s-vn/src")

from s2s_vn.STT.whisper_stt_handler import WhisperSTTHandler
from s2s_vn.pipeline.messages import VADAudio
import numpy as np

TEXT = "Xin chào, tôi là trợ lý giọng nói tiếng Việt."
SR = 48000

# 1. Gen audio bằng VieNeu
print("[test] gen audio bằng VieNeu ...")
from vieneu import Vieneu
v = Vieneu(mode="v3turbo")
t0 = time.time()
a = v.infer(TEXT, voice="Trúc Ly")
print(f"[test] gen xong {time.time()-t0:.2f}s, {len(a)} mẫu @{SR}Hz")

# 2. Resample 48k → 16k (thủ công linear)
import scipy.signal
a16 = scipy.signal.resample_poly(a, 16000, SR)
pcm16 = (a16 * 32767).astype(np.int16).tobytes()
print(f"[test] pcm16 {len(pcm16)} bytes @16kHz")

# 3. STT transcribe
print("[test] transcribe bằng PhoWhisper-medium ...")
import queue
iq, oq = queue.Queue(), queue.Queue()
h = WhisperSTTHandler(iq, oq)
t0 = time.time()
out = h.process(VADAudio(audio=pcm16, mode="final", turn_id=1, sample_rate=16000))
print(f"[test] STT {time.time()-t0:.2f}s")
print(f"[test] kết quả: {out}")
