"""E2E: connect WS thật → đợi model_ready → inject audio mẫu → STT→LLM→TTS → response.

Verify chuỗi pipeline hoạt động với config hiện tại (param settings đã đổi).
Không dùng mic — inject audio mẫu (24k → resample 16k, protocol = pipeline).
"""
import asyncio, json, base64
import websockets

PCM_PATH = "/home/tdkhanh/s2s-vn/scripts/test_input_24k.pcm"

async def main(url):
    # audio mẫu 24k → resample xuống 16k (protocol = pipeline = 16k)
    import numpy as np
    from scipy.signal import resample_poly
    pcm24 = open(PCM_PATH, "rb").read()
    x = np.frombuffer(pcm24, dtype=np.int16).astype(np.float32) / 32768.0
    y = resample_poly(x, 16000, 24000)
    pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes()
    print(f"[e2e] audio: {len(pcm)} bytes = {len(pcm)/32000:.2f}s 16kHz")
    async with websockets.connect(url) as ws:
        # session.created
        ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
        print(f"[e2e] {ev['type']} | session.model = {ev['session']['model']}")
        # đợi warmup model xong
        while True:
            ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=300))
            t = ev["type"]
            if t == "server.model_ready":
                print("[e2e] model_ready — model đã warmup")
                break
            if t == "error":
                print(f"[e2e] lỗi khi warmup: {ev}")
                return
            print(f"[e2e]   (chờ) {t}")
        # gửi audio thành chunk nhỏ
        print("[e2e] gửi audio...")
        chunk = 4096
        for i in range(0, len(pcm), chunk):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm[i:i+chunk]).decode(),
            }))
            await asyncio.sleep(0.005)
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        # thu events tới response.done
        transcript = ""
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=180)
            ev = json.loads(raw)
            t = ev["type"]
            extra = ""
            if t == "conversation.item.input_audio_transcription.delta":
                transcript += ev.get("delta", "")
                extra = repr(ev.get("delta", ""))
            elif t == "conversation.item.input_audio_transcription.completed":
                transcript = ev.get("transcript", "")
                extra = repr(transcript)
            elif t == "response.output_audio_transcript.delta":
                extra = repr(ev.get("delta", ""))
            elif t == "response.output_audio.delta":
                extra = f"audio={len(ev['delta'])}b"
            elif t == "response.done":
                st = ev["response"]["status"]
                usage = ev["response"].get("usage")
                print(f"[e2e] response.done status={st} usage={usage}")
                print(f"[e2e] STT transcript = {transcript!r}")
                print(f"[e2e] DONE")
                return
            elif t == "error":
                print(f"[e2e] ERROR: {json.dumps(ev, ensure_ascii=False)}")
                return
            print(f"[e2e]   {t} {extra}")

if __name__ == "__main__":
    url = "ws://127.0.0.1:8765/v1/realtime"
    asyncio.run(main(url))
