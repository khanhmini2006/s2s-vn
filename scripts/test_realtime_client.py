"""Test client: kết nối WS, nhận session.created, gửi audio, chờ response.done.

Mô phỏng openai-python SDK realtime flow. Dùng mock LLM server riêng.
"""
import sys, asyncio, json, base64
sys.path.insert(0, "/home/tdkhanh/s2s-vn/src")

import numpy as np
import scipy.signal
import websockets


def gen_viet_audio_pcm16() -> bytes:
    """Đọc audio mẫu 24k → resample xuống 16k (protocol = pipeline = 16k)."""
    with open("/home/tdkhanh/s2s-vn/scripts/test_input_24k.pcm", "rb") as f:
        pcm24 = f.read()
    x = np.frombuffer(pcm24, dtype=np.int16).astype(np.float32) / 32768.0
    y = scipy.signal.resample_poly(x, 16000, 24000)
    return (np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes()


async def main(ws_url: str):
    pcm16 = gen_viet_audio_pcm16()
    print(f"[client] audio 16k: {len(pcm16)} bytes, {len(pcm16)/32000:.2f}s")

    async with websockets.connect(ws_url) as ws:
        # chờ session.created
        ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        print(f"[client] recv: {ev['type']}")
        assert ev["type"] == "session.created", ev
        assert ev["session"]["audio"]["input"]["format"]["rate"] == 16000

        ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        print(f"[client] recv: {ev['type']}")
        assert ev["type"] == "conversation.created"

        # gửi session.update đổi voice
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {"audio": {"output": {"voice": "echo"}}},
        }))
        ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        print(f"[client] recv: {ev['type']} voice={ev['session']['audio']['output']['voice']}")
        assert ev["type"] == "session.updated"

        # gửi audio thành chunk nhỏ
        print("[client] gửi audio...")
        chunk = 4096  # ~128ms audio 16k
        for i in range(0, len(pcm16), chunk):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm16[i:i+chunk]).decode(),
            }))
            await asyncio.sleep(0.005)
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # thu events cho tới response.done (in từng cái)
        events = []
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=90)
            ev = json.loads(raw)
            t = ev["type"]
            events.append(t)
            extra = ""
            if t == "response.output_audio.delta":
                extra = f" audio={len(ev['delta'])}b"
            if t == "response.output_audio_transcript.delta":
                extra = f" text={ev.get('delta','')!r}"
            print(f"[client]   {t}{extra}")
            if t == "response.done":
                print(f"[client] response status: {ev['response']['status']}")
                print(f"[client] usage: {ev['response']['usage']}")
                break
            if t == "error":
                print(f"[client] ERROR: {ev}")
                break
        print(f"[client] events ({len(events)}): {events}")
        print("[client] DONE")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8765/v1/realtime"
    asyncio.run(main(url))
