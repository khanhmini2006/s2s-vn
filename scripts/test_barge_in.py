"""Test barge-in: gửi audio 1, chờ response bắt đầu, gửi audio 2 (ngắt lời).
Verify: response cũ bị cancel, không có audio cũ tiếp, response.done cancelled.
"""
import asyncio, json, base64, sys, time
import websockets

pcm24 = open("/home/tdkhanh/test_input_24k.pcm", "rb").read()


async def main(url):
    async with websockets.connect(url) as ws:
        # consume session.created, conversation.created
        for _ in range(2):
            ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        print("[t] connected")

        # gửi audio câu 1 (đợt 1)
        async def send_audio(buf):
            for i in range(0, len(buf), 4096):
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(buf[i:i+4096]).decode(),
                }))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        await send_audio(pcm24)
        print("[t] audio 1 gửi, chờ response bắt đầu...")

        # thu events cho tới response.created
        seen_delta = False
        for _ in range(60):
            ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
            t = ev["type"]
            if t == "response.output_audio.delta":
                seen_delta = True
                print(f"[t] response đang chạy, audio delta tới")
                break
            if t == "response.done":
                print(f"[t] response xong sớm: {ev['response']['status']}")
                return
        if not seen_delta:
            print("[t] không thấy audio delta")
            return

        # barge-in: gửi audio 2 ngay khi response đang chạy
        print("[t] BARGING IN — gửi audio 2...")
        await send_audio(pcm24)

        # thu events cho tới response.done — verify cancelled
        got_cancelled = False
        audio_after_barge = 0
        for _ in range(80):
            ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
            t = ev["type"]
            if t == "response.output_audio.delta":
                audio_after_barge += len(ev["delta"])
            if t == "response.done":
                st = ev["response"]["status"]
                reason = (ev["response"]["status_details"] or {}).get("reason")
                print(f"[t] response.done: status={st} reason={reason}")
                print(f"[t] audio còn gửi sau barge-in: {audio_after_barge}b")
                got_cancelled = (st == "cancelled")
                break
        print(f"[t] KẾT QUẢ: cancelled={got_cancelled}")
        return got_cancelled


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8765/v1/realtime"
    result = asyncio.run(main(url))
    sys.exit(0 if result else 1)
