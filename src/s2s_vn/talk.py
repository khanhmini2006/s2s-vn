"""Client mic/loa nói chuyện với Realtime server (tương tự `speech-to-speech talk`).

Mic 16k = WS 16k = loa 16k (pipeline = protocol, không resample). Không wake word — mở là nói luôn.

Playback dùng PlaybackBuffer (theo repo gốc): đệm audio + FILL SILENCE khi cạn
→ không bao giờ ALSA underrun (loa không bị đói dữ liệu).
Cần: pip install sounddevice websocket-client scipy numpy
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
import time

import numpy as np
import sounddevice as sd
import websocket
from scipy.signal import resample_poly

OAK_RATE = 16000
PROTOCOL_RATE = 16000
CHUNK_SAMPLES = 1280  # 80ms @16k


class PlaybackBuffer:
    """Thread-safe buffer audio PCM16 (bytes) — callback phát lấy; cạn → ghi silence."""

    def __init__(self, rate: int):
        self.rate = rate
        self._audio = bytearray()
        self._lock = threading.Lock()
        self._active_until = 0.0

    def clear(self) -> None:
        with self._lock:
            self._audio.clear()
            self._active_until = 0.0

    def append(self, audio: bytes) -> None:
        with self._lock:
            self._audio.extend(audio)
            # đánh dấu hoạt động ≥150ms (giữ buffer không bị clear sớm giữa các chunk)
            self._active_until = time.monotonic() + max(0.15, len(audio) / (2 * self.rate))

    def is_active(self) -> bool:
        with self._lock:
            return bool(self._audio) or time.monotonic() < self._active_until

    def write(self, outdata) -> None:
        """Callback của sounddevice gọi — lấy audio ra; thiếu → fill silence."""
        needed = len(outdata)
        with self._lock:
            available = min(needed, len(self._audio))
            if available:
                outdata[:available] = self._audio[:available]
                del self._audio[:available]
            if available < needed:
                outdata[available:] = b"\x00" * (needed - available)


class TalkClient:
    def __init__(self, url: str):
        self.url = url
        self.ws = None
        self.playback = PlaybackBuffer(PROTOCOL_RATE)
        self._out_stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        """Mic 16k → WS 16k (pipeline = protocol = 16k, không cần resample)."""
        if status:
            print(status)
        if self.ws and self.ws.sock and self.ws.sock.connected:
            audio_int16 = (indata[:, 0] * 32767).astype(np.int16)
            if PROTOCOL_RATE != OAK_RATE:
                resampled = resample_poly(
                    audio_int16.astype(np.float32), PROTOCOL_RATE, OAK_RATE)
                pcm = resampled.astype(np.int16).tobytes()
            else:
                pcm = audio_int16.tobytes()
            b64 = base64.b64encode(pcm).decode("utf-8")
            try:
                self.ws.send(json.dumps(
                    {"type": "input_audio_buffer.append", "audio": b64}))
            except Exception as e:
                print(f"[Error] WS Send failed: {e}")

    def _on_message(self, ws, message):
        data = json.loads(message)
        t = data.get("type")
        if t == "input_audio_buffer.speech_started":
            print("\n🎤 Đang nghe...", flush=True)
            self.playback.clear()  # bỏ audio trả lời cũ khi user bắt đầu nói
        elif t == "input_audio_buffer.speech_stopped":
            print("⏳ Đang xử lý...", flush=True)
        elif t == "response.output_audio_transcript.delta":
            print(data.get("delta", ""), end="", flush=True)
        elif t == "response.output_audio.delta":
            # PCM16 16k int16 → buffer phát (không cần convert float — phát raw int16)
            self.playback.append(base64.b64decode(data.get("delta", "")))
        elif t == "response.done":
            print("\n✅", flush=True)
        elif t == "error":
            print(f"\n[Server] Lỗi: {data.get('error')}", flush=True)

    def _output_callback(self, outdata, frames, time_info, status):
        """Loa 16k int16 — lấy từ buffer; thiếu → fill silence (chống underrun)."""
        self.playback.write(outdata)

    def _on_open(self, ws):
        print("[talk] Đã kết nối. Nói đi (Ctrl+C để thoát).", flush=True)
        self._start_playback()

    def _on_close(self, ws, close_status_code, close_msg):
        print("\n[talk] WS đóng.", flush=True)
        self._stop_playback()

    def connect(self):
        self.ws = websocket.WebSocketApp(
            self.url,
            on_message=self._on_message,
            on_open=self._on_open,
            on_close=self._on_close,
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def _start_playback(self):
        # RawOutputStream + callback: buffer cạn → fill silence, không underrun
        self._out_stream = sd.RawOutputStream(
            samplerate=PROTOCOL_RATE, channels=1, dtype="int16",
            callback=self._output_callback)
        self._out_stream.start()

    def _stop_playback(self):
        if self._out_stream:
            try:
                self._out_stream.stop()
                self._out_stream.close()
            except Exception as e:
                print(f"[talk] đóng stream lỗi: {e}")
            self._out_stream = None
        self.playback.clear()

    def start(self):
        self.connect()
        with sd.InputStream(samplerate=OAK_RATE, channels=1, dtype="float32",
                            blocksize=CHUNK_SAMPLES, callback=self._audio_callback):
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nThoát.")
                if self.ws:
                    self.ws.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="s2s-vn talk — client mic/loa nói chuyện với Realtime server")
    parser.add_argument("--url", default="ws://127.0.0.1:8765/v1/realtime",
                        help="WS URL Realtime server (mặc định ws://127.0.0.1:8765/v1/realtime)")
    args = parser.parse_args(argv)
    TalkClient(args.url).start()


if __name__ == "__main__":
    main()
