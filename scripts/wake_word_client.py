import argparse
import base64
import json
import queue
import threading
import time

import numpy as np
import sounddevice as sd
import websocket
from openwakeword.model import Model
from scipy.signal import resample_poly

# Constants
OAK_RATE = 16000
PROTOCOL_RATE = 16000
CHUNK_SAMPLES = 1280  # 80ms at 16000Hz
WS_URL = "ws://localhost:8765/v1/realtime"


class WakeWordClient:
    def __init__(self, wakeword_model="alexa"):
        self.oww_model = Model(wakeword_models=[wakeword_model])
        self.wakeword_name = list(self.oww_model.models.keys())[0]
        self.ws = None
        self.is_active = False
        self.audio_queue = queue.Queue()
        self.play_queue = queue.Queue()
        self.playing = False
        self.play_thread = None

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk (16000Hz)."""
        if status:
            print(status)
        # indata is shape (frames, 1) float32. openwakeword needs int16 16000Hz
        audio_int16 = (indata[:, 0] * 32767).astype(np.int16)

        if not self.is_active:
            # State: Waiting for wake word
            prediction = self.oww_model.predict(audio_int16)
            score = prediction[self.wakeword_name]
            if score > 0.5:
                print(f"\n[WakeWord] Đã nghe '{self.wakeword_name}'! Bắt đầu kết nối...")
                self.is_active = True
                self.connect_ws()
        else:
            # State: Active, send audio to server
            if self.ws and self.ws.sock and self.ws.sock.connected:
                # 16k = 16k — không cần resample
                resampled = resample_poly(audio_int16.astype(np.float32), PROTOCOL_RATE, OAK_RATE)
                resampled_int16 = resampled.astype(np.int16).tobytes()
                b64 = base64.b64encode(resampled_int16).decode("utf-8")
                msg = {"type": "input_audio_buffer.append", "audio": b64}
                try:
                    self.ws.send(json.dumps(msg))
                except Exception as e:
                    print(f"[Error] WS Send failed: {e}")

    def connect_ws(self):
        def on_message(ws, message):
            data = json.loads(message)
            event_type = data.get("type")
            if event_type == "input_audio_buffer.speech_started":
                print("[Server] Đang nghe...")
            elif event_type == "input_audio_buffer.speech_stopped":
                print("[Server] Đang xử lý...")
            elif event_type == "response.output_audio_transcript.delta":
                print(data.get("delta", ""), end="", flush=True)
            elif event_type == "response.output_audio.delta":
                b64 = data.get("delta", "")
                pcm16 = base64.b64decode(b64)
                # Received audio is 16000Hz PCM16.
                audio_float = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32767.0
                self.play_queue.put(audio_float)
            elif event_type == "response.done":
                print("\n[Server] Hoàn tất trả lời.")
                # We can close WS to wait for the next wake word, or keep it open.
                # Here we close it.
                self.close_ws()
            elif event_type == "error":
                print(f"\n[Server] Lỗi: {data.get('error')}")

        def on_open(ws):
            print("[Client] WS Đã kết nối.")
            self.start_playback_thread()

        def on_close(ws, close_status_code, close_msg):
            print("[Client] WS Đóng. Trở về trạng thái chờ Wake Word.")
            self.is_active = False
            self.stop_playback_thread()
            # Reset OWW state
            self.oww_model.reset()

        self.ws = websocket.WebSocketApp(WS_URL,
                                         on_message=on_message,
                                         on_open=on_open,
                                         on_close=on_close)
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()

    def close_ws(self):
        if self.ws:
            self.ws.close()
            self.ws = None

    def start_playback_thread(self):
        self.playing = True
        def player():
            # stream output at 16000Hz
            with sd.OutputStream(samplerate=PROTOCOL_RATE, channels=1, dtype='float32') as stream:
                while self.playing:
                    try:
                        chunk = self.play_queue.get(timeout=0.1)
                        stream.write(chunk)
                    except queue.Empty:
                        continue
        self.play_thread = threading.Thread(target=player)
        self.play_thread.daemon = True
        self.play_thread.start()

    def stop_playback_thread(self):
        self.playing = False
        if self.play_thread:
            self.play_thread.join()
            self.play_thread = None
        # Xóa hàng đợi audio cũ
        while not self.play_queue.empty():
            self.play_queue.get()

    def start(self):
        print(f"Bắt đầu lắng nghe wake word: '{self.wakeword_name}'...")
        with sd.InputStream(samplerate=OAK_RATE, channels=1, dtype='float32', blocksize=CHUNK_SAMPLES, callback=self._audio_callback):
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nThoát.")
                self.close_ws()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="alexa", help="Wake word model (e.g. alexa, hey_mycroft)")
    args = parser.parse_args()
    
    client = WakeWordClient(wakeword_model=args.model)
    client.start()
