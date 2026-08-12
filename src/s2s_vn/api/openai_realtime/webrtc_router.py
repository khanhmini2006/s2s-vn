"""WebRTC router: endpoint POST /v1/realtime/calls (signaling) + aiortc PC.

Mỗi call = 1 RTCPeerConnection + 1 RealtimeService (1 pipeline).
Audio: browser Opus 48k → aiortc decode → PCM16 16k mono → pipeline.
Reply: pipeline PCM16 16k → resample 48k stereo → AudioFrame → Opus → browser.
Events: RTCDataChannel "events" (JSON) thay WS.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from fractions import Fraction

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import AudioFrame, AudioResampler
from fastapi import Body, Request

from ...pipeline.messages import AudioOutput, ResponseDone
from ...s2s_pipeline import PipelineConfig
from .realtime_service import PIPELINE_RATE, RealtimeService

logger = logging.getLogger("s2s.webrtc")

PIPELINE_CHUNK = 1024  # bytes PCM16 16k = 32ms
RTP_RATE = 48000  # Opus (WebRTC)
RTP_SAMPLES = 960  # 20ms @48k


class PipelineOutputTrack(MediaStreamTrack):
    """MediaStreamTrack gửi PCM16 16k (từ pipeline audio_out) về browser.

    recv() trả av.AudioFrame 48k stereo s16 960 samples/frame (20ms).
    Pacing theo pts realtime.
    """

    kind = "audio"

    def __init__(self, loop):
        super().__init__()
        self.loop = loop
        self.pts = 0
        self.time_base = Fraction(1, RTP_RATE)
        # pipeline 16k mono → WebRTC 48k stereo
        self.up = AudioResampler("s16", "stereo", RTP_RATE)
        self._pcm_buf = bytearray()
        self._async_q = asyncio.Queue()
        self._t0 = None

    def feed_pcm16(self, pcm: bytes) -> None:
        try:
            self.loop.call_soon_threadsafe(self._async_q.put_nowait, pcm)
        except Exception:
            pass

    async def recv(self):
        import numpy as np

        # gom đủ 640 bytes (320 mẫu s16 mono @16k = 20ms)
        while len(self._pcm_buf) < 640:
            chunk = await self._async_q.get()
            self._pcm_buf.extend(chunk)
        mono = bytes(self._pcm_buf[:640])
        del self._pcm_buf[:640]

        arr = np.frombuffer(mono, dtype=np.int16).reshape(1, -1)
        in_frame = AudioFrame.from_ndarray(arr, format="s16", layout="mono")
        in_frame.sample_rate = PIPELINE_RATE
        in_frame.time_base = Fraction(1, PIPELINE_RATE)
        in_frame.pts = self.pts * PIPELINE_RATE // RTP_RATE

        out_frames = self.up.resample(in_frame)
        if out_frames:
            out = out_frames[0]
        else:
            out = AudioFrame(format="s16", layout="stereo", samples=RTP_SAMPLES)
            for p in out.planes:
                p.update(bytes(p.buffer_size))

        out.pts = self.pts
        out.time_base = self.time_base
        out.sample_rate = RTP_RATE
        self.pts += out.samples

        # pacing realtime (tránh gửi nhanh hơn thời gian thực)
        if self._t0 is None:
            self._t0 = time.monotonic()
        target = self.pts / RTP_RATE
        delay = target - (time.monotonic() - self._t0)
        if delay > 0:
            await asyncio.sleep(delay)
        return out


class WebRTCSession:
    """Một WebRTC call = 1 PC + 1 RealtimeService."""

    def __init__(self, pc: RTCPeerConnection, cfg: PipelineConfig):
        self.pc = pc
        self.cfg = cfg
        self.service: RealtimeService | None = None
        self.out_track: PipelineOutputTrack | None = None
        self.channel = None
        self.loop = asyncio.get_event_loop()

    def setup(self) -> None:
        self.service = RealtimeService(self.cfg, on_event=self._send_event)
        self.service.start_drain()

    def _send_event(self, ev: dict) -> None:
        if self.channel and self.channel.readyState == "open":
            try:
                self.loop.call_soon_threadsafe(
                    self._safe_send, json.dumps(ev, ensure_ascii=False))
            except Exception as e:
                logger.warning("dc send schedule fail: %r", e)

    def _safe_send(self, msg: str) -> None:
        try:
            self.channel.send(msg)
        except Exception as e:
            logger.warning("dc send fail: %r", e)

    async def _consume_inbound_audio(self, track):
        """Recv audio track từ browser → PCM16 16k → pipeline."""

        resampler = AudioResampler(format="s16", layout="mono", rate=PIPELINE_RATE)
        buf = bytearray()
        while True:
            try:
                frame = await track.recv()
            except Exception:
                return
            for f in resampler.resample(frame):
                pcm = f.to_ndarray().tobytes()
                buf.extend(pcm)
                while len(buf) >= PIPELINE_CHUNK:
                    self.service.pipeline.send_audio(bytes(buf[:PIPELINE_CHUNK]))
                    del buf[:PIPELINE_CHUNK]

    def attach_drain_audio(self) -> None:
        """Override service drain để audio_out → out_track (không encode WS)."""

        service = self.service
        orig = service._handle_audio

        def patched_handle_audio(item):
            if isinstance(item, AudioOutput):
                if self.out_track:
                    self.out_track.feed_pcm16(item.audio)
            elif isinstance(item, ResponseDone):
                orig(item)

        service._handle_audio = patched_handle_audio


def setup_webrtc(app, cfg: PipelineConfig, check_auth=None) -> None:
    """Gắn WebRTC signaling endpoint. check_auth giữ cho khớp signature (không áp dụng)."""

    sessions: set[RTCPeerConnection] = set()

    @app.post("/v1/realtime/calls")
    async def realtime_calls(request: Request, body: dict = Body(...)):
        pc = RTCPeerConnection()
        sess = WebRTCSession(pc, cfg)
        sessions.add(pc)

        @pc.on("connectionstatechange")
        async def on_state():
            if pc.connectionState in ("closed", "failed"):
                await cleanup(pc, sess)

        @pc.on("datachannel")
        def on_datachannel(channel):
            sess.channel = channel

            @channel.on("message")
            def on_message(message):
                try:
                    sess.service.handle_event(json.loads(message))
                except Exception as e:
                    logger.warning("data channel msg error: %r", e)

        @pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                sess.out_track = PipelineOutputTrack(asyncio.get_event_loop())
                pc.addTrack(sess.out_track)
                asyncio.ensure_future(sess._consume_inbound_audio(track))

        sess.setup()
        sess.attach_drain_audio()

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=body.get("sdp"), type=body.get("type")))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    async def cleanup(pc: RTCPeerConnection, sess: WebRTCSession) -> None:
        sess.service.stop()
        await pc.close()
        sessions.discard(pc)

    @app.on_event("shutdown")
    async def on_shutdown():
        await asyncio.gather(*[pc.close() for pc in sessions], return_exceptions=True)
