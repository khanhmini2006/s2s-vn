"""WebSocket router: endpoint /v1/realtime, session lifecycle, send loop.

Mỗi connection = 1 RealtimeService (1 pipeline). on_event từ drain thread
đưa server events qua asyncio.Queue → send task gửi WS.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ...s2s_pipeline import PipelineConfig
from .realtime_service import RealtimeService

logger = logging.getLogger("s2s.realtime")
router = APIRouter()


class Session:
    def __init__(self, ws: WebSocket, cfg: PipelineConfig):
        self.ws = ws
        self.events: asyncio.Queue[dict] = asyncio.Queue(maxsize=1000)
        self.service = RealtimeService(cfg, on_event=self.on_event)
        self.send_task: asyncio.Task | None = None

    def on_event(self, ev: dict) -> None:
        """Call từ drain thread → đẩy vào asyncio queue (thread-safe)."""
        loop = self.send_task.get_loop() if self.send_task else asyncio.get_event_loop()
        loop.call_soon_threadsafe(self.events.put_nowait, ev)


async def _send_loop(sess: Session) -> None:
    while True:
        ev = await sess.events.get()
        try:
            await sess.ws.send_text(json.dumps(ev, ensure_ascii=False))
        except Exception:
            logger.exception("send failed")
            return


@router.websocket("/v1/realtime")
async def realtime_endpoint(ws: WebSocket, cfg: PipelineConfig) -> None:
    await ws.accept()
    sess = Session(ws, cfg)
    sess.send_task = asyncio.create_task(_send_loop(sess))
    try:
        sess.service._emit("session.created", session=sess.service._session_object())
        sess.service._emit("conversation.created",
                           conversation={"id": sess.service.conversation_id,
                                         "object": "realtime.conversation"})
        sess.service.start_drain(sess.on_event)

        while True:
            # idle timeout: đóng session nếu không nhận event trong 5 phút
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=300)
            except asyncio.TimeoutError:
                logger.info("session idle timeout, closing")
                break
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                sess.service._emit_error("malformed JSON")
                continue
            if not isinstance(ev, dict):
                sess.service._emit_error("event must be JSON object")
                continue
            sess.service.handle_event(ev)
    except WebSocketDisconnect:
        pass
    finally:
        sess.send_task.cancel()
        sess.service.stop()


def setup_realtime(app, cfg: PipelineConfig, check_auth=None) -> None:
    """Gắn WS endpoint /v1/realtime với config."""

    async def handler(ws: WebSocket):
        # auth: query param ?api_key= hoặc Authorization header
        if check_auth:
            try:
                check_auth(ws.headers.get("authorization"),
                           ws.query_params.get("api_key"))
            except Exception:
                await ws.close(code=4401)
                return
        await realtime_endpoint(ws, cfg)

    app.websocket("/v1/realtime")(handler)
