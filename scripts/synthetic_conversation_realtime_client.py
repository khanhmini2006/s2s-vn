"""Client hội thoại giả lập: 1 hoặc N client, 1 hoặc nhiều lượt (turn), tải/soak test.

Mở 1 (hoặc N) kết nối WebSocket tới /v1/realtime và lặp qua ``--turns`` câu hỏi
mẫu mỗi client, cách nhau ``--interval`` giây. Mỗi turn:

  1. Tổng hợp câu hỏi bằng TTS backend đã đăng ký trong TTS_BACKENDS (mặc định
     vieneu; cache ra đĩa sau lần chạy đầu, theo tên backend).
  2. Stream audio câu hỏi + khoảng lặng cuối như input_audio_buffer.append.
  3. Chờ response.done.
  4. Ngủ tới khi đủ --interval kể từ lúc turn bắt đầu.

Port từ speech-to-speech/scripts/synthetic_conversation_realtime_client.py.
Khác biệt so với bản gốc:

  * Sinh audio mẫu qua get_tts_handler() (backend_registry, nguồn TTS_BACKENDS)
    thay vì macOS `say` (không có trên Linux) — chọn backend qua --tts-name
    (vieneu | mms-vie | piper-vie), tự resample về PROTOCOL_RATE_HZ.
  * Auth theo s2s-vn: `Authorization: Bearer $S2S_API_KEY` nếu server bật auth
    (env `S2S_API_KEY`); không bật thì không gửi header (bản gốc bắt buộc
    `HF_TOKEN`, s2s-vn coi auth là optional).
  * Bỏ toàn bộ nhánh `--lb-url` (load balancer nhiều compute-node) — s2s-vn
    chạy single-node, không có tầng LB.

Usage:
  * Test pool đồng thời (nhiều client cùng lúc, mỗi client 1 turn):
      python scripts/synthetic_conversation_realtime_client.py --clients 3 --turns 1

  * Soak 1 client dài hơi (60 turns, cách nhau 10s ~ 10 phút):
      python scripts/synthetic_conversation_realtime_client.py --turns 60 --interval 10

  * Soak cả pool (2 client x 60 turns):
      python scripts/synthetic_conversation_realtime_client.py --clients 2 --turns 60

  * Sinh câu hỏi mẫu bằng backend TTS khác (mms-vie / piper-vie):
      python scripts/synthetic_conversation_realtime_client.py --tts-name piper-vie

Mỗi client dùng offset câu hỏi riêng (coprime shift) để các client chạy song
song hỏi câu khác nhau mỗi turn — dễ phát hiện rò rỉ cross-session qua log.

Output nằm dưới --log-dir:
  * prompts/<tts_name>/prompt_NNN.wav  — cache audio câu hỏi theo backend (sinh 1 lần, dùng lại)
  * client_NNN/conversation.txt        — transcript từng client kèm timestamp
  * client_NNN/conversation.wav        — audio phản hồi của assistant nối lại
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import time
import wave
from pathlib import Path
from queue import Queue

import numpy as np
import soundfile as sf
import websockets
from scipy.signal import resample_poly

from s2s_vn.backend_registry import TTS_BACKENDS, get_tts_handler
from s2s_vn.pipeline.messages import TTSInput
from s2s_vn.s2s_pipeline import PipelineConfig

logger = logging.getLogger("synthetic_client")

PROTOCOL_RATE_HZ = 16000  # protocol = pipeline = 16kHz
CHUNK_MS = 20
BYTES_PER_SAMPLE = 2  # PCM16
CHUNK_BYTES = PROTOCOL_RATE_HZ * BYTES_PER_SAMPLE * CHUNK_MS // 1000
# Khoảng lặng cuối mỗi prompt để server VAD phát hiện speech_stopped và
# tự commit (server chỉ hỗ trợ server_vad turn_detection).
TRAILING_SILENCE_MS = 1500
# Offset câu hỏi mỗi client. 7 nguyên tố cùng nhau với len(PROMPTS)=40 nên mỗi
# client đi qua một hoán vị câu hỏi khác nhau.
PROMPT_SHIFT_PER_CLIENT = 7

# 40 câu hỏi tiếng Việt đa dạng — đủ ngắn cho 1 turn 10s, đủ khác nhau để
# response không lặp khuôn mẫu. Lặp vòng nếu --turns > len(PROMPTS).
PROMPTS: list[str] = [
    "Thủ đô của Việt Nam là gì?",
    "Kể cho tôi một câu chuyện cười về robot.",
    "Quang hợp hoạt động như thế nào?",
    "Hai cộng hai bằng mấy?",
    "Ai đã vẽ bức tranh Mona Lisa?",
    "Đại dương lớn nhất trên Trái Đất là gì?",
    "Gợi ý cho tôi một món ăn đơn giản.",
    "Tốc độ ánh sáng là bao nhiêu?",
    "Ai đã viết Truyện Kiều?",
    "Nhiệt độ sôi của nước là bao nhiêu độ C?",
    "Cho tôi một sự thật về sao Hỏa.",
    "Sự khác biệt giữa thời tiết và khí hậu là gì?",
    "Có bao nhiêu châu lục trên thế giới?",
    "Thơ haiku là gì?",
    "Công thức hóa học của nước là gì?",
    "Ai là vị vua đầu tiên của triều Nguyễn?",
    "Trọng lực là gì?",
    "Cho tôi một sự thật thú vị về cá heo.",
    "Ngọn núi cao nhất thế giới là núi nào?",
    "Nam châm hoạt động như thế nào?",
    "Ý nghĩa cuộc sống là gì, trong một câu thôi?",
    "Gợi ý cho tôi một cuốn sách ngắn nên đọc.",
    "Dân số Hà Nội khoảng bao nhiêu?",
    "Tủ lạnh giữ lạnh bằng cách nào?",
    "Ai đã phát minh ra điện thoại?",
    "Quốc gia nhỏ nhất thế giới là nước nào?",
    "Giải thích thuyết tương đối một cách đơn giản.",
    "Sự khác nhau giữa virus và vi khuẩn là gì?",
    "Sông dài nhất Việt Nam là sông nào?",
    "Trái tim con người hoạt động ra sao?",
    "Ngôn ngữ nào được nói nhiều nhất thế giới?",
    "Cho tôi một sự thật về hố đen.",
    "Làm sao để gấp một chiếc máy bay giấy?",
    "Ký hiệu hóa học của vàng là gì?",
    "Có bao nhiêu xương trong cơ thể người?",
    "Hành tinh nhỏ nhất trong hệ mặt trời là gì?",
    "Vì sao bầu trời có màu xanh?",
    "Sa mạc lớn nhất thế giới nằm ở đâu?",
    "Có bao nhiêu hành tinh trong hệ mặt trời?",
    "Tạm biệt nhé.",
]


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

_tts_handler_cache: dict[str, object] = {}


def _get_tts_handler(tts_name: str, voice: str):
    """Dựng + warmup 1 TTS handler theo tên (registry), cache lại theo tts_name
    (model load chậm, chỉ warmup 1 lần cho toàn bộ N prompt). `voice` chỉ có
    tác dụng khi tts_name=vieneu (mms-vie/piper-vie bỏ qua, 1 giọng cố định)."""
    if tts_name not in _tts_handler_cache:
        logger.info(f"Đang tải TTS backend '{tts_name}' để sinh câu hỏi mẫu...")
        cfg = PipelineConfig(tts_name=tts_name, tts_voice=voice, sample_rate=PROTOCOL_RATE_HZ)
        handler = get_tts_handler(Queue(), Queue(), cfg)
        handler.warmup()
        _tts_handler_cache[tts_name] = handler
    return _tts_handler_cache[tts_name]


def synthesize_with_tts(text: str, out_path: Path, tts_name: str, voice: str) -> None:
    """Sinh audio *text* bằng backend TTS đã đăng ký (TTS_BACKENDS), ghi WAV
    ở PROTOCOL_RATE_HZ (handler tự resample về rate này, xem *_tts_handler.py)."""
    handler = _get_tts_handler(tts_name, voice)
    handler.process(TTSInput(text=text, turn_id=0))
    pcm16 = bytearray()
    while not handler.output_queue.empty():
        out = handler.output_queue.get_nowait()
        audio_bytes = getattr(out, "audio", None)
        if audio_bytes:
            pcm16.extend(audio_bytes)
    write_wav(out_path, bytes(pcm16), rate=PROTOCOL_RATE_HZ)


def load_pcm16_mono(path: Path, target_rate: int = PROTOCOL_RATE_HZ) -> bytes:
    """Đọc WAV bất kỳ, trả PCM16 little-endian mono ở target_rate."""
    data, src_rate = sf.read(str(path), always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if src_rate != target_rate:
        mono = resample_poly(mono, target_rate, src_rate)
    mono = np.clip(mono, -1.0, 1.0)
    pcm16 = (mono * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def write_wav(path: Path, pcm16_bytes: bytes, rate: int = PROTOCOL_RATE_HZ) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(BYTES_PER_SAMPLE)
        w.setframerate(rate)
        w.writeframes(pcm16_bytes)


# ---------------------------------------------------------------------------
# Luồng WS mỗi turn
# ---------------------------------------------------------------------------


async def stream_prompt(ws, audio_pcm: bytes) -> None:
    """Gửi audio prompt theo nhịp thời gian thực, sau đó khoảng lặng."""
    for i in range(0, len(audio_pcm), CHUNK_BYTES):
        chunk = audio_pcm[i : i + CHUNK_BYTES]
        if len(chunk) < CHUNK_BYTES:
            chunk = chunk + b"\x00" * (CHUNK_BYTES - len(chunk))
        await ws.send(
            json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode("ascii")})
        )
        await asyncio.sleep(CHUNK_MS / 1000.0)
    silence_chunk = b"\x00" * CHUNK_BYTES
    silence_payload = base64.b64encode(silence_chunk).decode("ascii")
    for _ in range(TRAILING_SILENCE_MS // CHUNK_MS):
        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": silence_payload}))
        await asyncio.sleep(CHUNK_MS / 1000.0)


async def consume_until_response_done(
    ws,
    response_audio_out: bytearray,
    response_timeout_s: float,
) -> dict:
    """Đọc event server tới response.done (hoặc timeout). Trả tóm tắt turn."""
    info: dict = {
        "transcript_in": "",
        "transcript_out": "",
        "error": None,
    }
    deadline = time.monotonic() + response_timeout_s
    while time.monotonic() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - time.monotonic()))
        except asyncio.TimeoutError:
            info["error"] = "response_timeout"
            return info
        event = json.loads(raw)
        t = event.get("type", "")

        if t == "conversation.item.input_audio_transcription.completed":
            info["transcript_in"] += event.get("transcript", "")
        elif t == "response.output_audio.delta":
            delta = event.get("delta", "")
            if delta:
                response_audio_out.extend(base64.b64decode(delta))
        elif t == "response.output_audio_transcript.delta":
            info["transcript_out"] += event.get("delta", "")
        elif t == "response.output_audio_transcript.done":
            transcript = event.get("transcript")
            if transcript:
                info["transcript_out"] = transcript
        elif t == "response.done":
            return info
        elif t == "error":
            info["error"] = event.get("error", {}).get("message", "error event")
            return info
    info["error"] = "response_timeout"
    return info


def _truncate_transcript(s: str, max_len: int = 30) -> str:
    """Quote *s*; nếu dài hơn max_len thì cắt + ellipsis + tổng số ký tự."""
    if len(s) <= max_len:
        return repr(s)
    return f"{(s[:max_len] + '...')!r} ({len(s)} chars)"


# ---------------------------------------------------------------------------
# Driver mỗi client
# ---------------------------------------------------------------------------


async def run_client(
    client_id: int,
    args: argparse.Namespace,
    ws_url: str,
    extra_headers: list[tuple[str, str]],
    audio_files: list[tuple[str, Path]],
) -> dict:
    """Chạy hội thoại nhiều turn của 1 client. Trả tóm tắt kết quả."""
    summary: dict = {
        "client_id": client_id,
        "connected": False,
        "rejected": False,
        "completed": 0,
        "errors": 0,
        "error_msg": None,
    }

    log_dir = Path(args.log_dir) / f"client_{client_id:03d}"
    log_dir.mkdir(parents=True, exist_ok=True)
    transcript_log = log_dir / "conversation.txt"
    response_audio = bytearray()

    prefix = f"[c{client_id}]"

    try:
        async with websockets.connect(
            ws_url,
            max_size=2**24,
            additional_headers=extra_headers or None,
        ) as ws:
            summary["connected"] = True

            first = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
            if first.get("type") == "error" and "session_limit_reached" in str(first):
                summary["rejected"] = True
                summary["error_msg"] = first.get("error", {}).get("message", "limit reached")
                logger.warning(f"{prefix} REJECTED: {summary['error_msg']}")
            elif first.get("type") != "session.created":
                summary["error_msg"] = f"unexpected first event: {first.get('type')}"
                logger.error(f"{prefix} ERROR: {summary['error_msg']}")
            else:
                sess = first.get("session") or {}
                session_id = sess.get("id") or first.get("event_id") or "?"
                logger.info(f"{prefix} connected, session={session_id}")

                # conversation.created gửi ngay sau session.created (xem
                # websocket_router.py) — vét nốt trước khi bắt đầu turn.
                try:
                    nxt = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                    if nxt.get("type") != "conversation.created":
                        logger.warning(f"{prefix} unexpected 2nd event: {nxt.get('type')}")
                except asyncio.TimeoutError:
                    pass

                with transcript_log.open("w", encoding="utf-8") as log_f:
                    log_f.write(f"# Hội thoại giả lập, client={client_id}, session={session_id}\n")
                    log_f.write(f"# Target turns={args.turns}, interval={args.interval}s\n\n")

                    for turn_idx in range(args.turns):
                        # offset câu hỏi mỗi client (coprime shift) để các
                        # client song song hỏi câu khác nhau mỗi turn.
                        prompt_idx = (turn_idx + client_id * PROMPT_SHIFT_PER_CLIENT) % len(audio_files)
                        text, wav_path = audio_files[prompt_idx]

                        turn_start = time.monotonic()
                        turn_audio = load_pcm16_mono(wav_path)

                        logger.info(f"{prefix} turn {turn_idx + 1}/{args.turns} USER: {text!r}")
                        log_f.write(f"[turn {turn_idx + 1}/{args.turns}] USER: {text}\n")

                        try:
                            await stream_prompt(ws, turn_audio)
                        except websockets.exceptions.ConnectionClosed as e:
                            logger.info(
                                f"{prefix} turn {turn_idx + 1}/{args.turns} "
                                f"connection closed during send: {e}"
                            )
                            log_f.write(f"[turn {turn_idx + 1}/{args.turns}] CONNECTION_CLOSED: {e}\n")
                            summary["error_msg"] = f"connection_closed: {e}"
                            break

                        info = await consume_until_response_done(
                            ws, response_audio, args.response_timeout
                        )
                        turn_elapsed = time.monotonic() - turn_start

                        if info["error"]:
                            logger.info(
                                f"{prefix} turn {turn_idx + 1}/{args.turns} ERROR: {info['error']}"
                            )
                            log_f.write(f"[turn {turn_idx + 1}/{args.turns}] ERROR: {info['error']}\n\n")
                            summary["errors"] += 1
                        else:
                            summary["completed"] += 1
                            logger.info(
                                f"{prefix} turn {turn_idx + 1}/{args.turns} "
                                f"ASSISTANT: {_truncate_transcript(info['transcript_out'])}"
                            )
                            log_f.write(f"[turn {turn_idx + 1}/{args.turns}] STT: {info['transcript_in']}\n")
                            log_f.write(
                                f"[turn {turn_idx + 1}/{args.turns}] ASSISTANT: {info['transcript_out']}\n\n"
                            )
                        log_f.flush()

                        remaining = args.interval - turn_elapsed
                        if remaining > 0:
                            await asyncio.sleep(remaining)

    except Exception as e:  # noqa: BLE001 — đường chẩn đoán, ghi log là đủ
        summary["error_msg"] = f"{type(e).__name__}: {e}"
        logger.error(f"{prefix} EXCEPTION: {summary['error_msg']}")

    if response_audio:
        wav_out = log_dir / "conversation.wav"
        write_wav(wav_out, bytes(response_audio), rate=16000)  # AudioOutput = 16k pipeline rate
        logger.info(f"{prefix} wrote {len(response_audio)} bytes -> {wav_out}")
    return summary


# ---------------------------------------------------------------------------
# Top-level: sinh prompt, spawn client, tổng kết
# ---------------------------------------------------------------------------


async def run_all(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    # cache theo tts_name — mỗi backend sinh giọng khác nhau, không dùng chung
    prompts_dir = log_dir / "prompts" / args.tts_name
    prompts_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Sinh {len(PROMPTS)} câu hỏi bằng backend '{args.tts_name}' (cache tại {prompts_dir})...")
    audio_files: list[tuple[str, Path]] = []
    for i, text in enumerate(PROMPTS):
        wav_path = prompts_dir / f"prompt_{i:03d}.wav"
        if not wav_path.exists():
            synthesize_with_tts(text, wav_path, tts_name=args.tts_name, voice=args.voice)
        audio_files.append((text, wav_path))

    ws_url = args.url or f"ws://{args.host}:{args.port}/v1/realtime"
    extra_headers: list[tuple[str, str]] = []
    api_key = os.environ.get("S2S_API_KEY")
    if api_key:
        extra_headers.append(("Authorization", f"Bearer {api_key}"))
        logger.info("Auth: Bearer token đính kèm từ S2S_API_KEY env")
    else:
        logger.info("Auth: không đặt S2S_API_KEY — kết nối không auth (server mặc định tắt auth)")

    logger.info(
        f"Spawning {args.clients} client(s) tới {ws_url}, "
        f"{args.turns} turns mỗi client @ {args.interval:.1f}s interval"
    )

    summaries = await asyncio.gather(
        *(
            run_client(
                client_id=i,
                args=args,
                ws_url=ws_url,
                extra_headers=extra_headers,
                audio_files=audio_files,
            )
            for i in range(args.clients)
        )
    )

    logger.info("\n=== summary ===")
    for s in summaries:
        status = "rejected" if s["rejected"] else ("error" if s["error_msg"] else "ok")
        logger.info(
            f"  c{s['client_id']}: {status:8s} completed={s['completed']}/{args.turns} "
            f"errors={s['errors']} err={s['error_msg']}"
        )
    n_ok = sum(1 for s in summaries if s["connected"] and not s["rejected"] and not s["error_msg"])
    n_rej = sum(1 for s in summaries if s["rejected"])
    n_err = sum(1 for s in summaries if s["error_msg"] and not s["rejected"])
    total_turns = sum(s["completed"] for s in summaries)
    logger.info(f"=> {n_ok} client thành công, {n_rej} bị từ chối, {n_err} lỗi")
    logger.info(f"=> {total_turns} turn hoàn thành trên toàn pool")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Full ws:// hoặc wss:// URL tới /v1/realtime. Override --host/--port.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Chỉ dùng khi --url không set.")
    parser.add_argument("--port", type=int, default=8765, help="Chỉ dùng khi --url không set.")
    parser.add_argument(
        "--tts-name",
        default="vieneu",
        choices=list(TTS_BACKENDS),
        help=f"Backend TTS sinh câu hỏi mẫu (mặc định: vieneu — nguồn {list(TTS_BACKENDS)})",
    )
    parser.add_argument("--voice", default="Trúc Ly", help="Voice để sinh câu hỏi mẫu — chỉ áp dụng khi --tts-name=vieneu.")
    parser.add_argument("--clients", type=int, default=1, help="Số client song song (mặc định 1).")
    parser.add_argument("--turns", type=int, default=10, help="Số turn mỗi client (mặc định 10).")
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Số giây giữa các lần bắt đầu turn mỗi client. Tổng thời gian ≈ turns × interval. Mặc định 10.",
    )
    parser.add_argument(
        "--response-timeout",
        type=float,
        default=30.0,
        help="Thời gian chờ response.done mỗi turn trước khi coi là lỗi.",
    )
    parser.add_argument(
        "--log-dir",
        default="/tmp/synthetic_conversation",
        help="Thư mục gốc lưu cache prompt và log từng client.",
    )
    args = parser.parse_args()
    if args.clients < 1:
        parser.error("--clients phải >= 1")
    if args.turns < 1:
        parser.error("--turns phải >= 1")
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
