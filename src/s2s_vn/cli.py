"""CLI entry point: `s2s-vn serve|talk|local` (theo cấu trúc repo gốc speech-to-speech).

- serve: chạy Realtime server (WS + WebRTC). Flags pass-through cho `server.main()`
         (nguồn sự thật flags: api/openai_realtime/server.py). Config ưu tiên:
         config.json (nếu có) > flags CLI > default.
- talk:  client mic/loa nói chuyện với server đã chạy (chỉ --url).
- local: serve + talk in-process (loopback).
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="s2s-vn",
        description="Voice-agent pipeline tiếng Việt: VAD → STT → LLM → TTS, "
                    "OpenAI Realtime-compatible WebSocket + WebRTC")
    sub = parser.add_subparsers(dest="command", required=True)

    # serve: mọi flag --* sau "serve" pass-through cho server.main() (xem server.py main).
    # add_help=False → "--help" lọt extras → server.main(argv=["--help"]) in đủ flags.
    p_serve = sub.add_parser(
        "serve",
        help="Chạy Realtime server (WS + WebRTC). Xem `s2s-vn serve --help` cho flags.",
        add_help=False,
    )
    p_serve.set_defaults(_fn=cmd_serve)

    p_talk = sub.add_parser("talk", help="Client mic/loa nói chuyện với server")
    p_talk.add_argument("--url", default="ws://127.0.0.1:8765/v1/realtime",
                        help="WS URL Realtime server")
    p_talk.set_defaults(_fn=cmd_talk)

    p_local = sub.add_parser("local", help="Serve + talk in-process (loopback)")
    p_local.add_argument("--port", type=int, default=8765)
    p_local.add_argument("--config", default=None,
                         help="File config JSON (mặc định: config.json). "
                              "Vd: --config config-local.json")
    p_local.set_defaults(_fn=cmd_local)

    args, extras = parser.parse_known_args()
    args._fn(args, extras)


def cmd_serve(args, extras: list[str]) -> None:
    from .api.openai_realtime.server import main as server_main

    # truyền thẳng các flag sau "serve" (server.main() là nguồn sự thật flags)
    server_main(argv=extras)


def cmd_talk(args, _extras: list[str]) -> None:
    from .talk import main as talk_main

    talk_main(argv=["--url", args.url])


def cmd_local(args, _extras: list[str]) -> None:
    import socket
    import threading
    import time

    import uvicorn

    from .api.openai_realtime.server import create_app, load_config
    from .s2s_pipeline import PipelineConfig
    from .talk import main as talk_main

    port = args.port
    cfg = load_config(args.config) or PipelineConfig()
    app = create_app(cfg)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port))
    threading.Thread(target=server.run, daemon=True).start()

    # đợi server mở port rồi mới connect client
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.2)
    try:
        talk_main(argv=["--url", f"ws://127.0.0.1:{port}/v1/realtime"])
    finally:
        server.should_exit = True


if __name__ == "__main__":
    main()
