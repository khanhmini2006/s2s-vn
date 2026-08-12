"""Test LLM Proxy /v1/chat/completions — dùng mock OpenAI-compatible server local."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from s2s_vn.api.openai_realtime.server import create_app
from s2s_vn.s2s_pipeline import PipelineConfig

MOCK_REPLY = "Xin chào bạn, tôi khỏe, cảm ơn đã hỏi. Hôm nay thật đẹp trời."


class MockHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        chunks = MOCK_REPLY.split(" ")
        parts = [f'data: {{"choices":[{{"delta":{{"content":"{c} "}}}}]}}\n\n'
                 for c in chunks]
        parts.append('data: {"choices":[],"usage":{"prompt_tokens":10,'
                     '"completion_tokens":12}}\n\n')
        parts.append("data: [DONE]\n\n")
        body = "".join(parts).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def mock_llm():
    server = HTTPServer(("127.0.0.1", 0), MockHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def make_client(mock_llm):
    cfg = PipelineConfig(llm_backend="local", llm_base_url=mock_llm)
    return TestClient(create_app(cfg))


def test_proxy_non_stream(mock_llm):
    c = make_client(mock_llm)
    r = c.post("/v1/chat/completions", json={
        "model": "mock",
        "messages": [{"role": "user", "content": "xin chào"}],
    })
    assert r.status_code == 200
    data = r.json()
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "Xin chào bạn" in data["choices"][0]["message"]["content"]
    assert "usage" in data  # proxy không thu usage chi tiết (handler emit qua text_out)


def test_proxy_stream(mock_llm):
    c = make_client(mock_llm)
    r = c.post("/v1/chat/completions", json={
        "model": "mock",
        "messages": [{"role": "user", "content": "xin chào"}],
        "stream": True,
    })
    assert r.status_code == 200
    body = r.text
    assert "data: [DONE]" in body
    # delta stream tách từng từ — không nối liền
    assert "Xin" in body and "chào" in body


def test_proxy_passes_history(mock_llm):
    c = make_client(mock_llm)
    r = c.post("/v1/chat/completions", json={
        "model": "mock",
        "messages": [
            {"role": "system", "content": "Bạn là kế toán."},
            {"role": "user", "content": "Báo cáo Q2?"},
        ],
    })
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"]


def test_proxy_requires_messages(mock_llm):
    c = make_client(mock_llm)
    r = c.post("/v1/chat/completions", json={"model": "mock"})
    assert r.status_code == 400
