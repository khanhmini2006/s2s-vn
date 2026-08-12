"""Mock LLM server có tool calling: nếu request chứa tool message → trả text,
ngược lại trả tool_call get_weather.
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        messages = body.get("messages", [])
        # nếu có message role=tool → đã có output, trả lời text
        has_tool_msg = any(m.get("role") == "tool" for m in messages)
        if has_tool_msg:
            chunks = ["Hà Nội hôm nay ", "trời nắng ", "đẹp ", "lắm."]
        else:
            # yêu cầu gọi tool
            body2 = json.dumps({
                "choices": [{"delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc123",
                        "function": {"name": "get_weather",
                                     "arguments": "{\"city\":\"Hanoi\"}"},
                    }]
                }}]
            })
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            data = ("data: " + body2 + "\n\n" + "data: [DONE]\n\n").encode()
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        parts = [f'data: {{"choices":[{{"delta":{{"content":"{c} "}}}}]}}\n\n'
                 for c in chunks]
        parts.append('data: {"choices":[],"usage":{"prompt_tokens":10,'
                     '"completion_tokens":8}}\n\n')
        parts.append("data: [DONE]\n\n")
        data = "".join(parts).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8082
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
