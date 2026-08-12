"""Mock OpenAI-compatible server cho test (không cần key thật)."""
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        reply = ("Xin chào bạn, tôi khỏe, cảm ơn đã hỏi. "
                 "Hôm nay thật đẹp trời.")
        chunks = reply.split(" ")
        parts = [f'data: {{"choices":[{{"delta":{{"content":"{c} "}}}}]}}\n\n'
                 for c in chunks]
        parts.append('data: {"choices":[],"usage":{"prompt_tokens":15,'
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


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
