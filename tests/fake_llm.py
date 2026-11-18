"""Fake LLM gateway: an OpenAI-compatible /v1/chat/completions server for zero-spend tests.

Nothing here calls a real model; it never leaves this machine. Python stdlib only.

The behaviour is chosen by the API key the bot sends, so a test can switch modes between matches simply by
putting a different key in `start["llm"]` (which also checks that the bot uses the key from the latest message):

    fake-ok-...     answers at once with FAKE_LLM_REPLY (default: a reply that names no move, so the bot's
                    own parsing/fallback decides)
    fake-slow-...   sleeps FAKE_LLM_SLOW seconds (default 45, longer than any move clock), then answers
    fake-err-...    HTTP 402, like a key that ran out of budget
    fake-empty-...  200 with empty content (like Haiku spending every token on thinking)
    fake-cut-...    200 with a truncated reply and finish_reason "length"
    anything else   HTTP 401 (unknown key)

Every request is appended as one JSON line to the log file (time, key, mode, path, the full request body),
so a checker can ask: which keys were used, and did the opponent's chat text reach the model?

Run it alone:   python tests/fake_llm.py --port 4555 --log /tmp/fake_llm.jsonl
Then point a bot at it: LLM_BASE_URL=http://127.0.0.1:4555/v1 LLM_API_KEY=fake-ok-dev
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_REPLY = os.environ.get(
    "FAKE_LLM_REPLY", "I am a fake model used in tests. I will not pick a move for you.")
SLOW_SECONDS = float(os.environ.get("FAKE_LLM_SLOW", "45"))


def mode_of(key: str) -> str:
    for mode in ("ok", "slow", "err", "empty", "cut"):
        if key.startswith(f"fake-{mode}"):
            return mode
    return "unknown"


class FakeGateway:
    """Run in a thread: `with FakeGateway(log_path) as gw: ... gw.base_url ... gw.requests()`."""

    def __init__(self, log_path: str, host: str = "127.0.0.1", port: int = 0, reply: str = DEFAULT_REPLY):
        self.log_path = log_path
        self.reply = reply
        self._lock = threading.Lock()
        gw = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # keep the test output clean
                pass

            def _json(self, code: int, obj: dict):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _key(self) -> str:
                auth = self.headers.get("Authorization", "")
                return auth.split(" ", 1)[1].strip() if auth.lower().startswith("bearer ") else ""

            def do_GET(self):
                key = self._key()
                gw._record(key, self.path, None)
                if self.path.rstrip("/").endswith("/key/info"):
                    self._json(200, {"key": key[:12] + "...", "info": {"spend": 0.0, "max_budget": 0.75}})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw or b"{}")
                except ValueError:
                    body = {"_unparsed": raw.decode("utf-8", "replace")}
                key = self._key()
                mode = mode_of(key)
                gw._record(key, self.path, body)
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    return self._json(404, {"error": {"message": "only /v1/chat/completions is faked"}})
                if mode == "unknown":
                    return self._json(401, {"error": {"message": "fake gateway: unknown key", "type": "auth_error"}})
                if mode == "err":
                    return self._json(402, {"error": {"message": "fake gateway: budget exceeded", "type": "budget_exceeded"}})
                if mode == "slow":
                    time.sleep(SLOW_SECONDS)
                text, finish = gw.reply, "stop"
                if mode == "empty":
                    text = ""
                elif mode == "cut":
                    text, finish = gw.reply[: max(1, len(gw.reply) // 3)], "length"
                if body.get("stream"):
                    return self._stream(text, finish, body.get("model", "haiku"))
                self._json(200, {
                    "id": "fake-" + str(int(time.time() * 1000)), "object": "chat.completion",
                    "created": int(time.time()), "model": body.get("model", "haiku"),
                    "choices": [{"index": 0, "finish_reason": finish,
                                 "message": {"role": "assistant", "content": text}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                })

            def _stream(self, text: str, finish: str, model: str):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                base = {"id": "fake-stream", "object": "chat.completion.chunk", "created": int(time.time()), "model": model}
                chunks = [{**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]},
                          {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}]
                try:
                    for c in chunks:
                        self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):  # a bot hanging up mid-call is normal here
                pass

        self.server = Server((host, port), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def _record(self, key: str, path: str, body):
        line = json.dumps({"t": time.time(), "key": key, "mode": mode_of(key), "path": path, "body": body})
        with self._lock, open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def base_url(self, host: str = "127.0.0.1") -> str:
        return f"http://{host}:{self.port}/v1"

    def requests(self) -> list[dict]:
        try:
            with open(self.log_path, encoding="utf-8") as fh:
                return [json.loads(x) for x in fh if x.strip()]
        except OSError:
            return []

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from a Docker container")
    ap.add_argument("--port", type=int, default=4555)
    ap.add_argument("--log", default="fake_llm.jsonl")
    a = ap.parse_args()
    with FakeGateway(a.log, a.host, a.port) as gw:
        print(f"fake gateway on {gw.base_url(a.host)}, logging to {a.log} (Ctrl-C to stop)", flush=True)
        try:
            gw.thread.join()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
