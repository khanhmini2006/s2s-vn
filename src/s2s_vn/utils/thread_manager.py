"""ThreadManager: chạy mỗi handler trong một thread."""

from __future__ import annotations

import threading
from collections.abc import Iterable


class ThreadManager:
    def __init__(self, handlers: Iterable):
        self.handlers = list(handlers)
        self.threads: list[threading.Thread] = []
        self.stop_event = threading.Event()
        for h in self.handlers:
            h.stop_event = self.stop_event

    def start(self) -> None:
        for h in self.handlers:
            t = threading.Thread(target=h.run, name=f"handler-{h.name}", daemon=True)
            self.threads.append(t)
            t.start()

    def stop(self, join_timeout: float = 5.0) -> None:
        self.stop_event.set()
        # join song song: tất cả join cùng lúc, deadline chung
        import time
        deadline = time.monotonic() + join_timeout
        for t in self.threads:
            remaining = max(0.1, deadline - time.monotonic())
            t.join(remaining)

    @property
    def alive_count(self) -> int:
        return sum(1 for t in self.threads if t.is_alive())
