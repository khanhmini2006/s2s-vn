"""BaseHandler: một stage trong pipeline, chạy trong thread riêng."""

from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

In = TypeVar("In")
Out = TypeVar("Out")


class BaseHandler(Generic[In, Out], ABC):
    """Nhận input từ queue_in, process, đẩy output vào queue_out.

    Vòng lặp: get với timeout, gọi process(), put kết quả.
    PIPELINE_END là sentinel để thoát vòng lặp.
    """

    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        name: str,
        stop_event: threading.Event | None = None,
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.name = name
        self.stop_event = stop_event or threading.Event()

    def run(self) -> None:
        """Vòng lặp chính. Gọi từ thread manager."""
        while not self.stop_event.is_set():
            try:
                item = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            self._process_item(item)
        self._shutdown()

    def _process_item(self, item: In) -> None:
        try:
            outputs = self.process(item)
        except Exception as e:  # stage phải không chết dù lỗi
            self._handle_error(e, item)
            return
        if outputs is None:
            return
        if isinstance(outputs, (list, tuple)):
            for out in outputs:
                self._put(out)
        else:
            self._put(outputs)

    def _put(self, out: Out) -> None:
        self.output_queue.put(out)

    def _handle_error(self, e: Exception, item: In) -> None:
        print(f"[{self.name}] error processing: {e!r}")

    def _shutdown(self) -> None:
        pass

    @abstractmethod
    def process(self, item: In) -> Out | list[Out] | None:
        """Xử lý một input. Có thể trả về nhiều output."""


class PassThroughHandler(BaseHandler[In, Out]):
    """Chuyển input sang output không đổi. Dùng cho skeleton/handler giả."""

    def process(self, item: In) -> Out | list[Out] | None:
        return item  # type: ignore[return-value]


class SleepPassThroughHandler(PassThroughHandler[In, Out]):
    """PassThrough nhưng có delay mô phỏng thời gian xử lý thật."""

    def __init__(self, *args, delay_s: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_s = delay_s

    def process(self, item):
        time.sleep(self.delay_s)
        return item
