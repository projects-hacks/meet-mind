from __future__ import annotations

import queue
import threading
from typing import Any

from backend.agents.roomscribe.agent import RoomScribeAgent


class OCRWorker:
    def __init__(self, agent: RoomScribeAgent):
        self.agent = agent
        self._in_q: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._out_q: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._in_q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2)

    def submit(self, image: Any) -> None:
        if self._in_q.full():
            return
        try:
            self._in_q.put_nowait(image)
        except queue.Full:
            return

    def poll(self) -> dict[str, Any] | None:
        try:
            return self._out_q.get_nowait()
        except queue.Empty:
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            image = self._in_q.get()
            if image is None:
                return
            try:
                event = self.agent.image_to_text(image)
                if event.text:
                    self._out_q.put(self.agent.to_json(event))
            except Exception as exc:
                self._out_q.put({"type": "error", "text": f"OCR worker error: {exc}"})
