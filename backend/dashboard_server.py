"""Local dashboard API server for MeetMind (Option A: on-device only).

Provides:
- REST endpoints for commands and snapshots
- SSE stream for live UI updates
- localhost-only default behavior
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.agents.roomscribe.agent import RoomScribeAgent
from backend.agents.roomscribe.config import AgentConfig
from backend.agents.roomscribe.sources import (
    CameraOCRSource,
    MicrophoneSTTSource,
    SourceMessage,
    next_message,
)
from backend.core.config import ModelConfig, Perception
from backend.main import MeetMind

logger = logging.getLogger(__name__)


class PerceptionPayload(BaseModel):
    timestamp: str
    visual_text: list[str]
    visual_content_type: str
    visual_changed: bool
    audio_transcript: str
    audio_speech_detected: bool
    raw_image_path: str | None = None


class CaptureStartPayload(BaseModel):
    camera_interval: float = 4.0
    stt_model: str = "base"
    stt_language: str = "en"
    stt_chunk_seconds: float = 3.0
    stt_sample_rate: int = 16000
    ignore_people: bool = False


def _guess_visual_content_type(text: str) -> str:
    s = text.lower().strip()
    if not s:
        return "empty"
    if any(token in s for token in ("->", "flow", "diagram", "architecture", "pipeline")):
        return "diagram"
    if any(token in s for token in ("|", "table", "columns", "rows")):
        return "table"
    if any(token in s for token in ("- ", "•", "1.", "2.", "3.")):
        return "list"
    return "freeform"


class OCRWorker:
    def __init__(self, agent: RoomScribeAgent):
        self.agent = agent
        self._in_q: queue.Queue = queue.Queue(maxsize=1)
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


class RealtimeCaptureBridge:
    """Bridge RoomScribe camera+mic events to MeetMind Perception cycles."""

    def __init__(
        self,
        mm: MeetMind,
        hub: "LocalEventHub",
        lock: asyncio.Lock,
    ):
        self._mm = mm
        self._hub = hub
        self._lock = lock

        self._loop: asyncio.AbstractEventLoop | None = None
        self._msg_q: queue.Queue[SourceMessage] = queue.Queue()
        self._run_stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._agent: RoomScribeAgent | None = None
        self._ocr_worker: OCRWorker | None = None
        self._cam: CameraOCRSource | None = None
        self._mic: MicrophoneSTTSource | None = None

        self._state_lock = threading.Lock()
        self._running = False
        self._last_error = ""
        self._last_audio_ts = ""
        self._last_visual_ts = ""
        self._last_cycle_ts = ""
        self._cycles = 0
        self._audio_chunks = 0
        self._visual_events = 0

    def start(self, loop: asyncio.AbstractEventLoop, payload: CaptureStartPayload) -> dict[str, Any]:
        with self._state_lock:
            if self._running:
                return self.status()

            self._loop = loop
            self._run_stop.clear()
            self._last_error = ""
            self._last_audio_ts = ""
            self._last_visual_ts = ""
            self._last_cycle_ts = ""
            self._cycles = 0
            self._audio_chunks = 0
            self._visual_events = 0

            cfg = AgentConfig(
                camera_interval=max(0.5, payload.camera_interval),
                ignore_people=payload.ignore_people,
            )
            self._agent = RoomScribeAgent(cfg)
            self._ocr_worker = OCRWorker(self._agent)
            self._cam = CameraOCRSource(
                self._msg_q,
                interval_seconds=max(0.5, payload.camera_interval),
            )
            self._mic = MicrophoneSTTSource(
                self._msg_q,
                model_size=payload.stt_model,
                language=payload.stt_language,
                sample_rate=max(8000, payload.stt_sample_rate),
                chunk_seconds=max(1.0, payload.stt_chunk_seconds),
            )

            self._ocr_worker.start()
            self._cam.start()
            self._mic.start()

            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            self._running = True

        self._publish_status()
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._state_lock:
            if not self._running:
                return self.status()
            self._run_stop.set()

            if self._cam is not None:
                self._cam.stop()
                self._cam = None
            if self._mic is not None:
                self._mic.stop()
                self._mic = None
            if self._ocr_worker is not None:
                self._ocr_worker.stop()
                self._ocr_worker = None

            if self._thread is not None:
                self._thread.join(timeout=3)
                self._thread = None

            self._running = False

        self._publish_status()
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "running": self._running,
                "camera_active": self._cam is not None,
                "audio_active": self._mic is not None,
                "last_error": self._last_error,
                "last_audio_ts": self._last_audio_ts,
                "last_visual_ts": self._last_visual_ts,
                "last_cycle_ts": self._last_cycle_ts,
                "cycles": self._cycles,
                "audio_chunks": self._audio_chunks,
                "visual_events": self._visual_events,
                "model_id": self._agent.model_id if self._agent is not None else "",
            }

    def _run_loop(self) -> None:
        pending_audio: list[str] = []
        latest_visual = ""
        visual_changed = False
        flush_interval_s = 2.0
        last_flush = time.time()

        while not self._run_stop.is_set():
            if self._ocr_worker is not None:
                ocr_out = self._ocr_worker.poll()
                if ocr_out is not None:
                    if ocr_out.get("type") == "ocr" and ocr_out.get("text"):
                        text = str(ocr_out.get("text", "")).strip()
                        if text and text != latest_visual:
                            latest_visual = text
                            visual_changed = True
                            with self._state_lock:
                                self._visual_events += 1
                                self._last_visual_ts = datetime.now().strftime("%H:%M:%S")
                            self._publish_status()
                    elif ocr_out.get("type") == "error":
                        self._set_error(str(ocr_out.get("text", "OCR error")))

            msg = next_message(self._msg_q, timeout_seconds=0.3)
            if msg is not None:
                if msg.kind == "image" and self._ocr_worker is not None:
                    self._ocr_worker.submit(msg.payload)
                elif msg.kind == "stt":
                    text = str(msg.payload).strip()
                    if text:
                        pending_audio.append(text)
                        with self._state_lock:
                            self._audio_chunks += 1
                            self._last_audio_ts = datetime.now().strftime("%H:%M:%S")
                        self._publish_status()
                elif msg.kind == "error":
                    self._set_error(str(msg.payload))

            now = time.time()
            if now - last_flush < flush_interval_s:
                continue
            last_flush = now

            audio_text = " ".join(pending_audio).strip()
            if not audio_text and not visual_changed:
                continue

            visual_lines = [line.strip() for line in latest_visual.splitlines() if line.strip()]
            if not visual_lines and latest_visual.strip():
                visual_lines = [latest_visual.strip()]

            perception = Perception(
                timestamp=datetime.now().strftime("%H:%M:%S"),
                visual_text=visual_lines[:20],
                visual_content_type=_guess_visual_content_type(latest_visual),
                visual_changed=visual_changed,
                audio_transcript=audio_text[:1000],
                audio_speech_detected=bool(audio_text),
            )
            pending_audio = []
            visual_changed = False

            self._schedule_ingest(perception)

    def _schedule_ingest(self, perception: Perception) -> None:
        if self._loop is None:
            return
        fut = asyncio.run_coroutine_threadsafe(self._ingest(perception), self._loop)
        try:
            fut.result(timeout=20)
        except Exception as exc:
            self._set_error(f"Perception ingest error: {exc}")

    async def _ingest(self, perception: Perception) -> None:
        async with self._lock:
            result = await run_in_threadpool(self._mm.process_perception, perception)
            await self._hub.publish("cycle_result", result)
            await self._hub.publish("summary", self._mm.get_dashboard_payload())
            with self._state_lock:
                self._cycles += 1
                self._last_cycle_ts = datetime.now().strftime("%H:%M:%S")
            await self._hub.publish("capture_status", self.status())

    def _set_error(self, message: str) -> None:
        with self._state_lock:
            self._last_error = message
        self._publish_status()

    def _publish_status(self) -> None:
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._hub.publish("capture_status", self.status()),
            self._loop,
        )


class LocalEventHub:
    """Fan-out event hub for SSE subscribers."""

    def __init__(self):
        self._subscribers: set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        async with self._lock:
            self._subscribers.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue):
        async with self._lock:
            self._subscribers.discard(q)

    async def publish(self, event_type: str, data: dict[str, Any]):
        message = {"type": event_type, "data": data}
        stale: list[asyncio.Queue] = []
        async with self._lock:
            for q in self._subscribers:
                try:
                    q.put_nowait(message)
                except asyncio.QueueFull:
                    stale.append(q)
            for q in stale:
                self._subscribers.discard(q)


def _load_dashboard_html() -> str:
    project_root = Path(__file__).resolve().parent.parent
    html_path = project_root / "ui" / "dashboard" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<html><body><h2>MeetMind Dashboard</h2><p>UI file not found.</p></body></html>"


def _assert_local_policy(cfg: ModelConfig):
    cfg.validate()
    if not ModelConfig.is_loopback_host(cfg.dashboard_host):
        raise ValueError("Dashboard host must be localhost/127.0.0.1 for local desktop mode")


def create_dashboard_app(config: ModelConfig | None = None) -> FastAPI:
    cfg = config or ModelConfig()
    _assert_local_policy(cfg)

    app = FastAPI(title="MeetMind Local Dashboard", version="0.1.0")
    app.state.cfg = cfg
    app.state.meetmind = MeetMind(cfg)
    app.state.hub = LocalEventHub()
    app.state.lock = asyncio.Lock()
    app.state.capture_bridge = RealtimeCaptureBridge(
        app.state.meetmind,
        app.state.hub,
        app.state.lock,
    )

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_home():
        return _load_dashboard_html()

    @app.get("/api/health")
    async def health():
        mm: MeetMind = app.state.meetmind
        capture: RealtimeCaptureBridge = app.state.capture_bridge
        return {"ok": True, "health": mm.health, "capture": capture.status()}

    @app.get("/api/capture/status")
    async def capture_status():
        capture: RealtimeCaptureBridge = app.state.capture_bridge
        return {"ok": True, "capture": capture.status()}

    @app.post("/api/start-capture")
    async def start_capture(payload: CaptureStartPayload):
        capture: RealtimeCaptureBridge = app.state.capture_bridge
        loop = asyncio.get_running_loop()
        try:
            status = await run_in_threadpool(capture.start, loop, payload)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"ok": True, "capture": status}

    @app.post("/api/stop-capture")
    async def stop_capture():
        capture: RealtimeCaptureBridge = app.state.capture_bridge
        status = await run_in_threadpool(capture.stop)
        return {"ok": True, "capture": status}

    @app.get("/api/summary")
    async def summary():
        mm: MeetMind = app.state.meetmind
        return mm.get_dashboard_payload()

    @app.get("/api/events")
    async def recent_events(limit: int = 30):
        mm: MeetMind = app.state.meetmind
        events = mm.get_recent_events(limit=max(1, min(limit, 200)))
        normalized = []
        for e in events:
            data = e.get("data")
            if isinstance(data, str):
                try:
                    e["data"] = json.loads(data)
                except json.JSONDecodeError:
                    e["data"] = {"raw": data}
            normalized.append(e)
        return {"events": normalized}

    @app.post("/api/perceptions")
    async def ingest_perception(payload: PerceptionPayload):
        mm: MeetMind = app.state.meetmind
        hub: LocalEventHub = app.state.hub
        lock: asyncio.Lock = app.state.lock

        async with lock:
            perception = Perception(**payload.model_dump())
            result = await run_in_threadpool(mm.process_perception, perception)
            await hub.publish("cycle_result", result)
            await hub.publish("summary", mm.get_dashboard_payload())
            return result

    @app.post("/api/perceptions/sample")
    async def ingest_sample_perception():
        """Local test helper to drive dashboard without Agent 1 integration."""
        sample = Perception(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            visual_text=["API Gateway", "Auth Service", "PostgreSQL"],
            visual_content_type="diagram",
            visual_changed=True,
            audio_transcript="Sarah will implement OAuth by Friday. We decided PostgreSQL over MongoDB.",
            audio_speech_detected=True,
        )
        mm: MeetMind = app.state.meetmind
        hub: LocalEventHub = app.state.hub
        lock: asyncio.Lock = app.state.lock

        async with lock:
            result = await run_in_threadpool(mm.process_perception, sample)
            await hub.publish("cycle_result", result)
            await hub.publish("summary", mm.get_dashboard_payload())
            return result

    @app.post("/api/reset")
    async def reset_session():
        mm: MeetMind = app.state.meetmind
        hub: LocalEventHub = app.state.hub
        lock: asyncio.Lock = app.state.lock

        async with lock:
            await run_in_threadpool(mm.reset)
            payload = {"ok": True, "message": "session_reset"}
            await hub.publish("session", payload)
            await hub.publish("summary", mm.get_dashboard_payload())
            return payload

    @app.post("/api/artifacts/meeting-summary")
    async def generate_meeting_summary():
        mm: MeetMind = app.state.meetmind
        hub: LocalEventHub = app.state.hub
        lock: asyncio.Lock = app.state.lock

        async with lock:
            try:
                content = await run_in_threadpool(mm.generate_meeting_summary)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            payload = {"type": "meeting_summary", "content": content}
            await hub.publish("artifact", payload)
            return payload

    @app.get("/api/stream")
    async def stream():
        cfg_local: ModelConfig = app.state.cfg
        hub: LocalEventHub = app.state.hub
        mm: MeetMind = app.state.meetmind

        async def event_generator():
            q = await hub.subscribe()
            try:
                # Initial snapshot
                yield {
                    "event": "summary",
                    "data": json.dumps(mm.get_dashboard_payload()),
                }
                capture: RealtimeCaptureBridge = app.state.capture_bridge
                yield {
                    "event": "capture_status",
                    "data": json.dumps(capture.status()),
                }

                while True:
                    try:
                        msg = await asyncio.wait_for(q.get(), timeout=cfg_local.sse_heartbeat_s)
                        yield {
                            "event": msg["type"],
                            "data": json.dumps(msg["data"]),
                        }
                    except asyncio.TimeoutError:
                        yield {
                            "event": "heartbeat",
                            "data": json.dumps({"ok": True}),
                        }
            finally:
                await hub.unsubscribe(q)

        return EventSourceResponse(event_generator())

    return app


def run_dashboard_server(config: ModelConfig | None = None):
    cfg = config or ModelConfig()
    _assert_local_policy(cfg)
    app = create_dashboard_app(cfg)
    logger.info(
        "Starting MeetMind local dashboard on http://%s:%d (air_gapped=%s)",
        cfg.dashboard_host,
        cfg.dashboard_port,
        cfg.air_gapped,
    )
    uvicorn.run(app, host=cfg.dashboard_host, port=cfg.dashboard_port)
