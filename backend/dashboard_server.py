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
from pathlib import Path
from typing import Any
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.core.config import ModelConfig, Perception
from backend.main import MeetMind

logger = logging.getLogger(__name__)


class PerceptionPayload(BaseModel):
    timestamp: str
    event_type: str
    text: str


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

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_home():
        return _load_dashboard_html()

    @app.get("/api/health")
    async def health():
        mm: MeetMind = app.state.meetmind
        return {"ok": True, "health": mm.health}

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
            event_type="ocr",
            text="API Gateway, Auth Service, PostgreSQL. Decision: use PostgreSQL over MongoDB.",
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
