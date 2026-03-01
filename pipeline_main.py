#!/usr/bin/env python3
"""MeetMind Pipeline — single-process, all 3 agents, Apple Silicon safe.

Architecture
  Agent 1 (Perception):  Mic (faster-whisper, CPU) + optional Camera (VLM, GPU)
  Agent 2 (Scribe):      Batches perceptions every 5 s → updates one MeetingState
  Agent 3 (Analyst):     Reads latest MeetingState → outputs action + reasoning

Memory budget (avoids crash on 16 GB M-series):
  Mic STT:     ~150 MB  (faster-whisper base, CPU-only)
  Text LLM:    ~2.5 GB  (gemma-3-4b-it-4bit, shared Scribe + Analyst)
  Camera VLM:  ~2.5 GB  (only if --camera; coexists via GPU lock)

Every batch window writes to  data/session_<ts>.jsonl:
  scribe_batch  → list of deduped perceptions
  meeting_state → full MeetingState snapshot
  analyst_action→ action + params + reasoning

Usage
  python pipeline_main.py                    # mic-only (~3 GB, safest)
  python pipeline_main.py --camera           # mic + camera (~5.5 GB)
  python pipeline_main.py --duration 60      # 1-min demo, then stop
  python pipeline_main.py --stdin            # type lines manually (testing)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import queue
import re
import signal
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("pipeline")
log.setLevel(logging.INFO)

for _n in ("httpx", "httpcore", "urllib3", "faster_whisper"):
    logging.getLogger(_n).setLevel(logging.WARNING)


# ══════════════════════════════════════════════════════════════════════
#  Tiny helpers
# ══════════════════════════════════════════════════════════════════════

def ts_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

RESET_REQUESTED = threading.Event()

class Deduplicator:
    """Sliding-window fuzzy dedup for STT/OCR text."""

    def __init__(self, threshold: float = 0.75, cap: int = 30):
        self._hist: list[str] = []
        self._thr = threshold
        self._cap = cap

    def is_dup(self, text: str) -> bool:
        n = text.lower().strip()
        if not n or len(n) < 4:
            return True
        for prev in self._hist:
            if SequenceMatcher(None, n, prev).ratio() > self._thr:
                return True
        self._hist.append(n)
        if len(self._hist) > self._cap:
            self._hist = self._hist[-self._cap :]
        return False


class SessionLog:
    """Append-only JSONL session file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "a", encoding="utf-8")
        self.path = path

    def write(self, record: dict):
        self._f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


# ══════════════════════════════════════════════════════════════════════
#  Embedded Dashboard (SSE push to existing UI)
# ══════════════════════════════════════════════════════════════════════


class DashboardHub:
    """Thread-safe fan-out hub. Pipeline thread publishes, SSE clients consume."""

    def __init__(self) -> None:
        self._queues: list[queue.Queue] = []
        self._lock = threading.Lock()
        # Cache latest payloads so new SSE clients get an immediate snapshot
        self._latest_summary: dict = {}
        self._latest_capture: dict = {}

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=64)
        with self._lock:
            self._queues.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            try:
                self._queues.remove(q)
            except ValueError:
                pass

    def publish(self, event_type: str, data: dict) -> None:
        if event_type == "summary":
            self._latest_summary = data
        elif event_type == "capture_status":
            self._latest_capture = data
        msg = {"type": event_type, "data": data}
        with self._lock:
            dead: list[queue.Queue] = []
            for q in self._queues:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                try:
                    self._queues.remove(q)
                except ValueError:
                    pass


def _build_summary_payload(
    scribe: Any,
    analyst: Any,
    batch_n: int,
    avg_cycle: float,
    running: bool,
) -> dict:
    """Build a dashboard-compatible summary dict from Scribe + Analyst state."""
    s = scribe.state
    last_action = analyst.action_history[-1] if analyst.action_history else None
    suggestions = [
        t.get("content", "")
        for t in s.timeline
        if t.get("type") == "suggestion"
    ][-10:]
    insights = [
        t.get("content", "")
        for t in s.timeline
        if t.get("type") == "insight"
    ][-10:]
    return {
        "topic": s.topic,
        "domain": s.domain,
        "key_points": s.key_points,
        "action_items": [asdict(a) for a in s.action_items],
        "decisions": [asdict(d) for d in s.decisions],
        "gaps": [asdict(g) for g in s.gaps],
        "suggestions": suggestions,
        "insights": insights,
        "artifacts": s.artifacts,
        "timeline": s.timeline[-20:],
        "whiteboard": s.whiteboard_content,
        "perception_count": scribe.perception_count,
        "avg_cycle_time": avg_cycle,
        "air_gapped": False,
        "last_action": last_action,
        "suggestions": suggestions,
        "insights": insights,
    }


def start_dashboard_server(hub: DashboardHub, port: int = 8765) -> None:
    """Launch a lightweight FastAPI server in a daemon thread.

    Serves the existing dashboard UI and an SSE stream fed by DashboardHub.
    """
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from sse_starlette.sse import EventSourceResponse
    import uvicorn

    app = FastAPI(title="MeetMind Pipeline Dashboard")

    project_root = Path(__file__).resolve().parent
    assets_dir = project_root / "ui" / "dashboard"
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    html_path = assets_dir / "index.html"
    html_content = html_path.read_text(encoding="utf-8") if html_path.exists() else "<h1>UI not found</h1>"

    @app.get("/", response_class=HTMLResponse)
    async def root():
        return html_content

    @app.get("/api/summary")
    async def api_summary():
        return hub._latest_summary or {"topic": "Unknown", "domain": "general"}

    @app.get("/api/health")
    async def api_health():
        return {
            "ok": True,
            "health": {"scribe_healthy": True, "cycles": 0, "avg_cycle_time": 0, "air_gapped": False},
            "capture": hub._latest_capture,
        }

    @app.get("/api/capture/status")
    async def api_capture_status():
        return {"ok": True, "capture": hub._latest_capture}

    @app.get("/api/model-status")
    async def api_model_status():
        return {"ok": True, "status": {
            "vlm": {"status": "ready", "model_id": "pipeline"},
            "scribe": {"status": "ready", "model_id": "pipeline"},
            "analyst": {"status": "ready", "model_id": "pipeline"},
        }}

    # SSE endpoints
    @app.post("/api/reset")
    async def api_reset():
        RESET_REQUESTED.set()
        return {"ok": True}

    @app.post("/api/start-capture")
    async def api_start_capture():
        return {"ok": True, "capture": hub._latest_capture}

    @app.post("/api/stop-capture")
    async def api_stop_capture():
        return {"ok": True, "capture": hub._latest_capture}

    @app.post("/api/sample")
    @app.post("/api/perceptions/sample")
    async def api_sample():
        return {"ok": True}

    @app.post("/api/artifacts/meeting-summary")
    async def api_gen_summary():
        return {"type": "meeting_summary", "content": "(pipeline mode — use --duration to end session)"}

    @app.get("/api/stream")
    async def sse_stream():
        client_q = hub.subscribe()

        async def generator():
            try:
                # Send initial snapshot
                if hub._latest_summary:
                    yield {"event": "summary", "data": json.dumps(hub._latest_summary, default=str)}
                if hub._latest_capture:
                    yield {"event": "capture_status", "data": json.dumps(hub._latest_capture, default=str)}
                # Stream live events
                while True:
                    try:
                        msg = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: client_q.get(timeout=2.0)
                        )
                        yield {
                            "event": msg["type"],
                            "data": json.dumps(msg["data"], default=str),
                        }
                    except queue.Empty:
                        yield {"event": "heartbeat", "data": json.dumps({"ok": True})}
            finally:
                hub.unsubscribe(client_q)

        return EventSourceResponse(generator())

    def _run():
        # Suppress uvicorn access logs to keep terminal clean
        uvicorn.run(
            app, host="127.0.0.1", port=port,
            log_level="warning", access_log=False,
        )

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    log.info("Dashboard: http://127.0.0.1:%d", port)


# ══════════════════════════════════════════════════════════════════════
#  Agent 1 — perception sources
# ══════════════════════════════════════════════════════════════════════

PercEvent = dict[str, str]  # {"timestamp", "event_type", "text"}


def start_mic(
    out_q: queue.Queue[PercEvent],
    stop: threading.Event,
    model_size: str = "base",
    language: str = "en",
    chunk_seconds: float = 3.0,
):
    """Microphone → faster-whisper → PercEvent queue (daemon)."""
    from backend.agents.roomscribe.sources import MicrophoneSTTSource, SourceMessage

    raw_q: queue.Queue[SourceMessage] = queue.Queue()
    stt = MicrophoneSTTSource(
        raw_q,
        model_size=model_size,
        language=language,
        chunk_seconds=chunk_seconds,
    )
    stt.start()

    def _relay():
        while not stop.is_set():
            try:
                msg = raw_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if msg.kind == "stt" and msg.payload:
                t = str(msg.payload).strip()
                if t:
                    out_q.put({"timestamp": ts_now(), "event_type": "stt_raw", "text": t})
            elif msg.kind == "error":
                log.warning("Mic: %s", msg.payload)

    threading.Thread(target=_relay, daemon=True).start()
    return stt  # .stop() on shutdown


def start_camera(
    out_q: queue.Queue[PercEvent],
    vlm_agent: Any,
    stop: threading.Event,
    interval: float = 5.0,
):
    """Camera grab → VLM OCR → PercEvent queue (daemon)."""
    import cv2
    from PIL import Image

    def _loop():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log.warning("Camera not available — skipping.")
            print("\n❌ CRITICAL: Camera unavailable or blocked by macOS permissions. Video stream is offline.\n")
            return
        log.info("Camera open (every %.1fs)", interval)
        try:
            while not stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    stop.wait(1)
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                try:
                    ev = vlm_agent.image_to_text(img)
                    txt = re.sub(r"<\w+>", "", (ev.text or "")).strip()
                    if txt and txt != "NO_MEETING_CONTENT" and len(txt) > 3:
                        out_q.put(
                            {"timestamp": ts_now(), "event_type": "ocr", "text": txt}
                        )
                except Exception as exc:
                    log.warning("OCR: %s", exc)
                stop.wait(interval)
        finally:
            cap.release()

    threading.Thread(target=_loop, daemon=True).start()


def start_stdin(out_q: queue.Queue[PercEvent], stop: threading.Event):
    """Manual text entry for testing (instead of mic)."""

    def _loop():
        print("  Type lines to simulate speech. 'exit' to quit.\n")
        while not stop.is_set():
            try:
                line = input("  you> ").strip()
            except EOFError:
                stop.set()
                return
            if not line:
                continue
            if line.lower() in {"quit", "exit"}:
                stop.set()
                return
            out_q.put({"timestamp": ts_now(), "event_type": "stt_raw", "text": line})

    threading.Thread(target=_loop, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════
#  Main pipeline loop
# ══════════════════════════════════════════════════════════════════════


def drain(q: queue.Queue) -> list:
    out: list = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            break
    return out


def fmt_state(s: Any) -> str:
    return (
        f"topic={s.topic} | domain={s.domain} | "
        f"pts={len(s.key_points)} acts={len(s.action_items)} "
        f"decs={len(s.decisions)} gaps={len(s.gaps)}"
    )


def fmt_action(a: dict) -> str:
    name = a.get("action", "?")
    reason = a.get("reasoning", "")[:100]
    params = a.get("params", {})
    if name == "continue_observing":
        return f"{name}: {params.get('reason', reason)[:100]}"
    ps = json.dumps(params, default=str)
    if len(ps) > 120:
        ps = ps[:120] + "…"
    return f"{name}: {reason}  {ps}"


# ──────────────────────────────────────────────────────────────────────


def run(args: argparse.Namespace):
    stop = threading.Event()
    pq: queue.Queue[PercEvent] = queue.Queue()
    dedup = Deduplicator()

    sid = datetime.now().strftime("%Y%m%d_%H%M%S")
    slog = SessionLog(Path("data") / f"session_{sid}.jsonl")

    # ── Dashboard (optional) ─────────────────────────────────────────
    hub: DashboardHub | None = None
    if args.ui:
        hub = DashboardHub()
        start_dashboard_server(hub, port=args.port)
        # Publish initial capture status
        hub.publish("capture_status", {"running": True, "camera_active": args.camera, "audio_active": not args.stdin})

    # ── Load text LLM (shared Scribe + Analyst) ─────────────────────
    log.info("Loading text LLM…")
    from backend.core.gemma import get_model
    from backend.agents.scribe import Scribe
    from backend.agents.analyst import Analyst
    from backend.core.config import Perception

    text_id = args.text_model
    analyst_id = args.analyst_model

    if analyst_id and not Path(analyst_id).expanduser().exists():
        log.info("Analyst model not found locally → sharing text model (saves RAM)")
        analyst_id = text_id

    text_llm = get_model(text_id, backend="mlx")
    analyst_llm = text_llm if analyst_id == text_id else get_model(analyst_id, backend="mlx")

    scribe = Scribe(text_llm)
    analyst = Analyst(analyst_llm)
    
    # Force GPU allocation *before* starting actual transcription to prevent mid-conversation stalls
    log.info("Warming up Text LLM on GPU...")
    if hasattr(text_llm, '_ensure_loaded'):
        text_llm._ensure_loaded()
    elif hasattr(text_llm, 'generate'):
        text_llm.generate("warmup", max_tokens=1)
    log.info("Text LLM ready")

    # ── Perception sources ───────────────────────────────────────────
    mic_handle = None
    if args.stdin:
        start_stdin(pq, stop)
    else:
        log.info("Starting mic (whisper-%s, chunk=%.1fs)…", args.stt_model, args.stt_chunk)
        mic_handle = start_mic(pq, stop, args.stt_model, args.stt_language, args.stt_chunk)
        log.info("Mic ready")

    vlm_agent = None
    if args.camera:
        log.info("Loading VLM for camera…")
        from backend.agents.roomscribe.agent import RoomScribeAgent
        from backend.agents.roomscribe.config import AgentConfig

        vlm_cfg = AgentConfig(
            model=args.vlm_model or None,
            camera_interval=args.camera_interval,
            max_tokens=200,
            temperature=0.1,
            ignore_people=True,
        )
        vlm_agent = RoomScribeAgent(vlm_cfg)
        
        # Force VLM GPU Allocation so it doesn't freeze the first 15 seconds of the meeting
        log.info("Warming up VLM Camera Model on GPU... (this takes a moment)")
        if hub:
            hub.publish("model_status", {"vlm": {"status": "loading", "model_id": vlm_cfg.model or "default"}})
        
        # Camera Thread Launch
        start_camera(pq, vlm_agent, stop, args.camera_interval)
        log.info("Camera ready (model=%s)", vlm_agent.model_id)
        if hub:
            hub.publish("model_status", {
                "vlm": {"status": "ready", "model_id": vlm_agent.model_id},
                "scribe": {"status": "ready", "model_id": text_id},
                "analyst": {"status": "ready", "model_id": analyst_id},
            })

    # ── Signals ──────────────────────────────────────────────────────
    def _sig(sig, frame):
        log.info("Stopping…")
        stop.set()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    # ── Banner ───────────────────────────────────────────────────────
    src = "stdin" if args.stdin else "mic"
    if args.camera:
        src += " + camera"
    dur = f"{args.duration}s" if args.duration > 0 else "unlimited"

    print()
    print("=" * 60)
    print("  MeetMind Pipeline — Live Demo")
    print(f"  LLM:      {text_id}")
    if analyst_id != text_id:
        print(f"  Analyst:  {analyst_id}")
    print(f"  Sources:  {src}")
    print(f"  Batch:    every {args.batch_window}s  |  Duration: {dur}")
    if hub:
        print(f"  Dashboard: http://127.0.0.1:{args.port}")
    print(f"  Log:      {slog.path}")
    print("=" * 60)
    print("  Listening… (Ctrl+C to stop)\n")

    # ── Batch loop ───────────────────────────────────────────────────
    t0 = time.monotonic()
    batch_n = 0

    while not stop.is_set():
        if RESET_REQUESTED.is_set():
            from backend.core.meeting_state import MeetingState
            log.info("Resetting session state...")
            scribe._state = MeetingState()
            analyst._action_history = []
            if hub:
                hub._latest_summary = {}
            # Drain queue to clear pending audio
            drain(pq)
            RESET_REQUESTED.clear()
            print("  --- SESSION RESET ---")

        elapsed = time.monotonic() - t0
        if 0 < args.duration <= elapsed:
            log.info("Duration limit (%ss) reached.", args.duration)
            break

        stop.wait(timeout=args.batch_window)
        if stop.is_set():
            break

        raw = drain(pq)
        batch_n += 1

        clean = [e for e in raw if not dedup.is_dup(e.get("text", ""))]

        if not clean:
            tag = "(no input)" if not raw else f"({len(raw)} dup)"
            print(f"  [{batch_n:03d}] {tag}")
            continue

        n_stt = sum(1 for e in clean if e["event_type"] == "stt_raw")
        n_ocr = sum(1 for e in clean if e["event_type"] == "ocr")
        print(f"  [{batch_n:03d}] batch: {len(clean)} events (stt={n_stt} ocr={n_ocr})")

        slog.write(
            {"type": "scribe_batch", "batch": batch_n, "ts": ts_now(), "events": clean}
        )

        # Push raw perceptions to dashboard
        if hub:
            for ev in clean:
                etype = "stt" if ev["event_type"] == "stt_raw" else "ocr"
                hub.publish("raw_perception", {"type": etype, "text": ev["text"]})

        # ── Scribe: feed perceptions → update MeetingState ──
        state = scribe.state  # fallback if loop somehow empty
        
        # Performance fix: Squash perceptions to avoid N consecutive Scribe LLM generations
        stt_chunks = [ev["text"] for ev in clean if ev["event_type"] == "stt_raw"]
        ocr_chunks = [ev["text"] for ev in clean if ev["event_type"] == "ocr"]
        
        batched = []
        has_valuable_stt = False
        has_valuable_ocr = False
        
        stt_joined = " ".join(stt_chunks).strip()
        ocr_joined = " | ".join(ocr_chunks).strip()
        
        # Don't waste LLM time on empty/tiny audio bursts or empty visual frames
        if stt_joined and len(stt_joined.split()) > 2:
            batched.append({"timestamp": ts_now(), "event_type": "stt_raw", "text": stt_joined})
            has_valuable_stt = True
            
        if ocr_joined and "NO_MEETING_CONTENT" not in ocr_joined:
            batched.append({"timestamp": ts_now(), "event_type": "ocr", "text": ocr_joined})
            has_valuable_ocr = True

        for ev in batched:
            p = Perception(
                timestamp=ev["timestamp"],
                event_type=ev["event_type"],
                text=ev["text"],
            )
            try:
                state = scribe.process(p)
            except Exception as exc:
                log.warning("Scribe err: %s", exc)
                state = scribe.state

        slog.write(
            {"type": "meeting_state", "batch": batch_n, "ts": ts_now(), "state": asdict(state)}
        )
        print(f"        state:  {fmt_state(state)}")

        # Push state to dashboard
        if hub:
            elapsed_so_far = time.monotonic() - t0
            avg = round(elapsed_so_far / max(batch_n, 1), 2)
            summary = _build_summary_payload(scribe, analyst, batch_n, avg, True)
            hub.publish("summary", summary)

        # ── Analyst: decide action ──
        # Skip analyst logic if nothing meaningful happened this cycle to save ~8 seconds
        if not has_valuable_stt and not has_valuable_ocr:
            log.info("Skipping Analyst generation (empty cycle)")
            action = {"action": "continue_observing", "params": {"reason": "Not enough new context to analyze yet."}, "reasoning": "adaptive skip"}
        else:
            try:
                action = analyst.decide(state)
            except Exception as exc:
                log.warning("Analyst err: %s", exc)
                action = {
                    "action": "continue_observing",
                    "params": {"reason": str(exc)},
                    "reasoning": "error",
                }

        analyst.apply_action(action, state)
        slog.write(
            {"type": "analyst_action", "batch": batch_n, "ts": ts_now(), "action": action}
        )
        print(f"        action: {fmt_action(action)}")
        print()

        # Push cycle result + updated summary to dashboard
        if hub:
            hub.publish("cycle_result", {"action": action, "artifact": {}})
            elapsed_so_far = time.monotonic() - t0
            avg = round(elapsed_so_far / max(batch_n, 1), 2)
            hub.publish("summary", _build_summary_payload(scribe, analyst, batch_n, avg, True))

    # ── Shutdown ─────────────────────────────────────────────────────
    stop.set()
    if mic_handle and hasattr(mic_handle, "stop"):
        mic_handle.stop()

    elapsed = time.monotonic() - t0
    state = scribe.state

    print()
    print("=" * 60)
    print("  Session Complete")
    print(f"  Duration:    {elapsed:.0f}s  |  Batches: {batch_n}")
    print(f"  Topic:       {state.topic}")
    print(f"  Key points:  {len(state.key_points)}")
    for kp in state.key_points[-5:]:
        print(f"    - {kp[:80]}")
    print(f"  Actions:     {len(state.action_items)}")
    for ai in state.action_items:
        print(f"    - {ai.owner}: {ai.task}")
    print(f"  Decisions:   {len(state.decisions)}")
    print(f"  Gaps:        {len(state.gaps)}")
    print(f"  Log:         {slog.path}")
    print("=" * 60)

    slog.write(
        {
            "type": "session_end",
            "ts": ts_now(),
            "duration_s": round(elapsed, 1),
            "batches": batch_n,
            "final_state": asdict(state),
            "action_history": analyst.action_history,
        }
    )
    slog.close()


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════


def main():
    p = argparse.ArgumentParser(
        description="MeetMind Pipeline — single-process, all 3 agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline_main.py                    # mic only (~3 GB)
  python pipeline_main.py --camera           # mic + camera (~5.5 GB)
  python pipeline_main.py --duration 60      # 1-min demo
  python pipeline_main.py --stdin            # type input (testing)
  python pipeline_main.py --stt-model tiny   # lighter STT
  python pipeline_main.py --ui                # with live dashboard
  python pipeline_main.py --ui --stdin        # dashboard + manual input
""",
    )
    p.add_argument("--camera", action="store_true", help="Enable camera OCR (+~2.5 GB)")
    p.add_argument("--stdin", action="store_true", help="Stdin text instead of mic")
    p.add_argument(
        "--batch-window", type=float, default=5.0, help="Batch interval sec (default: 5)"
    )
    p.add_argument(
        "--duration", type=float, default=0, help="Auto-stop after N sec (0=unlimited)"
    )
    p.add_argument(
        "--camera-interval", type=float, default=5.0, help="Camera capture sec (default: 5)"
    )
    p.add_argument(
        "--stt-model", default="base", help="Whisper: tiny/base/small (default: base)"
    )
    p.add_argument("--stt-language", default="en", help="STT language (default: en)")
    p.add_argument(
        "--stt-chunk", type=float, default=3.0, help="Mic chunk sec (default: 3)"
    )
    p.add_argument(
        "--text-model",
        default="mlx-community/gemma-3-4b-it-4bit",
        help="Text LLM for Scribe + Analyst fallback",
    )
    p.add_argument(
        "--analyst-model",
        default="outputs/gpu-analyst-fused-4bit",
        help="Analyst model (falls back to text-model if missing)",
    )
    p.add_argument("--vlm-model", default=None, help="VLM for camera (auto if unset)")
    p.add_argument("--ui", action="store_true", help="Launch live dashboard UI")
    p.add_argument("--port", type=int, default=8765, help="Dashboard port (default: 8765)")

    args = p.parse_args()
    if args.stdin and args.camera:
        p.error("--stdin and --camera are mutually exclusive")
    run(args)


if __name__ == "__main__":
    main()
