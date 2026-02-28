from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import queue
import threading

from backend.agents.roomscribe.agent import RoomScribeAgent
from backend.agents.roomscribe.config import AgentConfig
from backend.agents.roomscribe.sources import (
    CameraOCRSource,
    MicrophoneSTTSource,
    SourceMessage,
    StdinSTTSource,
    next_message,
)
from backend.agents.scribe import ScribeBatcher
from backend.core.config import MeetingState, Perception


class OCRWorker:
    def __init__(self, agent: RoomScribeAgent) -> None:
        self.agent = agent
        self._in_q: queue.Queue = queue.Queue(maxsize=1)
        self._out_q: queue.Queue[dict] = queue.Queue()
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

    def submit(self, image) -> None:
        # Drop frame if OCR worker is still busy to keep STT responsive.
        if self._in_q.full():
            return
        try:
            self._in_q.put_nowait(image)
        except queue.Full:
            return

    def poll(self) -> dict | None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RoomScribe on-device agent prototype")
    parser.add_argument(
        "--model",
        default=None,
        help="MLX model repo/path",
    )
    parser.add_argument(
        "--camera-interval",
        type=float,
        default=4.0,
        help="Camera OCR capture interval in seconds (e.g., 3 to 5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max generation tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature",
    )
    parser.add_argument(
        "--ignore-people",
        action="store_true",
        help="When camera sees only people and no meeting artifact, suppress person description output.",
    )
    parser.add_argument(
        "--stt-source",
        choices=["both", "mic", "stdin", "camera"],
        default="both",
        help="Input source mode",
    )
    parser.add_argument(
        "--stt-model",
        default="base",
        help="faster-whisper local model size (tiny, base, small, medium, large-v3)",
    )
    parser.add_argument(
        "--stt-language",
        default="en",
        help="Language code for STT (example: en, vi)",
    )
    parser.add_argument(
        "--stt-chunk-seconds",
        type=float,
        default=3.0,
        help="Microphone chunk size in seconds",
    )
    parser.add_argument(
        "--stt-sample-rate",
        type=int,
        default=16000,
        help="Microphone sample rate",
    )
    parser.add_argument(
        "--batch-window-seconds",
        type=float,
        default=5.0,
        help="Seconds to collect valid stt/ocr events before printing one batch.",
    )
    return parser.parse_args()


def handle_message(msg: SourceMessage, agent: RoomScribeAgent | None = None) -> dict | None:
    if msg.kind == "stt":
        return {"type": "stt_raw", "text": str(msg.payload)}
    if msg.kind == "image":
        if agent is None:
            image = msg.payload  # type: ignore[assignment]
            return {"type": "camera_frame", "size": [image.width, image.height]}
        return None
    if msg.kind == "error":
        return {"type": "error", "text": str(msg.payload)}
    if msg.kind == "control" and msg.payload == "shutdown":
        return {"type": "control", "text": "shutdown"}
    return None


def build_analyst_input(perceptions: list[Perception]) -> dict:
    """Build a MeetingState-shaped payload for Agent 3 testing."""
    state = MeetingState()
    for p in perceptions:
        if not p.has_content():
            continue
        if p.is_visual():
            state.whiteboard_content = p.text
            state.timeline.append(
                {"time": p.timestamp[:5], "type": "visual", "content": p.text[:200]}
            )
            state.key_points.append(p.text[:150])
        elif p.is_audio():
            state.timeline.append(
                {"time": p.timestamp[:5], "type": "verbal", "content": p.text[:200]}
            )
            state.key_points.append(p.text[:150])
    state.trim_context()
    return {"type": "analyst_input", "meeting_state": asdict(state)}


def main() -> None:
    args = parse_args()
    cfg = AgentConfig(
        model=args.model,
        camera_interval=args.camera_interval,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        ignore_people=args.ignore_people,
    )
    agent = RoomScribeAgent(cfg) if args.stt_source in {"camera", "both"} else None
    ocr_worker = OCRWorker(agent) if agent is not None else None
    batcher = ScribeBatcher(window_seconds=args.batch_window_seconds)

    msg_q: queue.Queue[SourceMessage] = queue.Queue()
    stt = None
    cam = None

    if args.stt_source in {"mic", "both"}:
        stt = MicrophoneSTTSource(
            msg_q,
            model_size=args.stt_model,
            language=args.stt_language,
            sample_rate=args.stt_sample_rate,
            chunk_seconds=args.stt_chunk_seconds,
        )
    elif args.stt_source == "stdin":
        stt = StdinSTTSource(msg_q)

    if args.stt_source in {"camera", "both"}:
        cam = CameraOCRSource(msg_q, interval_seconds=args.camera_interval)

    if args.stt_source == "both":
        print("RoomScribe combined mode started (microphone + camera). Press Ctrl+C to stop.")
        print(f"RoomScribe model selected: {agent.model_id}")
    elif args.stt_source == "mic":
        print("RoomScribe microphone-only mode started. Press Ctrl+C to stop.")
    elif args.stt_source == "camera":
        print("RoomScribe camera-only mode started. Press Ctrl+C to stop.")
        print(f"RoomScribe model selected: {agent.model_id}")
    else:
        print("RoomScribe stdin-only mode started. Type transcript chunks (`exit` to stop).")
    print(f"Scribe batching active: every {args.batch_window_seconds:.1f}s")
    if stt is not None:
        stt.start()
    if cam is not None:
        cam.start()
    if ocr_worker is not None:
        ocr_worker.start()

    try:
        while True:
            if ocr_worker is not None:
                ocr_out = ocr_worker.poll()
                if ocr_out is not None:
                    if ocr_out.get("type") == "error":
                        print(json.dumps(ocr_out, ensure_ascii=True))
                    else:
                        print(
                            json.dumps(
                                {"type": "agent1_event", "event": ocr_out},
                                ensure_ascii=True,
                            )
                        )
                        batcher.add_event(ocr_out)
            msg = next_message(msg_q, timeout_seconds=0.5)
            if msg is None:
                batch_out, perceptions = batcher.flush_due()
                if batch_out is not None:
                    print(json.dumps(batch_out, ensure_ascii=True))
                    print(json.dumps(build_analyst_input(perceptions), ensure_ascii=True))
                continue
            if msg.kind == "image" and ocr_worker is not None:
                ocr_worker.submit(msg.payload)
                batch_out, perceptions = batcher.flush_due()
                if batch_out is not None:
                    print(json.dumps(batch_out, ensure_ascii=True))
                    print(json.dumps(build_analyst_input(perceptions), ensure_ascii=True))
                continue
            out = handle_message(msg, agent=agent)
            if out is None:
                batch_out, perceptions = batcher.flush_due()
                if batch_out is not None:
                    print(json.dumps(batch_out, ensure_ascii=True))
                    print(json.dumps(build_analyst_input(perceptions), ensure_ascii=True))
                continue
            if out.get("type") == "error":
                print(json.dumps(out, ensure_ascii=True))
            elif out.get("type") == "control":
                print(json.dumps(out, ensure_ascii=True))
            else:
                print(
                    json.dumps(
                        {"type": "agent1_event", "event": out},
                        ensure_ascii=True,
                    )
                )
                batcher.add_event(out)
            if out.get("type") == "control" and out.get("text") == "shutdown":
                break
            batch_out, perceptions = batcher.flush_due()
            if batch_out is not None:
                print(json.dumps(batch_out, ensure_ascii=True))
                print(json.dumps(build_analyst_input(perceptions), ensure_ascii=True))
    except KeyboardInterrupt:
        pass
    finally:
        if cam is not None:
            cam.stop()
        if stt is not None and hasattr(stt, "stop"):
            stt.stop()
        if ocr_worker is not None:
            ocr_worker.stop()


if __name__ == "__main__":
    main()
