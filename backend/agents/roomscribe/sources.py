from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image


@dataclass(slots=True)
class SourceMessage:
    kind: str
    payload: str | Image.Image


class StdinSTTSource:
    def __init__(self, out_q: queue.Queue[SourceMessage]) -> None:
        self.out_q = out_q
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while True:
            try:
                line = input("stt> ").strip()
            except EOFError:
                return
            if not line:
                continue
            if line.lower() in {"quit", "exit"}:
                self.out_q.put(SourceMessage(kind="control", payload="shutdown"))
                return
            self.out_q.put(SourceMessage(kind="stt", payload=line))


class MicrophoneSTTSource:
    def __init__(
        self,
        out_q: queue.Queue[SourceMessage],
        model_size: str = "base",
        language: str = "en",
        sample_rate: int = 16000,
        chunk_seconds: float = 3.0,
    ) -> None:
        self.out_q = out_q
        self.model_size = model_size
        self.language = language
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self._stop = threading.Event()
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        try:
            import sounddevice as sd
            from faster_whisper import WhisperModel
        except Exception as exc:
            self.out_q.put(
                SourceMessage(
                    kind="error",
                    payload=f"Microphone STT dependency error: {exc}",
                )
            )
            return

        try:
            stt_model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        except Exception as exc:
            self.out_q.put(
                SourceMessage(kind="error", payload=f"Failed to load STT model: {exc}")
            )
            return

        target_samples = int(self.sample_rate * self.chunk_seconds)
        buffer = np.array([], dtype=np.float32)

        def on_audio(indata, _frames, _time_info, status) -> None:
            if status:
                return
            mono = np.array(indata[:, 0], dtype=np.float32, copy=True)
            self._audio_q.put(mono)

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=on_audio,
            ):
                while not self._stop.is_set():
                    try:
                        chunk = self._audio_q.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    buffer = np.concatenate((buffer, chunk))
                    if buffer.size < target_samples:
                        continue

                    audio = buffer[:target_samples]
                    buffer = buffer[target_samples:]
                    text = self._transcribe(stt_model, audio)
                    if text:
                        self.out_q.put(SourceMessage(kind="stt", payload=text))
        except Exception as exc:
            self.out_q.put(SourceMessage(kind="error", payload=f"Microphone error: {exc}"))

    def _transcribe(self, stt_model, audio: np.ndarray) -> str:
        try:
            segments, _ = stt_model.transcribe(
                audio,
                language=self.language,
                vad_filter=True,
                beam_size=1,
                condition_on_previous_text=False,
            )
        except Exception as exc:
            self.out_q.put(SourceMessage(kind="error", payload=f"STT decode error: {exc}"))
            return ""
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        return text.strip()


class CameraOCRSource:
    def __init__(
        self,
        out_q: queue.Queue[SourceMessage],
        interval_seconds: float = 4.0,
        camera_index: int = 0,
    ) -> None:
        self.out_q = out_q
        self.interval_seconds = interval_seconds
        self.camera_index = camera_index
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.out_q.put(SourceMessage(kind="error", payload="Camera not available"))
            return
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if ok:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb)
                    self.out_q.put(SourceMessage(kind="image", payload=image))
                time.sleep(self.interval_seconds)
        finally:
            cap.release()


def next_message(
    q: queue.Queue[SourceMessage], timeout_seconds: float = 0.5
) -> Optional[SourceMessage]:
    try:
        return q.get(timeout=timeout_seconds)
    except queue.Empty:
        return None
