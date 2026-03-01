from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from PIL import Image

from backend.agents.roomscribe.config import AgentConfig
from backend.core.gemma import mlx_inference_lock


@dataclass(slots=True)
class Event:
    kind: str
    text: str
    ts: str


def resolve_model_id(
    user_model: str | None,
    default_model_candidates: tuple[str, ...],
    fallback_model: str,
    load_fn: Callable[[str], tuple[Any, Any]] | None = None,
) -> tuple[str, Any, Any]:
    if load_fn is None:
        from mlx_vlm import load as mlx_load

        load_fn = mlx_load

    attempted: list[str] = []

    if user_model:
        attempted.append(user_model)
        try:
            model, processor = load_fn(user_model)
            return user_model, model, processor
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load explicit model '{user_model}' (strict mode): {exc}"
            ) from exc

    for model_id in default_model_candidates:
        attempted.append(model_id)
        try:
            model, processor = load_fn(model_id)
            return model_id, model, processor
        except Exception:
            continue

    attempted.append(fallback_model)
    try:
        model, processor = load_fn(fallback_model)
        return fallback_model, model, processor
    except Exception as exc:
        attempted_joined = ", ".join(attempted)
        raise RuntimeError(
            "Failed to load any default Gemma models. "
            f"Attempted (in order): {attempted_joined}. Last error: {exc}"
        ) from exc


class RoomScribeAgent:
    def __init__(self, cfg: AgentConfig) -> None:
        self.cfg = cfg
        self.model_id, self.model, self.processor = resolve_model_id(
            user_model=cfg.model,
            default_model_candidates=cfg.default_model_candidates,
            fallback_model=cfg.fallback_model,
        )

    @staticmethod
    def _ts() -> str:
        return datetime.now(tz=timezone.utc).isoformat()

    @staticmethod
    def _to_text(output: Any) -> str:
        if hasattr(output, "text"):
            return str(output.text).strip()
        return str(output).strip()

    @staticmethod
    def _clean_ocr_text(text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"<pad>+", " ", cleaned)
        cleaned = cleaned.replace("<start_of_image>", " ")
        cleaned = cleaned.replace("<end_of_image>", " ")
        cleaned = cleaned.replace("<image>", " ")
        cleaned = re.sub(
            r"^(the most important (thing|element) in the image is|the most prominent element in the image is|the image shows|image description:|a person holding a sign with handwritten text|the sign asks|a person holding a sign with|a person holding|the text says|the handwritten text says|it says)\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        # Strip trailing conversational parts if it leaked
        cleaned = re.sub(r'["\']', '', cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Skip meaningless outputs that are mostly special-token artifacts.
        if not cleaned:
            return ""
        if re.fullmatch(r"[<>/_\-a-zA-Z0-9\s.]*", cleaned) and " " not in cleaned:
            return ""
        return cleaned

    def _build_camera_prompt(self) -> str:
        return (
             "You are a strict, emotionless OCR text scanner. Return direct factual text output only.\n"
             "RULES:\n"
             "1. Read ANY and ALL text visible on screens, whiteboards, paper, or slides.\n"
             "2. Describe any diagrams, charts, or flow arrows.\n"
             "3. IGNORE PEOPLE ENTIRELY. Do not describe the room. Do not describe who is holding a paper.\n\n"
             "CRITICAL CONSTRAINTS:\n"
             "- Never hallucinate. If there is no legible text in the image, reply EXACTLY with: NO_MEETING_CONTENT\n"
             "- Just output the exact text you see. Do not write conversational filler like 'A person holding a sign with handwritten text. The sign asks...'\n"
             "- Keep it short (max 4 lines)\n"
             "- Start directly with nouns/content; no preface\n"
             "- Do not output JSON/Markdown/code blocks\n"
             "Now scan this camera frame and output exactly what the text says."
        )

    def refine_stt(self, raw_text: str) -> Event:
        from mlx_vlm import generate

        prompt = (
            "You clean up an in-meeting transcript chunk. "
            "Fix punctuation and obvious ASR mistakes only. "
            "Do not add new content. Return plain text only.\n\n"
            f"Transcript chunk:\n{raw_text}"
        )
        with mlx_inference_lock:
            cleaned = generate(
                self.model,
                self.processor,
                prompt=prompt,
                max_tokens=self.cfg.max_tokens,
                temp=self.cfg.temperature,
                verbose=False,
            )
        return Event(kind="stt", text=self._to_text(cleaned), ts=self._ts())

    def image_to_text(self, image: Image.Image) -> Event:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        raw_prompt = self._build_camera_prompt()
        prompt = apply_chat_template(
            self.processor,
            self.model.config,
            raw_prompt,
            num_images=1,
        )

        with mlx_inference_lock:
            text = generate(
                self.model,
                self.processor,
                prompt=prompt,
                image=image,
                max_tokens=self.cfg.max_tokens,
                temp=self.cfg.temperature,
                verbose=False,
            )
        cleaned = self._clean_ocr_text(self._to_text(text))
        return Event(kind="ocr", text=cleaned, ts=self._ts())

    def to_json(self, event: Event) -> dict[str, Any]:
        return {"type": event.kind, "timestamp": event.ts, "text": event.text}
