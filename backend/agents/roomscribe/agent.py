from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from PIL import Image

from backend.agents.roomscribe.config import AgentConfig


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
            r"^(the most important (thing|element) in the image is|the most prominent element in the image is|the image shows|image description:)\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Skip meaningless outputs that are mostly special-token artifacts.
        if not cleaned:
            return ""
        if re.fullmatch(r"[<>/_\-a-zA-Z0-9\s.]*", cleaned) and " " not in cleaned:
            return ""
        return cleaned

    def _build_camera_prompt(self) -> str:
        people_rule = (
            "If the scene is mainly people and no meeting artifact is visible, "
            "respond with exactly: NO_MEETING_CONTENT."
            if self.cfg.ignore_people
            else (
                "If the scene is mainly people and no meeting artifact is visible, "
                "briefly describe the person in one short line."
            )
        )
        return (
            "You are a meeting-vision extractor. Return direct factual output only.\n"
            "Priority order:\n"
            "1) Whiteboard/paper/slides content\n"
            "2) Drawings/diagrams/arrows/flows and what they mean\n"
            "3) People only if no meeting artifact is visible\n\n"
            f"{people_rule}\n\n"
            "Output rules:\n"
            "- Plain text only\n"
            "- Keep it short (max 4 lines)\n"
            "- Start directly with nouns/content; no preface\n"
            "- Never write: 'the image shows', 'the most important', 'the most prominent'\n"
            "- Do not output JSON/Markdown/code blocks\n"
            "- Ignore camera noise, token artifacts, and UI junk\n\n"
            "Example 1:\n"
            "Input: whiteboard with text 'Q2 roadmap', arrows from API -> Mobile -> Billing\n"
            "Output: Q2 roadmap. Flow: API -> Mobile -> Billing.\n\n"
            "Example 2:\n"
            "Input: person face only, no board or paper\n"
            "Output: Person with glasses, facing camera.\n\n"
            "Example 3:\n"
            "Input: phone screen with Gmail inbox\n"
            "Output: Gmail inbox, unread emails list visible.\n\n"
            "Now analyze this frame."
        )

    def refine_stt(self, raw_text: str) -> Event:
        from mlx_vlm import generate

        prompt = (
            "You clean up an in-meeting transcript chunk. "
            "Fix punctuation and obvious ASR mistakes only. "
            "Do not add new content. Return plain text only.\n\n"
            f"Transcript chunk:\n{raw_text}"
        )
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
