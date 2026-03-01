"""Agent 2: Scribe — Structures perception data into a running meeting log.

Single Responsibility: Takes raw perceptions → maintains structured meeting state.
Depends on LLMProvider abstraction (Dependency Inversion).

Edge cases handled:
- Empty/no-content perceptions (skipped)
- JSON parse failures (state preserved)
- Context window overflow (key points trimmed)
- Fuzzy deduplication of key points
- Whiteboard content validation
"""

import json
import logging
import time
from dataclasses import asdict
from difflib import SequenceMatcher

from backend.core.config import Perception, MeetingState, LLMProvider, MAX_PROMPT_KEY_POINTS

logger = logging.getLogger(__name__)


SCRIBE_SYSTEM_PROMPT = """You are a meeting scribe. Given visual + audio observations, update the meeting log.

Rules:
- Only add NEW information, never repeat existing.
- CRITICAL: If the new observation contains `visual_text`, YOU MUST extract and summarize it into the `whiteboard_content` string.
- Keep all values SHORT (under 20 words each), except for whiteboard which can be longer.
- Respond with ONLY valid JSON, no markdown fences:

{"topic":"short topic","domain":"engineering|sales|operations|product|general","new_timeline_entry":{"time":"HH:MM","type":"visual|verbal|both","content":"brief description"},"whiteboard_content":"summarized text from camera/visual_text","new_key_points":["point1","point2"],"open_questions":["question1"]}"""


def _is_similar(a: str, b: str, threshold: float = 0.7) -> bool:
    """Fuzzy string similarity check for deduplication."""
    if not a or not b:
        return False
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio() > threshold


class Scribe:
    """Processes perceptions and builds a structured meeting log."""

    def __init__(self, llm: LLMProvider):
        self._llm = llm
        self._state = MeetingState()
        self._perception_count = 0
        self._consecutive_failures = 0

    def process(self, perception: Perception) -> MeetingState:
        """Process a new perception and update the meeting state.

        Returns the updated MeetingState. On failure, returns existing state unchanged.
        """
        # Skip empty perceptions
        if not perception.has_content():
            return self._state

        self._perception_count += 1

        prompt = self._build_prompt(perception)

        try:
            result = self._llm.generate_json(prompt, SCRIBE_SYSTEM_PROMPT)
            logger.info(f"Scribe LLM result keys: {list(result.keys()) if result else 'EMPTY'}")
            if result and result.get('new_key_points'):
                logger.info(f"Scribe found {len(result['new_key_points'])} new key points")
        except Exception as e:
            logger.error(f"Scribe LLM call failed: {e}")
            self._consecutive_failures += 1
            return self._state

        if result:
            self._apply_update(result, perception.timestamp)
            self._consecutive_failures = 0
            logger.info(
                f"Scribe state: topic={self._state.topic}, "
                f"key_points={len(self._state.key_points)}, "
                f"timeline={len(self._state.timeline)}"
            )
        else:
            self._consecutive_failures += 1
            logger.warning(
                f"Scribe got empty result (consecutive failures: {self._consecutive_failures})"
            )

        # Prevent context overflow
        self._state.trim_context()
        return self._state

    def _build_prompt(self, perception: Perception) -> str:
        """Build the prompt with current state + new perception.

        Limits state data sent to LLM to prevent context overflow.
        """
        state_summary = {
            "topic": self._state.topic,
            "domain": self._state.domain,
            "key_points_so_far": self._state.key_points[-MAX_PROMPT_KEY_POINTS:],
            "current_whiteboard": self._state.whiteboard_content[:300],
        }

        obs = perception.to_scribe_observation()
        new_observation = {
            "visual_text": obs["visual_text"][:500],  # Give Scribe enough context to read the board/screen
            "visual_content_type": obs["visual_content_type"],
            "visual_changed": obs["visual_changed"],
            "audio_transcript": str(obs["audio_transcript"])[:1000],  # Increased to handle longer audio chunks
        }

        return f"""Current meeting state:
{json.dumps(state_summary, indent=2)}

New observation at {perception.timestamp}:
{json.dumps(new_observation, indent=2)}

Update the meeting log with any new information."""

    def _apply_update(self, update: dict, timestamp: str):
        """Apply the LLM's structured update to our meeting state.

        Validates all fields before applying. Invalid data is silently skipped.
        """
        # Topic update
        topic = update.get("topic")
        if isinstance(topic, str) and topic.strip() and topic.lower() != "unknown":
            self._state.topic = topic.strip()[:100]

        # Domain validation
        domain = update.get("domain", "")
        valid_domains = {"engineering", "sales", "operations", "product", "general"}
        if isinstance(domain, str) and domain.lower() in valid_domains:
            self._state.domain = domain.lower()

        # Whiteboard content
        wb = update.get("whiteboard_content")
        if isinstance(wb, str) and wb.strip():
            self._state.whiteboard_content = wb.strip()[:1000]

        # Timeline entry
        entry = update.get("new_timeline_entry")
        if isinstance(entry, dict) and entry.get("content"):
            entry.setdefault("time", timestamp[:5] if timestamp else "??:??")
            entry["content"] = str(entry["content"])[:200]
            self._state.timeline.append(entry)

        # Key points — fuzzy deduplicate
        for point in update.get("new_key_points", []):
            if not isinstance(point, str) or not point.strip():
                continue
            point = point.strip()[:150]
            # Check for fuzzy duplicates
            if not any(_is_similar(point, existing) for existing in self._state.key_points):
                self._state.key_points.append(point)

    @property
    def state(self) -> MeetingState:
        return self._state

    @property
    def perception_count(self) -> int:
        return self._perception_count

    @property
    def is_healthy(self) -> bool:
        """Returns False if the Scribe has failed too many times in a row."""
        return self._consecutive_failures < 5

    def to_dict(self) -> dict:
        """Serialize current state for dashboard/storage."""
        return asdict(self._state)


class ScribeBatcher:
    """Agent 2 runtime batcher for upstream events.

    Collects valid RoomScribe events and emits a batch every fixed interval.
    """

    def __init__(self, window_seconds: float = 5.0):
        self.window_seconds = window_seconds
        self._batch: list[Perception] = []
        self._window_started = time.monotonic()

    def add_event(self, event: dict) -> None:
        perception = Perception.from_agent1_event(event)
        if perception is None:
            return
        self._batch.append(perception)

    def flush_due(self) -> tuple[dict | None, list[Perception]]:
        now = time.monotonic()
        if now - self._window_started < self.window_seconds:
            return None, []
        if not self._batch:
            self._window_started = now
            return None, []

        flushed = self._batch[:]
        out = {
            "type": "scribe_batch",
            "window_seconds": self.window_seconds,
            "event_count": len(flushed),
            "events": [p.to_event_dict() for p in flushed],
        }
        self._batch = []
        self._window_started = now
        return out, flushed
