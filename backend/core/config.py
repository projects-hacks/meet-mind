"""MeetMind configuration — single source of truth for all settings."""

from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Protocol


# ── Data Contracts (Interface between Agent 1 and Agent 2) ──

@dataclass
class Perception:
    """What Agent 1 (Perceiver) sends to Agent 2. This is the contract."""
    timestamp: str
    event_type: str  # "ocr" | "stt_raw"
    text: str

    def has_content(self) -> bool:
        """Check if this perception has any meaningful content."""
        return bool(self.text.strip())

    def is_visual(self) -> bool:
        return self.event_type == "ocr"

    def is_audio(self) -> bool:
        return self.event_type == "stt_raw"

    def summary(self) -> str:
        """Short summary for logging."""
        if self.is_visual():
            return f"visual({len(self.text)} chars)"
        if self.is_audio():
            return f"audio({len(self.text)} chars)"
        return "empty"

    @staticmethod
    def _infer_visual_content_type(text: str) -> str:
        lowered = text.lower()
        if any(token in lowered for token in ["->", "flow", "diagram", "architecture"]):
            return "diagram"
        if any(token in lowered for token in ["|", "table", "row", "column"]):
            return "table"
        if any(token in lowered for token in ["1.", "2.", "-", "*"]):
            return "list"
        return "freeform"

    @classmethod
    def from_agent1_event(cls, event: dict[str, Any]) -> "Perception | None":
        """Convert current RoomScribe output event into Perception.

        Supports:
        - {"type": "stt_raw", "text": "..."}
        - {"type": "ocr", "timestamp": "...", "text": "..."}
        """
        event_type = str(event.get("type", "")).strip()
        text = str(event.get("text", "")).strip()
        if event_type not in {"stt_raw", "ocr"} or not text:
            return None

        timestamp = str(event.get("timestamp", "")).strip()
        if not timestamp:
            timestamp = datetime.now(tz=timezone.utc).isoformat()

        if event_type == "stt_raw":
            return cls(
                timestamp=timestamp,
                event_type=event_type,
                text=text,
            )

        return cls(
            timestamp=timestamp,
            event_type=event_type,
            text=text,
        )

    def to_event_dict(self) -> dict[str, Any]:
        """Compact serialized form for batching/debug output."""
        visual_content_type = (
            self._infer_visual_content_type(self.text) if self.is_visual() else "empty"
        )
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "text": self.text,
            "visual_content_type": visual_content_type,
            "has_audio": self.is_audio(),
            "has_visual": self.is_visual(),
        }

    def to_scribe_observation(self) -> dict[str, Any]:
        """Canonical observation payload used by Scribe prompt construction."""
        if self.is_audio():
            return {
                "visual_text": [],
                "visual_content_type": "empty",
                "visual_changed": False,
                "audio_transcript": self.text,
            }
        return {
            "visual_text": [self.text],
            "visual_content_type": self._infer_visual_content_type(self.text),
            "visual_changed": True,
            "audio_transcript": "",
        }


@dataclass
class ActionItem:
    owner: str
    task: str
    deadline: str = "unspecified"
    priority: str = "medium"  # high | medium | low

    def key(self) -> str:
        """Unique key for deduplication."""
        return f"{self.owner.lower().strip()}:{self.task.lower().strip()[:50]}"


@dataclass
class Decision:
    decision: str
    alternatives_rejected: list[str] = field(default_factory=list)
    rationale: str = ""

    def key(self) -> str:
        return self.decision.lower().strip()[:60]


@dataclass
class Gap:
    topic: str
    gap_type: str = "unclear_scope"  # no_owner | no_deadline | no_decision | unclear_scope
    suggestion: str = ""

    def key(self) -> str:
        return f"{self.topic.lower().strip()[:40]}:{self.gap_type}"


@dataclass
class ArtifactRequest:
    artifact_type: str  # impl_plan | sales_brief | process_spec | meeting_summary | architecture_doc
    context_summary: str
    domain: str


# ── Context Window Limits ──
MAX_KEY_POINTS = 50       # Trim oldest when exceeded
MAX_TIMELINE_ENTRIES = 100
MAX_PROMPT_KEY_POINTS = 15  # How many key points to include in prompts


@dataclass
class MeetingState:
    """Running state of the meeting — accumulated by the Scribe."""
    topic: str = "Unknown"
    domain: str = "general"  # engineering | sales | operations | product | general
    timeline: list[dict[str, Any]] = field(default_factory=list)
    whiteboard_content: str = ""
    key_points: list[str] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def trim_context(self):
        """Prevent context window overflow by trimming old data."""
        if len(self.key_points) > MAX_KEY_POINTS:
            self.key_points = self.key_points[-MAX_KEY_POINTS:]
        if len(self.timeline) > MAX_TIMELINE_ENTRIES:
            self.timeline = self.timeline[-MAX_TIMELINE_ENTRIES:]

    def has_duplicate_action(self, item: ActionItem) -> bool:
        return any(a.key() == item.key() for a in self.action_items)

    def has_duplicate_decision(self, decision: Decision) -> bool:
        return any(d.key() == decision.key() for d in self.decisions)

    def has_duplicate_gap(self, gap: Gap) -> bool:
        return any(g.key() == gap.key() for g in self.gaps)


# ── Model Configuration ──

@dataclass
class ModelConfig:
    """Which Gemma models to use and how."""
    scribe_model: str = "mlx-community/gemma-3-4b-it-4bit"
    analyst_model: str = "mlx-community/gemma-3-4b-it-4bit"
    architect_model: str = "mlx-community/gemma-3-4b-it-4bit"
    max_tokens: int = 1024
    temperature: float = 0.3
    # Local runtime policy
    air_gapped: bool = False
    allow_remote_models: bool = True
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8765
    sse_heartbeat_s: float = 2.0

    @staticmethod
    def is_loopback_host(host: str) -> bool:
        return host in {"127.0.0.1", "localhost"}

    def validate(self):
        if self.air_gapped and self.allow_remote_models:
            raise ValueError("air_gapped=True requires allow_remote_models=False")
        if self.air_gapped and not self.is_loopback_host(self.dashboard_host):
            raise ValueError("air_gapped mode requires dashboard_host to be localhost/127.0.0.1")
        if self.dashboard_port <= 0 or self.dashboard_port > 65535:
            raise ValueError("dashboard_port must be between 1 and 65535")
        if self.sse_heartbeat_s <= 0:
            raise ValueError("sse_heartbeat_s must be > 0")


# ── LLM Provider Interface (Dependency Inversion) ──

class LLMProvider(Protocol):
    """Abstract interface for language model inference.
    Any backend (MLX, Ollama, API) can implement this."""

    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 1024) -> str:
        ...

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        ...


# ── Tool Definitions for Analyst Agent ──

ANALYST_TOOLS = [
    {"name": "extract_action_item", "description": "A task was assigned to someone with a deadline."},
    {"name": "log_decision", "description": "The team made a firm decision."},
    {"name": "flag_gap", "description": "Something important is unresolved."},
    {"name": "request_artifact", "description": "Enough context to generate a structured document."},
    {"name": "suggest_next_step", "description": "Suggest what the team should discuss or do next."},
    {"name": "provide_insight", "description": "Surface relevant context or a warning the team may have missed."},
    {"name": "continue_observing", "description": "Nothing actionable yet."},
]
