"""MeetMind main orchestrator — wires Agent 2 components together.

This is the entry point Agent 1 calls into.
Agent 1 captures perception → sends Perception object here → gets results.

Edge cases handled:
- Model load failure (returns error gracefully)
- Agent failures (state preserved)
- Empty perceptions (skipped)
- Duplicate actions (deduped)
"""

import argparse
import logging
import time
from dataclasses import asdict
from typing import Any, cast

from backend.core.config import Perception, MeetingState, ModelConfig
from backend.core.gemma import get_model
from backend.agents.scribe import Scribe
from backend.agents.analyst import Analyst
from backend.agents.architect import Architect
from backend.core.database import MeetingDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


JsonDict = dict[str, Any]


class MeetMind:
    """Main orchestrator for Agent 2.

    Pipeline: Perception → Scribe → Analyst → Architect → Storage
    Entry point: process_perception()
    """

    def __init__(self, config: ModelConfig | None = None):
        cfg = config or ModelConfig()
        cfg.validate()
        self._config = cfg

        # Initialize models (lazy-loaded on first use)
        scribe_llm = get_model(cfg.scribe_model, allow_remote_models=cfg.allow_remote_models)
        analyst_llm = get_model(cfg.analyst_model, allow_remote_models=cfg.allow_remote_models)
        architect_llm = get_model(cfg.architect_model, allow_remote_models=cfg.allow_remote_models)

        # Initialize agents
        self._scribe = Scribe(scribe_llm)
        self._analyst = Analyst(analyst_llm)
        self._architect = Architect(architect_llm)
        self._db = MeetingDB()
        self._cycle_count = 0
        self._total_time = 0.0

        logger.info("MeetMind Agent 2 initialized.")

    def process_perception(self, perception: Perception) -> JsonDict:
        """Main entry point — process a single perception from Agent 1.

        Args:
            perception: Structured perception data from Agent 1.

        Returns:
            dict with 'state', 'action', 'artifact', 'cycle_time_s', 'error'.
        """
        start = time.time()
        result: JsonDict = {
            "state": None, "action": None, "artifact": None,
            "cycle_time_s": 0, "error": None,
        }

        try:
            # Step 1: Scribe structures the perception
            state = self._scribe.process(perception)
            result["state"] = asdict(state)

            # Step 2: Analyst decides action
            action = self._analyst.decide(state)
            result["action"] = action

            # Step 3: Apply action (with dedup)
            self._analyst.apply_action(action, state)

            # Step 4: Generate artifact if requested
            if action.get("action") == "request_artifact":
                params = cast(JsonDict, action.get("params", {}))
                artifact_type = str(params.get("artifact_type", "meeting_summary"))
                try:
                    content = self._architect.generate(artifact_type, state)
                    result["artifact"] = {"type": artifact_type, "content": content}
                    state.artifacts.append({
                        "type": artifact_type,
                        "status": "completed",
                        "content": content,
                        "source": "analyst_request",
                    })
                except Exception as e:
                    logger.error(f"Artifact generation failed: {e}")
                    result["artifact"] = {"type": artifact_type, "content": "", "error": str(e)}

            # Step 5: Persist
            self._db.log_event(
                perception.timestamp,
                action.get("action", "unknown"),
                action,
            )
            self._db.save_state(state)

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            result["error"] = str(e)

        # Metrics
        elapsed = time.time() - start
        result["cycle_time_s"] = round(elapsed, 2)
        self._cycle_count += 1
        self._total_time += elapsed

        logger.info(
            f"Cycle {self._cycle_count} [{elapsed:.1f}s] "
            f"→ {cast(JsonDict, result.get('action', {})).get('action', '?')} "
            f"| pts:{len(self._scribe.state.key_points)} "
            f"| acts:{len(self._scribe.state.action_items)} "
            f"| decs:{len(self._scribe.state.decisions)} "
            f"| gaps:{len(self._scribe.state.gaps)}"
        )

        return result

    @property
    def state(self) -> MeetingState:
        return self._scribe.state

    def get_dashboard_payload(self) -> JsonDict:
        """Dashboard-ready payload of current meeting state."""
        s = self._scribe.state
        recent_suggestions = [
            t.get("content", "")
            for t in s.timeline
            if t.get("type") == "suggestion"
        ][-10:]
        recent_insights = [
            t.get("content", "")
            for t in s.timeline
            if t.get("type") == "insight"
        ][-10:]
        last_action = self._analyst.action_history[-1] if self._analyst.action_history else None

        return {
            "topic": s.topic,
            "domain": s.domain,
            "key_points": s.key_points,
            "action_items": [asdict(a) for a in s.action_items],
            "decisions": [asdict(d) for d in s.decisions],
            "gaps": [asdict(g) for g in s.gaps],
            "artifacts": s.artifacts,
            "timeline": s.timeline[-20:],
            "whiteboard": s.whiteboard_content,
            "perception_count": self._scribe.perception_count,
            "avg_cycle_time": round(self._total_time / max(self._cycle_count, 1), 2),
            "air_gapped": self._config.air_gapped,
            "last_action": last_action,
            "suggestions": recent_suggestions,
            "insights": recent_insights,
        }

    def get_summary(self) -> JsonDict:
        """Backward-compatible alias for dashboard summary payload."""
        return self.get_dashboard_payload()

    def generate_meeting_summary(self) -> str:
        """On-demand: generate final meeting summary artifact."""
        content = self._architect.generate("meeting_summary", self._scribe.state)
        self._scribe.state.artifacts.append({
            "type": "meeting_summary",
            "status": "completed",
            "content": content,
            "source": "manual",
        })
        self._db.log_event("agent", "artifact_generated", {"type": "meeting_summary"})
        self._db.save_state(self._scribe.state)
        return content

    def get_recent_events(self, event_type: str | None = None, limit: int = 30) -> list[JsonDict]:
        """Return recent persisted events for dashboard/history views."""
        return self._db.get_recent_events(event_type=event_type, limit=limit)

    def reset(self):
        """Clear state for a new meeting session."""
        cfg = self._config
        self._scribe = Scribe(get_model(cfg.scribe_model, allow_remote_models=cfg.allow_remote_models))
        self._analyst = Analyst(get_model(cfg.analyst_model, allow_remote_models=cfg.allow_remote_models))
        self._cycle_count = 0
        self._total_time = 0.0
        self._db.clear()
        logger.info("MeetMind reset for new meeting.")

    @property
    def health(self) -> JsonDict:
        """System health check."""
        return {
            "scribe_healthy": self._scribe.is_healthy,
            "cycles": self._cycle_count,
            "avg_cycle_time": round(self._total_time / max(self._cycle_count, 1), 2),
            "air_gapped": self._config.air_gapped,
        }


def _build_runtime_config(args: argparse.Namespace) -> ModelConfig:
    allow_remote_models = not args.no_remote_models
    if args.air_gapped:
        allow_remote_models = False

    cfg = ModelConfig(
        dashboard_host=args.host,
        dashboard_port=args.port,
        air_gapped=args.air_gapped,
        allow_remote_models=allow_remote_models,
    )
    cfg.validate()
    return cfg


def main():
    parser = argparse.ArgumentParser(description="MeetMind local runtime")
    parser.add_argument("--serve-dashboard", action="store_true", help="Run local dashboard API server")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Dashboard bind port (default: 8765)")
    parser.add_argument("--air-gapped", action="store_true", help="Enable strict local/air-gapped policy")
    parser.add_argument(
        "--no-remote-models",
        action="store_true",
        help="Disallow remote model IDs; require local model paths",
    )
    args = parser.parse_args()

    cfg = _build_runtime_config(args)

    if args.serve_dashboard:
        from backend.dashboard_server import run_dashboard_server

        run_dashboard_server(cfg)
        return

    logger.info("MeetMind initialized. Use --serve-dashboard to start local API + SSE server.")


if __name__ == "__main__":
    main()
