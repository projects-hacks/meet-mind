"""Agent 2: Architect — Generates structured documents from meeting context.

Single Responsibility: Takes artifact request + meeting context → generates document.
"""

import logging

from backend.core.config import MeetingState, LLMProvider

logger = logging.getLogger(__name__)


# Domain-specific generation templates
_TEMPLATES = {
    "architecture_doc": """Generate a technical architecture document from this meeting.
Include: ## System Overview, ## Components, ## Data Flow, ## Tech Stack, ## Open Questions.
Format in clean markdown.

Meeting context:
{context}

Key decisions: {decisions}
Action items: {action_items}""",

    "impl_plan": """Generate an implementation plan from this meeting.
Include: ## Objective, ## Tasks (with owners + deadlines as checklist), ## Dependencies, ## Risks.
Format as actionable markdown.

Meeting context:
{context}

Key decisions: {decisions}
Action items: {action_items}""",

    "sales_brief": """Generate a sales follow-up brief from this meeting.
Include: ## Client, ## Deal Summary, ## Next Steps, ## Objections to Address, ## Timeline.
Format for a sales rep to action immediately.

Meeting context:
{context}

Decisions: {decisions}""",

    "process_spec": """Generate a process spec from this meeting.
Include: ## Process Name, ## Steps (ordered), ## Inputs/Outputs, ## Quality Gates, ## Owners.
Format for operational handoff.

Meeting context:
{context}

Decisions: {decisions}
Action items: {action_items}""",

    "meeting_summary": """Generate a meeting summary.
Include: ## Topic, ## Key Decisions, ## Action Items (owner + deadline), ## Open Questions.

Meeting context:
{context}

Decisions: {decisions}
Action items: {action_items}
Gaps: {gaps}""",
}

ARCHITECT_SYSTEM = "You generate clean, structured business documents from meeting discussions. Use markdown formatting. Be concise and actionable."


class Architect:
    """Generates structured artifacts from meeting context.

    Open/Closed: add new artifact types by adding entries to _TEMPLATES.
    """

    def __init__(self, llm: LLMProvider):
        self._llm = llm

    def generate(self, artifact_type: str, state: MeetingState) -> str:
        """Generate a structured document from the meeting state.

        Returns the generated markdown document as a string.
        """
        template = _TEMPLATES.get(artifact_type, _TEMPLATES["meeting_summary"])

        decisions_str = "; ".join(d.decision for d in state.decisions) or "None yet"
        actions_str = "; ".join(
            f"{a.owner}: {a.task} (due: {a.deadline})" for a in state.action_items
        ) or "None yet"
        gaps_str = "; ".join(g.topic for g in state.gaps) or "None"

        prompt = template.format(
            context="\n".join(state.key_points),
            decisions=decisions_str,
            action_items=actions_str,
            gaps=gaps_str,
        )

        result = self._llm.generate(prompt, ARCHITECT_SYSTEM, max_tokens=2048)
        logger.info(f"Generated {artifact_type} ({len(result)} chars)")
        return result

    @staticmethod
    def supported_types() -> list[str]:
        return list(_TEMPLATES.keys())
