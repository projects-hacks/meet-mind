"""Agent 2: Analyst — Decides autonomous actions using Gemma 3 function calling.

Uses Gemma 3's native function calling capability.
For production: swap with fine-tuned FunctionGemma for faster, more accurate decisions.

Edge cases handled:
- Bare function name output (no JSON)
- FunctionGemma special token format
- Duplicate action prevention
- Terse / malformed model output
- Parameter inference from meeting state
"""

import json
import re
import logging
from typing import Any

from backend.core.config import (
    MeetingState, ActionItem, Decision, Gap,
    ANALYST_TOOLS, LLMProvider, MAX_PROMPT_KEY_POINTS,
)

logger = logging.getLogger(__name__)


JsonDict = dict[str, Any]


ANALYST_SYSTEM = """You are a proactive, highly intelligent meeting assistant and analyst. Based on the meeting state, act like a collaborative team member and decide the single most helpful action to unblock or guide the team.

You MUST respond with valid JSON in this exact format:
{"action":"function_name","params":{...},"reasoning":"why"}

Available actions (PRIORITIZE PROVIDING INSIGHTS AND SUGGESTIONS OVER FLAGGING GAPS):
- provide_insight: {"insight":"Provide detailed, actionable advice, context, or solutions to help the users solve the problem they are currently discussing","category":"architecture|strategy|risk|efficiency"}
- suggest_next_step: {"suggestion":"Detailed and helpful suggestion on what the team should do or discuss next","reason":"why"}
- extract_action_item: {"owner":"name","task":"what","deadline":"when","priority":"high|medium|low"}
- log_decision: {"decision":"what was decided","alternatives_rejected":["alt1"],"rationale":"why"}
- flag_gap: {"topic":"what","gap_type":"no_owner|no_deadline|no_decision|unclear_scope","suggestion":"fix"}
- request_artifact: {"artifact_type":"impl_plan|sales_brief|process_spec","context_summary":"key context","domain":"domain"}
- continue_observing: {"reason":"why no action"}

CRITICAL CONSTRAINT: DO NOT use `flag_gap` for general conversation points. Only use `flag_gap` if the team explicitly agreed to do something but forgot to assign an owner or deadline. If the team is brainstorming or discussing a topic, YOU MUST use `provide_insight` or `suggest_next_step`. Act as a helpful meeting participant. Provide detailed, thoughtful insights to genuinely help the team!


Respond with ONLY the JSON object."""

VALID_ACTIONS = {t["name"] for t in ANALYST_TOOLS}


def _parse_function_call(raw: str) -> JsonDict:
    """Parse a function call response from Gemma 3.

    Handles:
    1. FunctionGemma: <start_function_call>call:func{param:val}<end_function_call>
    2. Gemma 3 native JSON: {"name": "func", "arguments": {...}}
    3. Our JSON format: {"action": "func", "params": {...}}
    4. Bare function name (just "log_decision")
    """
    if not raw or not raw.strip():
        return _fallback("Empty model output")

    text = raw.strip()

    # Format 1: FunctionGemma special tokens
    fc_match = re.search(r'call:(\w+)\{(.+?)\}', text)
    if fc_match:
        func_name = fc_match.group(1)
        params_str = fc_match.group(2)
        params = {}
        for pair in re.findall(r'(\w+):<escape>(.*?)<escape>', params_str):
            params[pair[0]] = pair[1]
        if func_name in VALID_ACTIONS:
            return {"action": func_name, "params": params, "reasoning": "FunctionGemma call"}

    # Format 2 & 3: JSON extraction
    # Strip markdown fences
    clean = text
    if "```" in clean:
        clean = re.sub(r'```\w*\n?', '', clean).strip()

    # Try direct JSON parse
    try:
        data = json.loads(clean)
        return _normalize_json(data)
    except json.JSONDecodeError:
        pass

    # Find JSON object anywhere in text
    start = clean.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(clean)):
            if clean[i] == "{":
                depth += 1
            elif clean[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(clean[start:i + 1])
                        return _normalize_json(data)
                    except json.JSONDecodeError:
                        break

    # Format 4: Bare function name
    text_lower = text.lower().replace(" ", "_").replace("-", "_")
    for name in VALID_ACTIONS:
        if name in text_lower:
            logger.info(f"Matched bare function name: {name}")
            return {"action": name, "params": {}, "reasoning": "Bare function name"}

    return _fallback(f"Could not parse: {text[:150]}")


def _normalize_json(data: JsonDict) -> JsonDict:
    """Normalize different JSON formats to our standard {"action", "params", "reasoning"}."""
    # Format: {"name": "func", "arguments": {...}}
    if "name" in data and data["name"] in VALID_ACTIONS:
        return {
            "action": data["name"],
            "params": data.get("arguments", data.get("parameters", {})),
            "reasoning": data.get("reasoning", ""),
        }
    # Format: {"action": "func", "params": {...}}
    if "action" in data and data["action"] in VALID_ACTIONS:
        return {
            "action": data["action"],
            "params": data.get("params", data.get("parameters", {})),
            "reasoning": data.get("reasoning", ""),
        }
    return _fallback(f"Valid JSON but unrecognized format: {list(data.keys())}")


def _fallback(reason: str) -> JsonDict:
    logger.warning(f"Analyst fallback: {reason}")
    return {"action": "continue_observing", "params": {"reason": reason}, "reasoning": "fallback"}


class Analyst:
    """Decides autonomous actions based on meeting state."""

    def __init__(self, llm: LLMProvider):
        self._llm = llm
        self._action_history: list[JsonDict] = []
        self._consecutive_observe = 0

    def decide(self, state: MeetingState) -> JsonDict:
        """Analyze meeting state and return an action decision."""
        # Let the Analyst run even with minimal state — it can still
        # suggest_next_step or provide_insight from timeline/whiteboard data
        if not state.key_points and not state.timeline and not state.whiteboard_content:
            return _fallback("No meeting content captured yet.")

        prompt = self._build_prompt(state)

        try:
            raw = self._llm.generate(prompt, ANALYST_SYSTEM, max_tokens=256)
            logger.info(f"Analyst raw output: {raw[:200]}")
        except Exception as e:
            logger.error(f"Analyst LLM call failed: {e}")
            return _fallback(f"LLM error: {e}")

        result = _parse_function_call(raw)

        # Track consecutive observations
        if result["action"] == "continue_observing":
            self._consecutive_observe += 1
        else:
            self._consecutive_observe = 0

        self._action_history.append(result)
        return result

    def _build_prompt(self, state: MeetingState) -> str:
        recent = [a['action'] for a in self._action_history[-5:]]
        return f"""Meeting: {state.topic} | Domain: {state.domain}
Key points: {'; '.join(state.key_points[-MAX_PROMPT_KEY_POINTS:])}
Whiteboard: {state.whiteboard_content[:300]}
Recent timeline: {'; '.join(e.get('content','') for e in state.timeline[-5:])}
Tracked: {len(state.action_items)} actions, {len(state.decisions)} decisions, {len(state.gaps)} gaps
Recent agent actions: {', '.join(recent) if recent else 'none'}
Consecutive observations without action: {self._consecutive_observe}

What is the single most important action to take now? If the team is discussing a problem, use `provide_insight` with a highly detailed suggestion on how to solve it."""

    def apply_action(self, action: JsonDict, state: MeetingState) -> MeetingState:
        """Apply an action to the meeting state. Deduplicates before adding."""
        name = action.get("action", "")
        params = action.get("params", {})

        if name == "extract_action_item":
            item = ActionItem(
                owner=str(params.get("owner", "Unassigned")).strip(),
                task=str(params.get("task", "")).strip(),
                deadline=str(params.get("deadline", "unspecified")).strip(),
                priority=str(params.get("priority", "medium")).strip().lower(),
            )
            if item.task and not state.has_duplicate_action(item):
                state.action_items.append(item)
                logger.info(f"Action item added: {item.owner} → {item.task}")
            elif item.task:
                logger.info(f"Duplicate action item skipped: {item.task}")

        elif name == "log_decision":
            dec = Decision(
                decision=str(params.get("decision", "")).strip(),
                alternatives_rejected=params.get("alternatives_rejected", []),
                rationale=str(params.get("rationale", "")).strip(),
            )
            if dec.decision and not state.has_duplicate_decision(dec):
                state.decisions.append(dec)
                logger.info(f"Decision logged: {dec.decision}")

        elif name == "flag_gap":
            gap = Gap(
                topic=str(params.get("topic", "")).strip(),
                gap_type=str(params.get("gap_type", "unclear_scope")).strip(),
                suggestion=str(params.get("suggestion", "")).strip(),
            )
            if gap.topic and not state.has_duplicate_gap(gap):
                state.gaps.append(gap)
                logger.info(f"Gap flagged: {gap.topic}")

        elif name == "request_artifact":
            state.artifacts.append({
                "type": params.get("artifact_type", "meeting_summary"),
                "context": params.get("context_summary", ""),
                "domain": params.get("domain", state.domain),
                "status": "pending",
            })
            logger.info(f"Artifact requested: {params.get('artifact_type')}")

        elif name == "suggest_next_step":
            # Treated as a special timeline entry
            state.timeline.append({
                "time": "agent",
                "type": "suggestion",
                "content": f"💡 {params.get('suggestion', '')}",
            })

        elif name == "provide_insight":
            state.timeline.append({
                "time": "agent",
                "type": "insight",
                "content": f"🔍 {params.get('insight', '')}",
            })

        return state

    @property
    def action_history(self) -> list[JsonDict]:
        return self._action_history
