"""One-shot, advisory plan for the existing ReAct executor."""
from __future__ import annotations

import json
from typing import Any


PLAN_TOOL = [{
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": "Submit a concise high-level execution plan. Do not select concrete APIs or claim business facts.",
        "parameters": {
            "type": "object",
            "properties": {
                "phases": {"type": "array", "minItems": 1, "maxItems": 8,
                           "items": {"type": "string"}},
                "constraints": {"type": "array", "items": {"type": "string"}},
                "completion_checks": {"type": "array", "items": {"type": "string"}},
                "goals": {"type": "array", "maxItems": 12, "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "description": {"type": "string"},
                        "source_requirement": {"type": "string"},
                        "constraints": {"type": "object", "additionalProperties": True},
                        "dependencies": {"type": "array", "items": {"type": "string"}},
                        "completion_criteria": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["id", "description", "source_requirement", "constraints",
                                 "dependencies", "completion_criteria"],
                    "additionalProperties": False,
                }},
                "destructive_preconditions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["phases", "constraints", "completion_checks"],
            "additionalProperties": False,
        },
    },
}]


def one_shot_plan(model: Any, instruction: str) -> tuple[dict[str, list[str]] | None, dict[str, Any], str | None]:
    """Call the main model once. An invalid plan safely falls back to ordinary ReAct."""
    try:
        system_prompt = (
            "Create a short advisory plan for an AppWorld task. Include major phases, "
            "important user constraints, completion checks, and when possible a small list "
            "of atomic goals. Each goal must cite an exact substring of the user task in "
            "source_requirement; constraints must be copied from explicit user requirements, "
            "not inferred facts. Give dependencies only when the task explicitly requires an "
            "order. When the task explicitly requires preserving, exporting, backing up, or "
            "collecting data before an irreversible/destructive action, split that prerequisite "
            "into its own Goal, make the dependency explicit, and list that Goal ID in "
            "destructive_preconditions. Do not infer a prerequisite or ordering when the task "
            "does not state one. Do not name or guess APIs, credentials, account IDs, or facts. "
            "Do not declare any goal complete. Submit one tool call only. The original task and "
            "actual tool results always take precedence."
        )
        response, usage = model.generate_tool_calls([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ], PLAN_TOOL, tool_choice="required")
        calls = response.get("tool_calls") or []
        if len(calls) != 1 or calls[0].get("function", {}).get("name") != "submit_plan":
            return None, usage, "invalid_plan_action"
        payload = json.loads(calls[0]["function"]["arguments"])
        allowed = {"phases", "constraints", "completion_checks", "goals", "destructive_preconditions"}
        if (not isinstance(payload, dict) or
                not {"phases", "constraints", "completion_checks"} <= set(payload) or
                set(payload) - allowed):
            return None, usage, "invalid_plan_fields"
        for key in ("phases", "constraints", "completion_checks"):
            values = payload[key]
            if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() or len(v) > 400 for v in values):
                return None, usage, "invalid_plan_values"
        if not 1 <= len(payload["phases"]) <= 8:
            return None, usage, "invalid_plan_phases"
        if "goals" in payload and not isinstance(payload["goals"], list):
            return None, usage, "invalid_plan_goals"
        if "destructive_preconditions" in payload and not isinstance(payload["destructive_preconditions"], list):
            return None, usage, "invalid_destructive_preconditions"
        return payload, usage, None
    except Exception as exc:
        # Account authorization and transport failures are infrastructure errors.
        # Replaying the same request through the executor would hide the outage.
        if any(cls.__name__ == "SIWCError" for cls in type(exc).__mro__):
            raise
        return None, {}, type(exc).__name__
