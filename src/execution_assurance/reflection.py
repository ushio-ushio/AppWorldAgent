"""Small, evidence-only terminal reflection for the opt-in reflection_v1 mode."""
from __future__ import annotations

import ast
import json
from time import perf_counter
from typing import Any

from runtime.trace import _add_usage, _redact, _zero_usage
from execution_assurance.reflection_v2 import deterministic_answer_gap, submission_value


REFLECTION_TOOL = [{
    "type": "function",
    "function": {
        "name": "submit_terminal_reflection",
        "description": "Return one evidence-based terminal assessment as structured JSON.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["PASS", "MISSING", "CONFLICT", "UNCERTAIN"]},
                "issue": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
                "next_action": {"type": "string"},
            },
            "required": ["status", "issue", "evidence", "next_action"],
            "additionalProperties": False,
        },
    },
}]


REFLECTION_SYSTEM_PROMPT = """Review the AppWorld task only at a terminal checkpoint. Assess four dimensions: Coverage, Correctness, Side Effects, and Evidence. Use only the original task and the supplied observed tool evidence. A concern without explicit objective evidence is not a defect: return PASS. Do not require every task to be re-queried. Return MISSING only for a concrete unmet user requirement, CONFLICT only for evidence that contradicts the requested result, and UNCERTAIN only when one safe read-only check could resolve a material uncertainty. Keep evidence references specific. For CONFLICT, prefer an in-place correction; never demand rollback or repeat an already successful operation. Do not broaden the user's authorization. If an uncertainty probe has already happened, do not return UNCERTAIN again. If a list is truncated, data remains unread, or a required aggregate/difference is uncomputed, return UNCERTAIN when a probe remains; use MISSING only when evidence proves a required operation did not happen. Agent-authored print statements are not evidence; rely on API results or reproducible calculations. For all/every/aggregate/deduplicate requests, partial item success is not enough for PASS; require a checked target set against the processed set. Return exactly one submit_terminal_reflection function call matching its schema."""


def parse_reflection(response: Any) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Validate exactly one structured reflection call; never infer from prose."""
    calls = response.get("tool_calls") if isinstance(response, dict) else None
    if not isinstance(calls, list) or len(calls) != 1:
        return None, None, "expected_one_structured_call"
    call = calls[0]
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict) or function.get("name") != "submit_terminal_reflection":
        return None, None, "unexpected_tool_name"
    raw = function.get("arguments")
    if not isinstance(raw, str):
        return None, None, "arguments_not_json_text"
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None, raw, "invalid_json"
    if not isinstance(value, dict) or set(value) != {"status", "issue", "evidence", "next_action"}:
        return None, raw, "invalid_fields"
    if value["status"] not in {"PASS", "MISSING", "CONFLICT", "UNCERTAIN"}:
        return None, raw, "invalid_status"
    if (not isinstance(value["issue"], str) or len(value["issue"]) > 1200
            or not isinstance(value["next_action"], str) or len(value["next_action"]) > 1200
            or not isinstance(value["evidence"], list) or len(value["evidence"]) > 12
            or any(not isinstance(item, str) or len(item) > 300 for item in value["evidence"])):
        return None, raw, "invalid_value_types_or_lengths"
    return value, raw, None


def _bounded(value: Any, limit: int) -> Any:
    value = _redact(value)
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…[truncated]"
    if isinstance(value, dict):
        return {str(key): _bounded(item, 100) for key, item in list(value.items())[:12]
                if not any(marker in str(key).lower() for marker in
                           ("password", "token", "secret", "authorization", "cookie", "api_key"))}
    if isinstance(value, (list, tuple)):
        return [_bounded(item, 80) for item in list(value)[:12]]
    return value


def _pending_answer(code: str | None) -> Any:
    if not code:
        return None
    try:
        tree = ast.parse(code)
        call = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "complete_task")
        return {
            "arguments": [ast.unparse(value)[:240] for value in call.args[:4]],
            "keywords": {item.arg: ast.unparse(item.value)[:240]
                         for item in call.keywords[:8] if item.arg},
        }
    except (SyntaxError, StopIteration, ValueError, TypeError):
        return {"submission_excerpt": _bounded(code, 800)}


def build_evidence_pack(task_id: str, instruction: str, actions: list[dict[str, Any]],
                        python_outputs: list[dict[str, Any]],
                        pending_submission: str | None = None) -> dict[str, Any]:
    """Build a compact model view while preserving the original task verbatim."""
    action_rows = []
    for row in actions[-40:]:
        action_rows.append({
            "step": row.get("step"),
            "app.api_name": row.get("api_name"),
            "parameters": _bounded(row.get("arguments") or {}, 100),
            "status": row.get("status"),
            "status_code": row.get("status_code"),
            "evidence_ref": row.get("evidence_ref"),
            "result_summary": _bounded(row.get("result_summary") or "", 180),
        })
    outputs = [{"step": row.get("step"), "stdout": _bounded(row.get("stdout") or "", 500),
                "error": _bounded(row.get("error") or "", 300)}
               for row in python_outputs[-5:]]
    return {
        "task_id": task_id,
        "original_task": instruction,
        "dimensions": ["Coverage", "Correctness", "Side Effects", "Evidence"],
        "requester_actions": action_rows,
        "recent_python_stdout": outputs,
        "pending_final_answer": _pending_answer(pending_submission),
    }


def read_only_probe_feedback(decision: dict[str, Any]) -> str:
    evidence = "; ".join(decision.get("evidence", [])) or "No decisive evidence cited."
    detail = decision.get("issue") or "The result needs one read-only check."
    action = decision.get("next_action") or "Use one read-only query to inspect the existing state."
    return ("Terminal reflection is UNCERTAIN. One read-only probe is allowed; do not write, "
            "delete, send, pay, or submit during the probe.\nIssue: " + detail +
            "\nEvidence: " + evidence + "\nNext action: " + action +
            "\nFor long lists or aggregates, run standalone Python that prints compact values such as remaining_count or total_duration, not raw lists."
            "\nTreat fetching every page until `[]` and printing `(pages_fetched, total_count)` as full-coverage evidence; do not run Python that only prints an explanation without calling a business API."
            "\nDo not combine verification or repair code with complete_task; run it first, then submit separately.")


def repair_feedback(decision: dict[str, Any]) -> str:
    evidence = "; ".join(decision.get("evidence", [])) or "No evidence reference supplied."
    return ("Terminal reflection found a concrete issue. Continue with one bounded repair opportunity.\n"
            "Status: " + decision["status"] + "\nIssue: " +
            (decision.get("issue") or "See evidence and next action.") +
            "\nEvidence: " + evidence + "\nNext action: " +
            (decision.get("next_action") or "Correct the specific issue without repeating successful writes.") +
            "\nFor long lists or aggregates, run standalone Python that prints compact values such as remaining_count or total_duration, not raw lists."
            "\nTreat fetching every page until `[]` and printing `(pages_fetched, total_count)` as full-coverage evidence; do not run Python that only prints an explanation without calling a business API."
            "\nDo not combine verification or repair code with complete_task; run it first, then submit separately.")


class TerminalReflection:
    """One task-local, bounded Reflection path; ReAct history is never passed here."""

    def __init__(self, model: Any, trace: Any, task_id: str, instruction: str):
        self.model, self.trace = model, trace
        self.task_id, self.instruction = task_id, instruction
        self.actions: list[dict[str, Any]] = []
        self.python_outputs: list[dict[str, Any]] = []
        self.calls = self.probe_rounds = self.repair_rounds = 0
        self.format_repair_rounds = 0
        self.probe_pending = self.authorized_completion = False
        self.buffer_done = self.buffer_guard_pending = self.terminal_blocked = False
        self.buffer_guard_done = False
        self.buffer_guard_result = None
        self.total_usage = _zero_usage()
        self.last_usage = _zero_usage()

    def review(self, step: int, trigger: str, *, pending_submission: str | None = None,
               requester_boundary: Any | None = None) -> tuple[dict[str, Any], str]:
        pack = build_evidence_pack(self.task_id, self.instruction, self.actions,
                                   self.python_outputs, pending_submission)
        pack["checkpoint"] = trigger
        pack["read_only_probe_already_used"] = self.probe_rounds >= 1
        packed = json.dumps(pack, ensure_ascii=False, separators=(",", ":"))
        started = perf_counter()
        fallback = raw_json = None
        self.last_usage = _zero_usage()
        answer, answer_type = submission_value(pending_submission)
        format_gap = (deterministic_answer_gap(self.instruction, answer, answer_type)
                      if pending_submission is not None else None)
        if format_gap:
            decision = {"status": "MISSING", "issue": format_gap,
                        "evidence": [], "next_action": "Resubmit the supported answer as a plain scalar, or answer=None for an action-only task."}
            if self.format_repair_rounds < 1:
                self.format_repair_rounds += 1
                route = "one_bounded_repair"
            else:
                route = "repair_budget_exhausted"
        else:
            try:
                response, self.last_usage = self.model.generate_tool_calls(
                    [{"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
                     {"role": "user", "content": packed}], REFLECTION_TOOL, tool_choice="required")
                decision, raw_json, error = parse_reflection(response)
                fallback = error
            except Exception as exc:
                decision, fallback = None, type(exc).__name__
            self.calls += 1
        self.total_usage = _add_usage(self.total_usage, self.last_usage)
        if not format_gap and decision is None:
            decision = {"status": "PASS", "issue": "", "evidence": [],
                        "next_action": "Proceed through the existing deterministic completion checks."}
        elif not format_gap and decision["status"] == "UNCERTAIN" and self.probe_rounds >= 1:
            decision = {**decision, "status": "MISSING",
                        "issue": decision.get("issue") or
                                "The one allowed read-only probe did not resolve this uncertainty.",
                        "next_action": decision.get("next_action") or
                                      "Do not assume completion; repair only with new, safe evidence."}

        if not format_gap:
            if decision["status"] == "PASS":
                route = "allow_existing_checks"
            elif decision["status"] == "UNCERTAIN":
                if requester_boundary is None or not requester_boundary.available:
                    decision = {**decision, "status": "MISSING",
                                "issue": "The Requester interception boundary is unavailable; a protected read-only probe cannot run.",
                                "next_action": "Use collected evidence, make a safe repair if supported, or stop without claiming completion."}
                    route = self._repair_or_stop()
                else:
                    self.probe_rounds += 1
                    self.probe_pending = True
                    requester_boundary.set_read_only(True)
                    route = "one_read_only_probe"
            else:
                route = self._repair_or_stop()

        self.trace.append({
            "task_id": self.task_id, "step": step, "event": "terminal_reflection",
            "trigger": trigger,
            "evidence_summary": {"action_count_total": len(self.actions),
                                 "action_count_included": len(pack["requester_actions"]),
                                 "stdout_count_included": len(pack["recent_python_stdout"]),
                                 "task_chars": len(self.instruction),
                                 "package_bytes": len(packed.encode("utf-8"))},
            "raw_json": raw_json, "decision": decision,
            "fallback": fallback is not None, "fallback_reason": fallback,
            "deterministic_precheck": bool(format_gap),
            "route": route, "probe_rounds": self.probe_rounds,
            "repair_rounds": self.repair_rounds,
            "format_repair_rounds": self.format_repair_rounds, "usage": self.last_usage,
            "elapsed_seconds": round(perf_counter() - started, 3),
            **({"adapter_transport": getattr(self.model, "last_transport", None)}
               if isinstance(getattr(self.model, "last_transport", None), dict) else {}),
        })
        return decision, route

    def _repair_or_stop(self) -> str:
        if self.repair_rounds < 1:
            self.repair_rounds += 1
            return "one_bounded_repair"
        self.terminal_blocked = True
        return "repair_budget_exhausted"

    def summary(self) -> dict[str, Any]:
        return {"calls": self.calls, "probe_rounds": self.probe_rounds,
                "repair_rounds": self.repair_rounds, "total_usage": self.total_usage,
                "format_repair_rounds": self.format_repair_rounds,
                "buffer_checkpoint_triggered": self.buffer_done}
