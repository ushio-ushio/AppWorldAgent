"""Independent, evidence-led terminal audit for the opt-in reflection_v2 mode."""
from __future__ import annotations

import ast
import json
import re
from time import perf_counter
from typing import Any

from agentlab.runtime.trace import _add_usage, _redact, _zero_usage


AUDITOR_SYSTEM_PROMPT = """You are an independent AppWorld completion auditor. Use only the original instruction, the objective Requester ledger, and the proposed answer. Do not infer facts from executor reasoning, code, search history, or claims. Process in order: original instruction -> atomic conditions (people/scope, time boundaries, exclusions, prerequisites such as backup-before-delete) -> compare each condition with objective ledger evidence -> verdict. Return PASS only when every applicable condition is supported; FAIL for a concrete unmet or conflicting condition; UNCERTAIN when the ledger cannot establish a material condition. For action-only tasks with verified business state require answer=None; never accept explanatory prose. For questions return only the requested scalar/entity value. Return one submit_terminal_audit call with verdict, evidence_gaps, and next_verification. Keep the structured JSON concise (under 1500 characters); the executor receives a shorter extracted summary."""
AUDIT_TOOL = [{"type": "function", "function": {
    "name": "submit_terminal_audit", "description": "Return a concise evidence-based terminal verdict.",
    "parameters": {"type": "object", "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "FAIL", "UNCERTAIN"]},
        "evidence_gaps": {"type": "string", "maxLength": 1100},
        "next_verification": {"type": "string", "maxLength": 500}},
        "required": ["verdict", "evidence_gaps", "next_verification"], "additionalProperties": False}}
}]
_REFUSAL = re.compile(r"unable to verify|cannot verify|can't verify|missing\s+(?:the\s+)?pdf|could not verify|not enough information", re.I)
_NUMERIC_ASK = re.compile(r"how much|what amount|total|sum|amount|cost|price|duration|number of|count|多少|金额|总额|合计|数量|时长", re.I)
_UNIT = re.compile(r"\$|\b(?:USD|dollars?|minutes?|hours?|seconds?|km|miles?|items?|songs?|records?)\b", re.I)


def submission_value(code: str | None) -> tuple[Any, str]:
    """Read the submitted answer expression without evaluating executor code."""
    if not code:
        return None, "NoneType"
    try:
        tree = ast.parse(code)
        call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "complete_task")
        value = next((kw.value for kw in call.keywords if kw.arg == "answer"), None)
        if value is None:
            return None, "NoneType"
        if isinstance(value, ast.Constant):
            return value.value, type(value.value).__name__
        return ast.unparse(value)[:500], "expression"
    except (SyntaxError, StopIteration, TypeError, ValueError):
        return "<unparsed submission>", "unknown"


def deterministic_answer_gap(instruction: str, answer: Any, answer_type: str) -> str | None:
    """Reject only clear refusal prose or obvious non-scalar numeric answers."""
    text = str(answer) if answer is not None else ""
    if _REFUSAL.search(text):
        return "The answer is a refusal/uncertainty statement, not the requested result. Submit the best supported value."
    if answer_type in {"dict", "list", "tuple", "set"}:
        return "The submitted answer is not a scalar/entity value. Extract the requested value."
    if _NUMERIC_ASK.search(instruction) and isinstance(answer, str):
        if _UNIT.search(answer) or (len(answer) > 64 and len(answer.split()) > 8):
            return "The task asks for a numeric value; submit only the scalar without units or explanation."
    return None


def _safe(value: Any, limit: int = 180) -> Any:
    value = _redact(value)
    if isinstance(value, str):
        return value[:limit] + ("…" if len(value) > limit else "")
    if isinstance(value, dict):
        return {str(k): _safe(v, 80) for k, v in list(value.items())[:10]
                if not any(s in str(k).lower() for s in ("password", "token", "secret", "authorization", "cookie"))}
    if isinstance(value, (list, tuple)):
        return [_safe(v, 60) for v in value]
    return value


def build_audit_context(instruction: str, ledger: list[dict[str, Any]], code: str | None) -> list[dict[str, str]]:
    """Build exactly four inputs; executor messages and search history are never accepted."""
    answer, answer_type = submission_value(code)
    compact = {"requester_ledger": _safe(ledger, 12000),
               "proposed_answer": {"python_type": answer_type, "content": _safe(answer, 500)}}
    return [{"role": "system", "content": AUDITOR_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"original_instruction": instruction,
                **compact}, ensure_ascii=False, separators=(",", ":"))}]


def _parse(response: Any) -> tuple[dict[str, str] | None, str | None]:
    calls = response.get("tool_calls") if isinstance(response, dict) else None
    try:
        fn = calls[0]["function"]
        data = json.loads(fn["arguments"])
        if (len(calls) != 1 or fn.get("name") != "submit_terminal_audit"
                or set(data) != {"verdict", "evidence_gaps", "next_verification"}
                or data["verdict"] not in {"PASS", "FAIL", "UNCERTAIN"}
                or any(not isinstance(data[k], str) for k in data)):
            return None, "invalid_audit_contract"
        if (len(raw := json.dumps(data, ensure_ascii=False, separators=(",", ":"))) > 1500
                or len(data["evidence_gaps"]) > 1100 or len(data["next_verification"]) > 500):
            return None, "audit_contract_too_long"
        return data, None
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None, "invalid_audit_response"


def audit_feedback(decision: dict[str, Any], *, max_chars: int = 480) -> str:
    """Only structured concise audit fields flow back to the ReAct executor."""
    payload = {"verdict": decision.get("verdict", "UNCERTAIN"),
               "evidence_gaps": str(decision.get("evidence_gaps", ""))[:360],
               "next_verification": str(decision.get("next_verification", ""))[:160]}
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    while len(text) > max_chars:
        key = "evidence_gaps" if len(payload["evidence_gaps"]) >= len(payload["next_verification"]) else "next_verification"
        payload[key] = payload[key][:-max(1, min(20, len(payload[key])))]
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return text


class TerminalAudit:
    """Bounded audit interface compatible with the existing submission router."""
    def __init__(self, model: Any, trace: Any, task_id: str, instruction: str):
        self.model, self.trace, self.task_id, self.instruction = model, trace, task_id, instruction
        self.calls = self.repair_rounds = self.format_repair_rounds = self.probe_rounds = 0
        self.probe_pending = self.authorized_completion = self.terminal_blocked = False
        self.buffer_done = self.buffer_guard_pending = self.buffer_guard_done = False
        self.buffer_guard_result = None
        self.total_usage, self.last_usage = _zero_usage(), _zero_usage()

    def review(self, step: int, trigger: str, *, pending_submission: str | None = None,
               requester_boundary: Any | None = None, ledger: list[dict[str, Any]] | None = None):
        answer, answer_type = submission_value(pending_submission)
        gap = deterministic_answer_gap(self.instruction, answer, answer_type)
        packed = build_audit_context(self.instruction, ledger or [], pending_submission)
        raw, fallback = None, None
        started = perf_counter()
        self.last_usage = _zero_usage()
        if gap:
            verdict = {"verdict": "FAIL", "evidence_gaps": gap[:180],
                       "next_verification": "Return the supported scalar/entity answer."}
        else:
            try:
                response, self.last_usage = self.model.generate_tool_calls(packed, AUDIT_TOOL,
                                                                           tool_choice="required")
                verdict, fallback = _parse(response)
                raw = response.get("tool_calls") if isinstance(response, dict) else None
            except Exception as exc:
                verdict, fallback = None, type(exc).__name__
            if verdict is None:
                verdict = {"verdict": "UNCERTAIN",
                           "evidence_gaps": "Independent audit did not return a valid verdict.",
                           "next_verification": "Check the objective ledger and resubmit once."}
        self.calls += 1
        self.total_usage = _add_usage(self.total_usage, self.last_usage)
        status = {"PASS": "PASS", "FAIL": "MISSING", "UNCERTAIN": "UNCERTAIN"}[verdict["verdict"]]
        decision = {"status": status, "issue": verdict["evidence_gaps"], "evidence": [],
                    "next_action": verdict["next_verification"], **verdict}
        if gap:
            if self.format_repair_rounds < 1:
                self.format_repair_rounds += 1
                route = "one_bounded_repair"
            else:
                route = "repair_budget_exhausted"
        elif status == "PASS":
            route = "allow_existing_checks"
        elif self.repair_rounds < 1:
            self.repair_rounds += 1
            route = "one_bounded_repair"
        else:
            route = "repair_budget_exhausted"
        self.trace.append({"task_id": self.task_id, "step": step, "event": "terminal_audit",
            "trigger": trigger, "input_items": ["auditor_l0", "original_instruction", "requester_ledger", "pending_answer"],
            "ledger_count": len(ledger or []), "answer_type": answer_type,
            "deterministic_precheck": bool(gap),
            "answer_chars": len(str(answer)), "raw_response": raw,
            "decision": verdict, "fallback_reason": fallback, "route": route,
            "usage": self.last_usage, "elapsed_seconds": round(perf_counter() - started, 3)})
        return decision, route

    def summary(self) -> dict[str, Any]:
        return {"calls": self.calls, "repair_rounds": self.repair_rounds,
                "format_repair_rounds": self.format_repair_rounds,
                "probe_rounds": 0, "total_usage": self.total_usage}
