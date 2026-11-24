"""Bounded, evidence-based completion check for plan-guided ReAct."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any


def batch_set_difference(text: str) -> dict[str, Any] | None:
    """Check a model program's explicit post-readback set receipt, if present.

    The receipt is evidence from the AppWorld shell output, not a trusted claim
    about hidden task correctness. Callers also require a recorded read request.
    """
    for line in text.splitlines():
        if not line.startswith("AGENTLAB_BATCH_CHECK="):
            continue
        try:
            data = json.loads(line.split("=", 1)[1])
            fields = (data["expected"], data["observed"], data["failed"])
            if any(not isinstance(items, list) or len(items) > 1000 or
                   any(not isinstance(item, (str, int)) for item in items) for items in fields):
                return None
            expected, observed, failed = (set(map(str, items)) for items in fields)
            return {"expected_count": len(expected), "observed_count": len(observed),
                    "missing": sorted(expected - observed)[:20],
                    "unexpected": sorted(observed - expected)[:20],
                    "failed": sorted(failed)[:20],
                    "match": expected == observed and not failed}
        except (ValueError, KeyError, TypeError):
            return None
    return None


_INTENT = {
    "delete": (r"\b(delete|remove|cleanup|clear)\b", ("delete", "remove", "clear")),
    "move": (r"\b(move|relocate|organize|arrange)\b", ("move", "relocate")),
    "create": (r"\b(create|request|send|add)\b|\bmake\s+(?:\w+\s+){0,2}requests?\b",
               ("create", "request", "send", "add")),
    "update": (r"\b(update|edit|modify|rename)\b", ("update", "edit", "modify", "rename")),
}


@dataclass(frozen=True)
class CompletionCheck:
    status: str  # SATISFIED, MISSING, UNKNOWN
    allow: bool
    reason: str
    observed: tuple[str, ...]
    missing: tuple[str, ...]
    attempt: int

    def feedback(self) -> str:
        return ("Completion check " + self.status + ".\nObserved: " +
                ("; ".join(self.observed) or "No relevant business operation evidence.") +
                "\nMissing or uncertain: " + ("; ".join(self.missing) or "None identified.") +
                "\n" + self.reason)


class CompletionGuard:
    """Checks recorded API operations, not the order of advisory plan phases.

    A readback proves that a check was attempted, never that all task goals passed.
    The official evaluator remains the only source of task success.
    """

    def __init__(self, instruction: str, *, max_repairs: int = 2):
        self.instruction = instruction
        self.max_repairs = max_repairs
        self.attempts = 0
        self.intent = tuple(name for name, (pattern, _) in _INTENT.items()
                            if re.search(pattern, instruction, re.I))

    def review(self, evidence: list[dict[str, Any]], *, unresolved_error: bool,
               batch_uncertain: bool = False, batch_gap: bool = False,
               assurance_state: Any | None = None) -> CompletionCheck:
        self.attempts += 1
        valid = [row for row in evidence if row.get("ok") and row.get("api_name")]
        reads = [row for row in valid if row.get("method") in {"GET", "HEAD"}]
        writes = [row for row in valid if row.get("method") not in {"GET", "HEAD"} and
                  not re.search(r"\.(login|logout|complete_task|refresh_token)$", str(row["api_name"]), re.I)]
        observed = tuple(dict.fromkeys(str(row["api_name"]) for row in writes))[:8]
        if assurance_state is not None:
            goals = getattr(assurance_state, "goals", {})
            if getattr(assurance_state, "mode", None) == "enforce":
                missing_preconditions = [goal_id for goal_id in
                    getattr(assurance_state, "destructive_preconditions", [])
                    if goal_id not in goals or getattr(goals[goal_id].status, "value", goals[goal_id].status)
                    != "VERIFIED"]
            else:
                missing_preconditions = []
            if missing_preconditions:
                return CompletionCheck(
                    "MISSING", False,
                    "Required destructive-operation preconditions have not been verified.",
                    observed,
                    tuple(f"Goal {goal_id} is not VERIFIED" for goal_id in missing_preconditions),
                    self.attempts,
                )
        if unresolved_error:
            return CompletionCheck("MISSING", False, "Resolve or check the last execution error before submitting.",
                                   observed, ("Unresolved tool execution error",), self.attempts)
        if batch_gap:
            return CompletionCheck("MISSING", False,
                                   "The reported expected and read-back sets differ; correct or explain each item.",
                                   observed, ("Batch set difference or failed item remains",), self.attempts)
        if not self.intent:
            return CompletionCheck("SATISFIED", True, "No known completion blocker; this is not an official success verdict.",
                                   observed, (), self.attempts)
        matching = [row for row in writes if any(
            token in str(row["api_name"]).lower().split(".")[-1]
            for kind in self.intent for token in _INTENT[kind][1])]
        def has_readback(actions: list[dict[str, Any]]) -> bool:
            if not actions:
                return False
            latest = max((int(row["step"]), int(row.get("order", 0))) for row in actions)
            apps = {str(row["api_name"]).split(".")[0] for row in actions}
            return any((int(row["step"]), int(row.get("order", 0))) > latest and
                       str(row["api_name"]).split(".")[0] in apps for row in reads)

        if not matching:
            missing = ("No observed business write matches an explicit user action; "
                       "a valid no-op still needs visible state evidence.",)
            if writes:
                return CompletionCheck("UNKNOWN", has_readback(writes) or self.attempts > self.max_repairs,
                                       "A business write and later readback were observed, but API naming does "
                                       "not prove the exact user requirement. Inspect its outcome.",
                                       observed, missing, self.attempts)
            if any(row.get("empty_result") for row in reads):
                return CompletionCheck("UNKNOWN", self.attempts > self.max_repairs,
                                       "An empty read was observed. Confirm its filters and coverage; "
                                       "a no-op remains uncertain until official evaluation.",
                                       observed, missing, self.attempts)
            return CompletionCheck("MISSING", False,
                                   "Continue execution or provide explicit evidence that no write is needed.",
                                   observed, missing, self.attempts)
        if batch_uncertain:
            return CompletionCheck("UNKNOWN", has_readback(writes) or self.attempts > self.max_repairs,
                                   "Provide per-item outcomes and compare expected versus read-back target sets. "
                                   "The batch has not been proven complete.", observed,
                                   ("Batch write outcome remains uncertain",), self.attempts)
        readback = has_readback(matching)
        if not readback and self.attempts <= self.max_repairs:
            return CompletionCheck("UNKNOWN", False, "Read back the affected business state where possible.",
                                   observed, ("No post-write readback observed",), self.attempts)
        return CompletionCheck("SATISFIED" if readback else "UNKNOWN", True,
                               "No known blocker remains; business correctness is not proven by this check.",
                               observed, () if readback else ("Post-write state unverified",), self.attempts)
