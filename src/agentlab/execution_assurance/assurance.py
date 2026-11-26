"""Task-local execution assurance for AppWorld's in-process Requester boundary.

The module is deliberately deterministic. It records every API request made
through AppWorld's Requester, gates only explicitly declared destructive
preconditions, and never treats a model claim or an API call count as proof.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import re
from typing import Any, Mapping
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4


class OperationType(str, Enum):
    READ = "READ"
    WRITE = "WRITE"
    DESTRUCTIVE = "DESTRUCTIVE"
    UNKNOWN = "UNKNOWN"


class GoalStatus(str, Enum):
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


_READ_ONLY_POST_PREFIXES = (
    "search", "query", "list", "show", "read", "find", "lookup", "check",
    "verify", "validate", "login", "sign_in", "authenticate",
)


def read_only_probe_allowed(method: str, api_name: str) -> bool:
    """Allow safe HTTP reads and clearly query/authentication-style POST APIs."""
    method = str(method or "").upper()
    if method in {"GET", "HEAD", "OPTIONS"}:
        return True
    if method != "POST":
        return False
    operation = str(api_name or "").rsplit(".", 1)[-1].lower()
    return any(operation == prefix or operation.startswith(prefix + "_")
               for prefix in _READ_ONLY_POST_PREFIXES)


class ReadOnlyProbeBlocked(RuntimeError):
    """Raised at the Requester boundary when a reflection probe attempts a write."""


def classify_operation(api_name: str, metadata: Mapping[str, Any] | None) -> OperationType:
    """Classify only from official method metadata; never guess from API names."""
    method = str((metadata or {}).get("method", "")).upper()
    if method in {"GET", "HEAD", "OPTIONS"}:
        return OperationType.READ
    if method == "DELETE":
        return OperationType.DESTRUCTIVE
    if method in {"POST", "PUT", "PATCH"}:
        return OperationType.WRITE
    return OperationType.UNKNOWN


_SENSITIVE_KEY = re.compile(
    r"password|(?:access|refresh|id|bearer)?[_ -]?token|api[_ -]?key|secret|authorization|cookie",
    re.I,
)


def _safe_value(value: Any, *, key: str = "", depth: int = 0) -> Any:
    if _SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if depth > 4:
        return "[NESTED]"
    if value is None or type(value) in (bool, int, float):
        return value
    if isinstance(value, str):
        if len(value) > 240:
            return {"truncated": True, "length": len(value)}
        return value
    if isinstance(value, Mapping):
        return {str(k): _safe_value(v, key=str(k), depth=depth + 1)
                for k, v in list(value.items())[:50]}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item, depth=depth + 1) for item in value[:50]]
    return f"<{type(value).__name__}>"


class RequesterEventParser:
    """Map actual AppWorld Requester calls to task-allowed API metadata."""

    def __init__(self, api_metadata: Mapping[str, Mapping[str, Any]]):
        self.routes: dict[str, list[tuple[str, str, re.Pattern[str]]]] = {}
        self._query_salt = uuid4().hex
        for name, schema in api_metadata.items():
            path, method = schema.get("path"), schema.get("method")
            if not isinstance(path, str) or not isinstance(method, str):
                continue
            pattern = "/".join(r"[^/]+" if part.startswith("{") and part.endswith("}")
                               else re.escape(part) for part in path.rstrip("/").split("/"))
            self.routes.setdefault(method.upper(), []).append(
                (str(name).lower(), path.rstrip("/"), re.compile(pattern)))
        for rows in self.routes.values():
            rows.sort(key=lambda row: row[1].count("{"))

    def request_evidence(self, requests: Any) -> tuple[dict[str, Any], ...]:
        """Return sanitized route evidence only; credentials and raw bodies are omitted."""
        evidence = []
        for request in requests if isinstance(requests, (list, tuple)) else ():
            method = str(request.get("method", "")).upper()
            parsed = urlsplit(str(request.get("url", "")))
            path = parsed.path.rstrip("/")
            if parsed.hostname and "." in parsed.hostname:
                app = parsed.hostname.split(".", 1)[0]
                if not path.startswith(f"/{app}/"):
                    path = f"/{app}{path}"
            match = next(((name, route) for name, route, pattern in self.routes.get(method, [])
                          if pattern.fullmatch(path)), None)
            params = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
            data = request.get("data")
            if isinstance(data, dict):
                params.update(data)
            row: dict[str, Any] = {"method": method, "api_name": match[0] if match else None,
                                   "route": match[1] if match else None}
            if match:
                row["target_key"] = __import__("hashlib").sha256(path.encode()).hexdigest()
                for key in ("page_index", "page_limit"):
                    value = params.get(key)
                    if type(value) is int or (isinstance(value, str) and value.isascii() and value.isdecimal()):
                        row[key] = int(value)
                ignored = {"page_index", "page_limit", "access_token"}
                filters = {key: value for key, value in params.items() if key not in ignored
                           and not any(secret in key.lower() for secret in ("token", "password", "secret", "key"))}
                query = [self._query_salt, params.get("access_token"), filters]
                row["query_key"] = __import__("hashlib").sha256(
                    json.dumps(query, sort_keys=True, default=str).encode()).hexdigest()
                identity = {key: value for key, value in filters.items()
                            if key.lower() == "id" or key.lower().endswith(("_id", "_ids"))}
                if identity:
                    row["entity_key"] = __import__("hashlib").sha256(
                    json.dumps([self._query_salt, identity], sort_keys=True, default=str).encode()).hexdigest()
            evidence.append(row)
        return tuple(evidence)


@dataclass
class GoalRecord:
    goal_id: str
    description: str
    source_requirement: str
    source_requirement_matched: bool
    constraints: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    completion_criteria: list[str] = field(default_factory=list)
    status: GoalStatus = GoalStatus.PENDING
    evidence_ids: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)


@dataclass
class ExecutionEvent:
    event_id: str
    task_id: str
    step: int
    action_call_id: str
    api_name: str
    arguments: dict[str, Any]
    operation_type: OperationType
    execution_status: str
    result_reference: str | None = None
    related_goal_ids: list[str] = field(default_factory=list)
    http_status: int | None = None
    risk_level: str = "unknown"
    risk_source: str | None = None
    risk_review_required: bool = False
    preconditions_declared: bool = False
    gate_would_block: bool = False
    gate_blocked: bool = False
    missing_preconditions: list[str] = field(default_factory=list)
    reason: str | None = None
    response_item_count: int | None = field(default=None, repr=False, compare=False)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        if self.response_item_count is None:
            row.pop("response_item_count", None)
        row["operation_type"] = self.operation_type.value
        return row


@dataclass
class ExecutionAssuranceState:
    task_id: str
    instruction: str
    mode: str
    goals: dict[str, GoalRecord] = field(default_factory=dict)
    destructive_preconditions: list[str] = field(default_factory=list)
    events: list[ExecutionEvent] = field(default_factory=list)
    unresolved_issues: list[dict[str, Any]] = field(default_factory=list)
    plan_valid: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"shadow", "enforce", "reflection_v1", "reflection_v2"}:
            raise ValueError("assurance mode must be shadow, enforce, reflection_v1, or reflection_v2")

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "mode": self.mode,
            "plan_valid": self.plan_valid,
            "goals": {key: {**asdict(goal), "status": goal.status.value}
                      for key, goal in self.goals.items()},
            "destructive_preconditions": list(self.destructive_preconditions),
            "events": [event.as_dict() for event in self.events],
            "unresolved_issues": list(self.unresolved_issues),
        }


class GoalTracker:
    """Single task-scoped source of goal status and evidence references."""

    def __init__(self, state: ExecutionAssuranceState):
        self.state = state

    @classmethod
    def from_plan(cls, task_id: str, instruction: str, plan: dict[str, Any] | None,
                  *, mode: str = "shadow") -> "GoalTracker":
        state = ExecutionAssuranceState(task_id=task_id, instruction=instruction, mode=mode)
        plan = plan if isinstance(plan, dict) else {}
        rows = plan.get("goals")
        if not isinstance(rows, list) or not rows:
            # Backward-compatible one-shot plans become explicit pending checks.
            criteria = plan.get("completion_checks") or ["Complete all explicit user requirements."]
            rows = [{"id": f"G{i + 1}", "description": str(item),
                     "source_requirement": instruction,
                     "completion_criteria": [str(item)]}
                    for i, item in enumerate(criteria[:12])]
            # Fallback goals remain useful for context/diagnostics, but are not
            # a strong enough contract to authorize irreversible operations.
            state.plan_valid = False
            if plan and not plan.get("goals"):
                state.unresolved_issues.append({"kind": "planner_goal_fallback",
                    "detail": "Planner supplied no structured goals; completion checks are tracked conservatively."})
        valid_rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        try:
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("goal must be an object")
                goal_id = row.get("id")
                description = row.get("description")
                if (not isinstance(goal_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", goal_id)
                        or goal_id in seen or not isinstance(description, str) or not description.strip()):
                    raise ValueError("goal id/description is invalid")
                seen.add(goal_id)
                deps = row.get("dependencies", [])
                criteria = row.get("completion_criteria", [])
                constraints = row.get("constraints", {})
                source = row.get("source_requirement", instruction)
                if (not isinstance(deps, list) or any(not isinstance(v, str) for v in deps)
                        or not isinstance(criteria, list) or any(not isinstance(v, str) for v in criteria)
                        or not isinstance(constraints, dict) or not isinstance(source, str)):
                    raise ValueError("goal fields have invalid types")
                if source not in instruction:
                    raise ValueError("goal source_requirement is not an exact task substring")
                valid_rows.append({"id": goal_id, "description": description[:500],
                    "source_requirement": source[:1000], "dependencies": deps,
                    "completion_criteria": [v[:300] for v in criteria[:12]],
                    "constraints": _safe_value(constraints)})
            for row in valid_rows:
                if set(row["dependencies"]) - seen:
                    raise ValueError("goal references an unknown dependency")
            state.goals = {row["id"]: GoalRecord(
                goal_id=row["id"], description=row["description"],
                source_requirement=row["source_requirement"],
                source_requirement_matched=(row["source_requirement"] in instruction),
                constraints=row["constraints"], dependencies=row["dependencies"],
                completion_criteria=row["completion_criteria"])
                for row in valid_rows}
            preconditions = plan.get("destructive_preconditions", [])
            if not isinstance(preconditions, list) or any(not isinstance(v, str) for v in preconditions):
                raise ValueError("destructive_preconditions must be goal IDs")
            if set(preconditions) - set(state.goals):
                raise ValueError("unknown destructive precondition goal")
            state.destructive_preconditions = list(dict.fromkeys(preconditions))
            cls._validate_dag(state.goals)
        except (TypeError, ValueError) as exc:
            state.plan_valid = False
            state.goals.clear()
            state.destructive_preconditions.clear()
            state.unresolved_issues.append({"kind": "invalid_planner_goal_contract",
                "detail": type(exc).__name__})
            checks = plan.get("completion_checks") or ["Complete all explicit user requirements."]
            for i, item in enumerate(checks[:12]):
                gid = f"G{i + 1}"
                state.goals[gid] = GoalRecord(gid, str(item), instruction, True,
                                              completion_criteria=[str(item)])
        return cls(state)

    @staticmethod
    def _validate_dag(goals: dict[str, GoalRecord]) -> None:
        visiting: set[str] = set(); visited: set[str] = set()
        def visit(goal_id: str) -> None:
            if goal_id in visiting:
                raise ValueError("goal dependency cycle")
            if goal_id in visited:
                return
            visiting.add(goal_id)
            for dependency in goals[goal_id].dependencies:
                visit(dependency)
            visiting.remove(goal_id); visited.add(goal_id)
        for goal_id in goals:
            visit(goal_id)

    def render_context(self, *, max_chars: int = 1800) -> str:
        rows = []
        for goal in self.state.goals.values():
            evidence = ",".join(goal.evidence_ids[-3:]) or "none"
            gap = "; ".join(goal.unresolved[-2:]) or "not yet verified"
            rows.append(f"{goal.goal_id}: {goal.status.value}; {goal.description[:180]}; "
                        f"evidence={evidence}; gap={gap}")
        if not rows:
            rows.append("No structured goals; do not treat an agent claim as proof.")
        text = "EXECUTION ASSURANCE (task-local evidence; this status is not an evaluator result):\n" + "\n".join(rows)
        return text[:max_chars]

    def related_goal_ids(self, api_name: str, description: str = "") -> list[str]:
        """Return a link only for a clear unique lexical match; otherwise none."""
        def terms(text: str) -> set[str]:
            return {x for x in re.findall(r"[a-z0-9]+", text.lower().replace("_", " ").replace(".", " ")) if len(x) > 2}
        api_terms = terms(api_name + " " + description)
        scored = []
        for goal in self.state.goals.values():
            goal_terms = terms(goal.description + " " + " ".join(goal.completion_criteria))
            score = len(api_terms & goal_terms) / max(len(api_terms), 1)
            if score >= 0.45:
                scored.append((score, goal.goal_id))
        scored.sort(reverse=True)
        if scored and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.2):
            return [scored[0][1]]
        return []

    def apply_event(self, event: ExecutionEvent) -> None:
        for goal_id in event.related_goal_ids:
            goal = self.state.goals.get(goal_id)
            if not goal:
                continue
            if event.execution_status == "failed":
                goal.status = GoalStatus.FAILED
                goal.unresolved.append(f"{event.api_name} failed; see {event.event_id}")
            elif event.execution_status == "unknown":
                goal.status = GoalStatus.UNKNOWN
                goal.unresolved.append(f"{event.api_name} outcome is uncertain; inspect {event.event_id}")
            elif event.execution_status == "blocked":
                goal.unresolved.append(f"{event.api_name} was blocked before execution; see {event.event_id}")
            elif event.operation_type != OperationType.READ and goal.status != GoalStatus.VERIFIED:
                goal.status = GoalStatus.UNKNOWN
                goal.unresolved.append(f"{event.api_name} executed; target state is not verified")
            if event.result_reference and event.result_reference not in goal.evidence_ids:
                goal.evidence_ids.append(event.result_reference)

class ActionGateBlocked(RuntimeError):
    pass


class ActionGate:
    _HIGH_RISK_DESCRIPTION = re.compile(
        r"permanent(?:ly)?\s+(?:delete|remove|erase)|delete.{0,30}account|terminate.{0,30}account|"
        r"close.{0,30}account|transfer.{0,30}(?:fund|payment)|(?:payment|fund).{0,30}transfer|"
        r"deactivate.{0,30}account|send.{0,20}payment", re.I)

    def __init__(self, state: ExecutionAssuranceState, api_metadata: Mapping[str, Mapping[str, Any]],
                 tracker: GoalTracker | None = None):
        self.state = state
        self.api_metadata = {k.lower(): v for k, v in api_metadata.items()}
        self.tracker = tracker

    def check(self, api_name: str) -> dict[str, Any]:
        metadata = self.api_metadata.get(api_name.lower(), {})
        op = classify_operation(api_name, metadata)
        target_goal_ids = (self.tracker.related_goal_ids(api_name, str(metadata.get("description", "")))
                           if self.tracker else [])
        declared = list(self.state.destructive_preconditions)
        for goal_id in target_goal_ids:
            target = self.state.goals.get(goal_id)
            if target:
                declared.extend(target.dependencies)
        declared = list(dict.fromkeys(declared))
        missing = [goal_id for goal_id in declared
                   if self.state.goals.get(goal_id) is None or
                   self.state.goals[goal_id].status != GoalStatus.VERIFIED]
        if op == OperationType.DESTRUCTIVE and not self.state.plan_valid:
            missing.insert(0, "valid_structured_plan")
        should_block = op == OperationType.DESTRUCTIVE and bool(missing)
        blocked = should_block and self.state.mode == "enforce"
        reason = ("Destructive action requires verified task preconditions: " + ", ".join(missing)
                  if should_block else None)
        method = str(metadata.get("method", "")).upper()
        description = str(metadata.get("description", ""))
        risk_level = ("destructive" if op == OperationType.DESTRUCTIVE else
                      "high_risk_candidate" if method in {"POST", "PUT", "PATCH"}
                      and self._HIGH_RISK_DESCRIPTION.search(description) else "ordinary")
        risk_source = "official_http_method" if op == OperationType.DESTRUCTIVE else (
            "official_schema_description" if risk_level == "high_risk_candidate" else None)
        risk_review_required = risk_level in {"destructive", "high_risk_candidate"}
        return {"operation_type": op, "allowed": not blocked, "blocked": blocked,
                "would_block": should_block, "missing_preconditions": missing,
                "target_goal_ids": target_goal_ids, "preconditions_declared": bool(declared),
                "risk_level": risk_level, "risk_source": risk_source,
                "risk_review_required": risk_review_required,
                "reason": reason, "enforcement": self.state.mode}


class OutcomeGate:
    def __init__(self, state: ExecutionAssuranceState, tracker: GoalTracker,
                 api_metadata: Mapping[str, Mapping[str, Any]]):
        self.state = state
        self.tracker = tracker
        self.api_metadata = {k.lower(): v for k, v in api_metadata.items()}
        self.action_gate = ActionGate(state, api_metadata, tracker)
        self._active: dict[str, Any] | None = None

    def begin_action(self, step: int, action_call_id: str) -> None:
        self._active = {"step": step, "action_call_id": action_call_id, "event_ids": []}

    def end_action(self, action_call_id: str, observation_id: str | None) -> list[ExecutionEvent]:
        if self._active is None or self._active["action_call_id"] != action_call_id:
            return []
        events = [event for event in self.state.events if event.event_id in self._active["event_ids"]]
        for event in events:
            event.result_reference = observation_id
            self.tracker.apply_event(event)
        self._active = None
        return events

    def link_observation(self, action_call_id: str, observation_id: str | None) -> list[ExecutionEvent]:
        if not observation_id:
            return []
        linked = []
        for event in self.state.events:
            if event.action_call_id != action_call_id or event.result_reference is not None:
                continue
            event.result_reference = observation_id
            linked.append(event)
            for goal_id in event.related_goal_ids:
                goal = self.state.goals.get(goal_id)
                if goal and observation_id not in goal.evidence_ids:
                    goal.evidence_ids.append(observation_id)
        return linked

    def _append_event(self, *, api_name: str, arguments: Mapping[str, Any], status: str,
                      operation: OperationType, gate: dict[str, Any], http_status: int | None = None,
                      reason: str | None = None,
                      response_item_count: int | None = None) -> ExecutionEvent:
        active = self._active or {"step": 0, "action_call_id": "external", "event_ids": []}
        related = self.tracker.related_goal_ids(api_name,
                    str(self.api_metadata.get(api_name.lower(), {}).get("description", "")))
        event = ExecutionEvent(
            event_id="ev_" + uuid4().hex[:16], task_id=self.state.task_id,
            step=int(active["step"]), action_call_id=str(active["action_call_id"]),
            api_name=api_name, arguments=_safe_value(dict(arguments)), operation_type=operation,
            execution_status=status, related_goal_ids=related, http_status=http_status,
            reason=reason,
            risk_level=str(gate.get("risk_level", "unknown")),
            risk_source=gate.get("risk_source"),
            risk_review_required=bool(gate.get("risk_review_required")),
            preconditions_declared=bool(gate.get("preconditions_declared")),
            gate_would_block=bool(gate.get("would_block")),
            gate_blocked=bool(gate.get("blocked")),
            missing_preconditions=list(gate.get("missing_preconditions", [])),
            response_item_count=response_item_count)
        self.state.events.append(event)
        active["event_ids"].append(event.event_id)
        if self._active is None:
            self.tracker.apply_event(event)
        if status in {"failed", "unknown"}:
            self.state.unresolved_issues.append({"event_id": event.event_id,
                "kind": "api_execution_" + status, "api_name": api_name,
                "operation_type": operation.value, "reason": reason})
        return event

    def request(self, original_request: Any, api_name: str, arguments: dict[str, Any]) -> Any:
        decision = self.action_gate.check(api_name)
        operation = OperationType(decision["operation_type"])
        if decision["would_block"]:
            self.state.unresolved_issues.append({"kind": "action_gate_precondition",
                "api_name": api_name, "blocked": decision["blocked"],
                "missing_preconditions": decision["missing_preconditions"]})
        if decision["blocked"]:
            self._append_event(api_name=api_name, arguments=arguments, status="blocked",
                               operation=operation, gate=decision, reason=decision["reason"])
            raise ActionGateBlocked(decision["reason"] or "Action blocked by assurance policy")
        try:
            result = original_request(**arguments)
        except Exception as exc:
            text = str(exc)
            uncertain = operation in {OperationType.WRITE, OperationType.DESTRUCTIVE} and bool(
                re.search(r"timeout|timed out|connection reset|connection aborted|broken pipe", text, re.I))
            status = "unknown" if uncertain else "failed"
            self._append_event(api_name=api_name, arguments=arguments, status=status,
                               operation=operation, gate=decision,
                               reason=type(exc).__name__ if not uncertain else "write_outcome_uncertain")
            raise
        http_status = None
        # Requester normally raises on unsuccessful HTTP responses. In
        # non-raising mode, preserve uncertainty rather than infer success.
        success = True
        if isinstance(result, dict) and result.get("status_code") not in (None, 200, 201, 202, 204):
            try:
                http_status = int(result["status_code"])
                success = False
            except (TypeError, ValueError):
                success = False
        event = self._append_event(api_name=api_name, arguments=arguments,
                           status="succeeded" if success else "failed", operation=operation,
                           gate=decision, http_status=http_status,
                            reason=None if success else "requester_returned_non_success_status",
                            response_item_count=(len(result) if self.state.mode in {"reflection_v1", "reflection_v2"}
                                                 and isinstance(result, list) else None))
        return result


def pagination_hints(events: Iterable[Any], api_retriever: Any = None) -> str:
    """Warn when a full page-zero response was not followed by a later page."""
    rows = list(events)
    def page_number(value: Any) -> int | None:
        try:
            return int(value) if (type(value) is int or
                                  isinstance(value, str) and value.isascii() and value.isdecimal()) else None
        except ValueError:
            return None
    later_pages = {str(getattr(event, "api_name", "")).lower() for event in rows
                   if isinstance(getattr(event, "arguments", None), Mapping)
                   and (page_number(event.arguments.get("page_index")) or 0) >= 1}
    schemas = {doc.qualified_name.lower(): doc.official_doc
               for doc in getattr(api_retriever, "documents", ())}
    hints, seen = [], set()
    for event in rows:
        if getattr(event, "execution_status", None) != "succeeded":
            continue
        api_name = str(getattr(event, "api_name", ""))
        normalized = api_name.lower()
        arguments = getattr(event, "arguments", {})
        arguments = arguments if isinstance(arguments, Mapping) else {}
        schema = schemas.get(normalized, {})
        parameters = schema.get("parameters", []) if isinstance(schema, Mapping) else []
        schema_paged = any(isinstance(row, Mapping) and row.get("name") == "page_index" for row in parameters)
        if not ("page_index" in arguments or "page_limit" in arguments or schema_paged):
            continue
        page = page_number(arguments.get("page_index", 0))
        count = getattr(event, "response_item_count", None)
        if page != 0 or normalized in later_pages or normalized in seen or type(count) is not int:
            continue
        limit = page_number(arguments.get("page_limit")) if "page_limit" in arguments else None
        threshold = limit if limit is not None and limit > 0 else 5
        if count < threshold:
            continue
        hints.append(f"[Pagination Hint] {api_name} returned {count} items on page 0 and page 1 was not fetched; "
                     "paginate until empty list [] to confirm full coverage.")
        seen.add(normalized)
    return "\n".join(hints)

class RequesterBoundary:
    """Per-task monkey-patch of the AppWorld Requester instance (never vendor code)."""

    def __init__(self, requester: Any, outcome_gate: OutcomeGate):
        self.requester = requester
        self.outcome_gate = outcome_gate
        self.original_request = getattr(requester, "request", None)
        self._installed = False
        self.read_only = False
        self.blocked_attempts: list[dict[str, Any]] = []

    def set_read_only(self, enabled: bool) -> None:
        self.read_only = bool(enabled)

    def drain_blocked_attempts(self) -> list[dict[str, Any]]:
        attempts, self.blocked_attempts = self.blocked_attempts, []
        return attempts

    @property
    def available(self) -> bool:
        return callable(self.original_request) and self._installed

    def install(self) -> None:
        if self._installed:
            return
        if not callable(self.original_request):
            raise TypeError("AppWorld Requester has no callable request method")
        original = self.original_request
        gate = self.outcome_gate
        def wrapped(_app_name: str, _api_name: str, *args: Any, **kwargs: Any) -> Any:
            api_name = f"{_app_name}.{_api_name}"
            if self.read_only:
                metadata = gate.api_metadata.get(api_name.lower(), {})
                method = str(metadata.get("method", "")).upper()
                if not read_only_probe_allowed(method, api_name):
                    self.blocked_attempts.append({"api_name": api_name, "method": method,
                                                  "reason": "reflection_probe_read_only"})
                    raise ReadOnlyProbeBlocked(
                        f"Reflection probe is read-only: {api_name} ({method or 'unknown method'}) was not executed. "
                        "Use a documented GET or query/login-style POST request.")
            if gate._active is None:
                return original(_app_name, _api_name, *args, **kwargs)
            def invoke(**request_kwargs: Any) -> Any:
                return original(_app_name, _api_name, *args, **request_kwargs)
            return gate.request(invoke, api_name, kwargs)
        self.requester.request = wrapped
        self._installed = True

    def restore(self) -> None:
        if self._installed:
            self.requester.request = self.original_request
            self._installed = False


def make_assurance(plan: dict[str, Any] | None, task_id: str, instruction: str,
                   api_metadata: Mapping[str, Mapping[str, Any]], *, mode: str = "shadow") -> tuple[GoalTracker, OutcomeGate]:
    tracker = GoalTracker.from_plan(task_id, instruction, plan, mode=mode)
    return tracker, OutcomeGate(tracker.state, tracker, api_metadata)
