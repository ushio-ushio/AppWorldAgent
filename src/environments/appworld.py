from __future__ import annotations

import ast
import json
from typing import Any
from uuid import uuid4

from execution_assurance.assurance import RequesterEventParser
from execution_assurance.recovery import (
    _local_correction, _persistent_python_names, _safe_during_uncertain_write,
    _same_target_readback, _correction_recovered, _executed_write_fingerprints,
    write_operation_fingerprints,
)
from runtime.trace import _redact as redact
from tool_engineering.tools import ApiRetriever, ToolRegistry


class AppWorldTool:
    """One persistent AppWorld instance is injected for the full graph invocation."""

    def __init__(self, world: Any, *, enable_api_retrieval: bool = True,
                 observation_store: ObservationStore | None = None,
                 working_memory: WorkingMemory | None = None,
                 task_state: TaskState | None = None,
                 manual_retrieval: bool = True, schema_aware_retrieval: bool = False,
                 outcome_gate: Any | None = None,
                 api_retriever: ApiRetriever | None = None):
        self.world = world
        self.last_operation_fingerprints: list[tuple[str, str]] = []
        self.observed_write_fingerprints: set[str] = set()
        self.outcome_gate = outcome_gate
        self.assurance_boundary_available = False
        self.assurance_step = 0
        self.assurance_call_id: str | None = None
        self.last_assurance_events: list[Any] = []
        self.last_execution_summary: dict[str, Any] | None = None
        self.consecutive_401: dict[str, int] = {}
        self.api_retriever = api_retriever
        if (self.api_retriever is None and enable_api_retrieval
                and hasattr(world.task, "api_docs") and hasattr(world.task, "allowed_apps")):
            self.api_retriever = ApiRetriever(world.task.api_docs, world.task.allowed_apps,
                                              schema_aware=schema_aware_retrieval)
        self.request_parser = None
        if self.api_retriever is not None:
            self.request_parser = RequesterEventParser({doc.qualified_name: doc.official_doc
                                                        for doc in self.api_retriever.documents})
        self.registry = ToolRegistry(
            search_api=self._search_api,
            get_api_schema=self._get_api_schema,
            execute_python=self.execute_python,
            execution_summary_provider=lambda: self.last_execution_summary,
            api_retrieval_enabled=enable_api_retrieval,
            schema_aware_retrieval=schema_aware_retrieval,
            search_history=observation_store.search_history if observation_store and manual_retrieval else None,
            read_observation=observation_store.read_observation if observation_store and manual_retrieval else None,
            remember_fact=working_memory.remember_fact if working_memory else None,
            list_memory=working_memory.list_memory if working_memory else None,
            invalidate_memory=working_memory.invalidate_memory if working_memory else None,
            update_task_state=task_state.update_task_state if task_state else None,
        )

    def _search_api(self, query: str | None = None, app: str | None = None, top_k: int = 5,
                    queries: list[str] | None = None,
                    app_name: str | None = None) -> Any:
        if self.api_retriever is None:
            raise RuntimeError("API retrieval is disabled for this run.")
        return self.api_retriever.search_api(query=query, app=app, top_k=top_k,
                                             queries=queries, app_name=app_name)

    def _get_api_schema(self, api_name: str) -> dict[str, Any] | None:
        if self.api_retriever is None:
            raise RuntimeError("API retrieval is disabled for this run.")
        return self.api_retriever.get_api_schema(api_name)

    def execute_python(self, code: str) -> tuple[str, str | None]:
        self.last_execution_summary = None
        if (self.outcome_gate is not None and self.outcome_gate.state.mode == "enforce"
                and not self.assurance_boundary_available):
            message = ("Execution Assurance is in enforce mode, but this environment has no verified "
                       "AppWorld Requester interception boundary; Python execution was refused.")
            return message, message
        fingerprints = []
        self.last_operation_fingerprints = []
        if self.api_retriever is not None:

            fingerprints = write_operation_fingerprints(code, self.api_retriever)
            for fingerprint, _api_name in fingerprints:
                already_observed = fingerprint in self.observed_write_fingerprints
                if already_observed:
                    return ("Write operation blocked because this exact operation was already attempted; inspect its prior Observation before deciding what to do.",
                            "write_replay_blocked")
        self.last_operation_fingerprints = fingerprints
        requester = getattr(self.world, "requester", None)
        requester_rows = getattr(requester, "requests", None)
        request_start = len(requester_rows) if isinstance(requester_rows, list) else 0
        if self.outcome_gate is not None and self.assurance_call_id:
            self.outcome_gate.begin_action(self.assurance_step, self.assurance_call_id)
        observation = ""
        error = None
        try:
            observation = str(self.world.execute(code))
            error = observation if observation.startswith("Execution failed.") else None
        except Exception as exc:
            message = f"Execution adapter error ({type(exc).__name__}): {exc}"
            observation, error = message, message
        finally:
            if self.outcome_gate is not None and self.assurance_call_id:
                self.last_assurance_events = self.outcome_gate.end_action(self.assurance_call_id, None)
        # Record side-effect attempts only after observing real Requester calls.
        # A Python/Shell error before the first request must not poison the
        # replay ledger merely because the source code mentioned a write API.
        requests_after = getattr(requester, "requests", None)
        parser = self.request_parser
        requests = ()
        if (parser is not None and isinstance(requester_rows, list)
                and requests_after is requester_rows and len(requests_after) > request_start):
            requests = parser.request_evidence(requests_after[request_start:])
        events = self.last_assurance_events
        inferred_status = None
        if error:
            import re
            status_match = re.search(r"Response status code is (\d{3})", error)
            inferred_status = int(status_match.group(1)) if status_match else None
        inferred_assigned = False
        for event in events:
            api_name = str(getattr(event, "api_name", ""))
            app_name = api_name.split(".", 1)[0].lower()
            status_code = getattr(event, "http_status", None)
            if (status_code is None and not inferred_assigned
                    and getattr(event, "execution_status", "") == "failed"):
                status_code = inferred_status
                inferred_assigned = inferred_status is not None
            if getattr(event, "execution_status", "") == "succeeded":
                self.consecutive_401.pop(app_name, None)
            elif status_code == 401 and app_name:
                self.consecutive_401[app_name] = self.consecutive_401.get(app_name, 0) + 1
        if error or any(getattr(event, "execution_status", "") != "succeeded" for event in events):
            successes = [event for event in events if event.execution_status == "succeeded"]
            failures = [event for event in events if event.execution_status in {"failed", "blocked", "unknown"}]
            first = failures[0] if failures else None
            self.last_execution_summary = {
                "total_calls": max(len(requests), len(events)),
                "succeeded_calls": len(successes), "failed_calls": len(failures),
                "succeeded_mutations": [event.api_name for event in successes
                    if str(getattr(event.operation_type, "value", event.operation_type)) in {"WRITE", "DESTRUCTIVE"}
                    and not event.api_name.endswith((".login", ".logout", ".complete_task"))],
                "first_failure": ({"endpoint": first.api_name, "status_code": first.http_status,
                                   "error": first.reason} if first else None),
                "consecutive_401": dict(self.consecutive_401),
            }
            if first and self.last_execution_summary["first_failure"]["status_code"] is None and inferred_status:
                self.last_execution_summary["first_failure"]["status_code"] = inferred_status
            if first and self.last_execution_summary["first_failure"]["status_code"] is None and error:
                import re
                status = re.search(r"Response status code is (\d{3})", error)
                if status:
                    self.last_execution_summary["first_failure"]["status_code"] = int(status.group(1))
        if (fingerprints and parser is not None and hasattr(parser, "request_evidence") and
                isinstance(requester_rows, list) and requests_after is requester_rows and
                len(requests_after) > request_start):
            for fingerprint, api_name in _executed_write_fingerprints(fingerprints, requests):
                self.observed_write_fingerprints.add(fingerprint)
        return observation, error

    def set_assurance_action(self, step: int, tool_call_id: str) -> None:
        self.assurance_step = step
        self.assurance_call_id = tool_call_id

    def link_assurance_observation(self, tool_call_id: str, observation_id: str | None) -> list[Any]:
        if self.outcome_gate is None:
            return []
        return self.outcome_gate.link_observation(tool_call_id, observation_id)

    @staticmethod
    def _is_complete_task_call(node: ast.AST) -> bool:
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "complete_task"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "supervisor"
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "apis")

    @staticmethod
    def _contains_completion_reference(tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "complete_task":
                names = {child.id for child in ast.walk(node.value) if isinstance(child, ast.Name)}
                attrs = {child.attr for child in ast.walk(node.value) if isinstance(child, ast.Attribute)}
                if "apis" in names and "supervisor" in attrs:
                    return True
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr" and node.args):
                base = node.args[0]
                names = {child.id for child in ast.walk(base) if isinstance(child, ast.Name)}
                attrs = {child.attr for child in ast.walk(base) if isinstance(child, ast.Attribute)}
                method_name = (len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
                               and node.args[1].value == "complete_task")
                if "apis" in names and ("supervisor" in attrs or method_name):
                    return True
        return False

    @classmethod
    def _completion_form(cls, code: str) -> str | None:
        """Recognize only an isolated direct call or a single print wrapper.

        `None` means no syntactically valid complete_task call was found.
        "invalid" means a completion call exists but its surrounding Python
        structure is outside the explicitly supported submission protocol.
        """
        try:
            tree = ast.parse(code)
        except (SyntaxError, TypeError):
            return None
        completion_calls = [node for node in ast.walk(tree) if cls._is_complete_task_call(node)]
        if not completion_calls:
            return "invalid" if cls._contains_completion_reference(tree) else None
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
            return "invalid"
        expression = tree.body[0].value
        all_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        if cls._is_complete_task_call(expression):
            return "direct" if len(completion_calls) == 1 and len(all_calls) == 1 else "invalid"
        if (isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name)
                and expression.func.id == "print" and len(expression.args) == 1
                and not expression.keywords and cls._is_complete_task_call(expression.args[0])
                and len(completion_calls) == 1 and len(all_calls) == 2):
            return "print"
        return "invalid"
