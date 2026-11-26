from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from agentlab.agent.planner import one_shot_plan
from agentlab.agent.prompt import ContextManager
from agentlab.context_engineering.dynamic import DynamicContextBuilder
from agentlab.memory.store import ObservationStore
from agentlab.context_engineering.context import CacheFirstContextBuilder
from agentlab.context_engineering.unified import UnifiedContextBuilder, UnifiedContextConfig
from agentlab.environments.appworld import AppWorldTool
from agentlab.execution_assurance.recovery import (
    _correction_recovered, _documented_readback_only, _executed_write_fingerprints,
    _failed_python_action, _guard_blocking_correction, _has_unresolved_business_error,
    _local_correction, _same_target_readback, _safe_during_uncertain_write,
)
from agentlab.execution_assurance.completion_guard import CompletionGuard, batch_set_difference
from agentlab.execution_assurance.assurance import (
    RequesterBoundary, make_assurance, pagination_hints, RequesterEventParser,
)
from agentlab.execution_assurance.reflection import TerminalReflection, read_only_probe_feedback, repair_feedback
from agentlab.execution_assurance.reflection_v2 import TerminalAudit, audit_feedback
from agentlab.memory.store import WorkingMemory
from agentlab.runtime.task_state import TaskState
from agentlab.runtime.trace import TraceLogger, _zero_usage, _add_usage
from agentlab.models.siwc_responses import SIWCContextOverflowError
from agentlab.tool_engineering.tools import ApiRetriever, app_capability_catalog


class AgentState(TypedDict, total=False):
    task_id: str
    messages: list[dict[str, str]]
    step_count: int
    finished: bool
    error: str | None
    observation: str
    stop_reason: str | None
    usage: dict[str, float | int]
    total_usage: dict[str, float | int]
    model_response: str
    step_started_at: str
    step_started_perf: float
    step_duration_seconds: float
    pending_tool_calls: list[dict[str, Any]]
    assistant_message: dict[str, Any]
    protocol_retry_count: int
    protocol_error: str | None
    tool_call_count: int
    python_execution_count: int
    active_goal_id: str | None


class Model(Protocol):
    def generate_tool_calls(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *,
                            tool_choice: str = "required") -> tuple[dict[str, Any], dict[str, Any]]: ...


class LangGraphAgent:
    def __init__(self, model: Model, *, max_steps: int = 30, max_cost_usd: float = 0.05,
                 max_input_tokens: int | None = 60000, max_output_tokens: int = 6000,
                 enable_api_retrieval: bool = True, tool_choice: str = "required",
                 max_protocol_retries: int = 1, context_mode: str = "full",
                 schema_aware_retrieval: bool = False, programmatic_execution: bool = False,
                 unified_config: UnifiedContextConfig | None = None,
                 plan_guided_react: bool = False,
                 assurance_mode: str = "shadow",
                 max_no_progress_steps: int = 18):
        if context_mode not in {"full", "dynamic", "unified", "cache_first"}:
            raise ValueError("Unknown context_mode.")
        self.model = model
        self.max_steps = max_steps
        self.max_cost_usd = max_cost_usd
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.enable_api_retrieval = enable_api_retrieval
        self.tool_choice = tool_choice
        self.max_protocol_retries = max(0, min(1, max_protocol_retries))
        self.context_mode = context_mode
        self.schema_aware_retrieval = schema_aware_retrieval
        self.programmatic_execution = programmatic_execution
        self.unified_config = unified_config or UnifiedContextConfig()
        if assurance_mode not in {"shadow", "enforce", "reflection_v1", "reflection_v2"}:
            raise ValueError("assurance_mode must be 'shadow', 'enforce', 'reflection_v1', or 'reflection_v2'.")
        if assurance_mode in {"enforce", "reflection_v1", "reflection_v2"} and not plan_guided_react:
            raise ValueError(f"assurance_mode={assurance_mode} requires plan_guided_react and its task-local plan.")
        self.assurance_mode = assurance_mode
        self.plan_guided_react = plan_guided_react
        self.max_no_progress_steps = max(1, max_no_progress_steps)

    def build_graph(self, world: Any, trace: TraceLogger, *, runtime: dict[str, Any] | None = None):
        runtime = runtime or {}
        observation_store = runtime.get("observation_store")
        if observation_store is None and (self.context_mode != "full" or runtime.get("assurance_enabled")):
            observation_store = ObservationStore(world.task_id)
        working_memory = WorkingMemory(observation_store) if observation_store and self.context_mode == "unified" else None
        task_state = runtime.get("task_state")
        if task_state is None and observation_store and self.context_mode == "unified":
            task_state = TaskState(world.task_id, world.task.instruction, observation_store)
        if runtime.get("working_memory") is not None:
            working_memory = runtime["working_memory"]
        context_builder = (
            CacheFirstContextBuilder(observation_store, model=self.model)
            if self.context_mode == "cache_first" and observation_store else
            UnifiedContextBuilder(observation_store, working_memory, task_state, self.unified_config)
            if self.context_mode == "unified" else
            DynamicContextBuilder(observation_store, strategy=self.context_mode)
            if observation_store else None
        )
        tool = runtime.get("tool") or AppWorldTool(
            world, enable_api_retrieval=self.enable_api_retrieval,
            schema_aware_retrieval=self.schema_aware_retrieval,
            observation_store=observation_store,
            working_memory=working_memory if self.unified_config.memory else None,
            task_state=task_state if self.unified_config.task_state else None,
            manual_retrieval=self.unified_config.manual_retrieval if self.context_mode == "unified" else True,
            outcome_gate=runtime.get("outcome_gate"),
            api_retriever=runtime.get("api_retriever"))
        if runtime.get("assurance_enabled"):
            api_metadata = ({doc.qualified_name: doc.official_doc
                             for doc in tool.api_retriever.documents}
                            if tool.api_retriever else {})
            tracker, outcome_gate = make_assurance(runtime.get("assurance_plan"), world.task_id,
                world.task.instruction, api_metadata, mode=runtime.get("assurance_mode", "shadow"))
            runtime["goal_tracker"] = tracker
            runtime["assurance_state"] = tracker.state
            runtime["outcome_gate"] = outcome_gate
            tool.outcome_gate = outcome_gate
            requester = getattr(world, "requester", None)
            remote_environment = getattr(world, "remote_environment_url", None)
            if (requester is not None and callable(getattr(requester, "request", None))
                    and not remote_environment):
                try:
                    boundary_hook = RequesterBoundary(requester, outcome_gate)
                    boundary_hook.install()
                    runtime["requester_boundary"] = boundary_hook
                    runtime["api_boundary_coverage"] = "in_process_appworld_requester"
                    tool.assurance_boundary_available = boundary_hook.available
                except Exception as exc:
                    runtime["api_boundary_coverage"] = "unavailable_requester_hook_failed"
                    tracker.state.unresolved_issues.append({"kind": "api_boundary_unavailable",
                        "detail": f"Requester hook installation failed: {type(exc).__name__}"})
            else:
                runtime["api_boundary_coverage"] = "unavailable_remote_or_missing_requester"
                tracker.state.unresolved_issues.append({"kind": "api_boundary_unavailable",
                    "detail": "Runtime cannot intercept API calls inside this environment mode."})
            trace.append({"task_id": world.task_id, "event": "execution_assurance_initialized",
                          "mode": tracker.state.mode, "goal_count": len(tracker.state.goals),
                          "plan_valid": tracker.state.plan_valid,
                          "api_boundary_coverage": runtime["api_boundary_coverage"],
                          "destructive_precondition_count": len(tracker.state.destructive_preconditions)})
        runtime.update(observation_store=observation_store, working_memory=working_memory,
                       task_state=task_state, context_builder=context_builder, tool=tool)
        reflection_v1_enabled = self.assurance_mode == "reflection_v1"
        reflection_enabled = self.assurance_mode in {"reflection_v1", "reflection_v2"}
        reflection = (TerminalReflection(self.model, trace, world.task_id, world.task.instruction)
                      if reflection_v1_enabled else
                      TerminalAudit(self.model, trace, world.task_id, world.task.instruction)
                      if self.assurance_mode == "reflection_v2" else None)
        if reflection:
            runtime["terminal_reflection"] = reflection

        def same_target_readback(correction: dict[str, Any], requests: Any,
                                 outcome: dict[str, Any]) -> bool:
            if reflection_enabled:
                return _same_target_readback(
                    correction, requests, outcome=outcome,
                    request_events=tool.last_assurance_events)
            return _same_target_readback(correction, requests)

        def run_buffer_guard(state: AgentState) -> Any:
            if reflection and reflection.buffer_guard_done:
                return reflection.buffer_guard_result
            check = runtime["completion_guard"].review(
                runtime["lite_evidence"],
                unresolved_error=_has_unresolved_business_error(runtime),
                batch_uncertain=bool(runtime.get("lite_unverified_batch")),
                batch_gap=bool(runtime.get("lite_batch_gap")),
                assurance_state=runtime.get("assurance_state"))
            reflection.buffer_guard_done, reflection.buffer_guard_result = True, check
            trace.append({"task_id": state["task_id"], "step": state.get("step_count", 0),
                          "event": "completion_guard", "trigger": "reserved_step_buffer",
                          "status": check.status, "allowed": check.allow,
                          "attempt": check.attempt, "observed": check.observed,
                          "missing": check.missing})
            return check

        if runtime.get("lite"):
            schemas = ({doc.qualified_name: doc.official_doc for doc in tool.api_retriever.documents}
                       if tool.api_retriever else {})
            runtime["lite_request_parser"] = tool.request_parser or RequesterEventParser(schemas)
            runtime["completion_guard"] = CompletionGuard(world.task.instruction)
            runtime["lite_evidence"] = []
        if self.assurance_mode == "reflection_v2":
            runtime["audit_ledger"] = []

        def review_terminal(step: int, trigger: str, *, pending_submission: str | None = None):
            if self.assurance_mode == "reflection_v2":
                return reflection.review(step, trigger, pending_submission=pending_submission,
                                         ledger=runtime.get("audit_ledger", []))
            return reflection.review(step, trigger, pending_submission=pending_submission,
                                     requester_boundary=runtime.get("requester_boundary"))

        def agent_node(state: AgentState) -> dict[str, Any]:
            if state.get("step_count", 0) >= self.max_steps:
                return {"finished": True, "stop_reason": "max_steps_reached",
                        "pending_tool_calls": [], "total_usage": state.get("total_usage", _zero_usage()),
                        "usage": _zero_usage()}
            started = perf_counter()
            timestamp = datetime.now(timezone.utc).isoformat()
            step = state.get("step_count", 0) + 1
            response = ""
            usage = _zero_usage()
            assistant_message: dict[str, Any] = {}
            pending_tool_calls: list[dict[str, Any]] = []
            protocol_error = None
            try:
                response_model = self.model
                if not hasattr(response_model, "generate_tool_calls"):
                    raise TypeError("Selected model adapter does not support Function Calling.")
                model_messages = state["messages"]
                available_tools = tool.registry.definitions
                if reflection and reflection.probe_pending:
                    available_tools = [definition for definition in available_tools
                                       if definition.get("function", {}).get("name") == "execute_python"]
                if context_builder is not None:
                    assurance_context = (runtime["goal_tracker"].render_context()
                                         if runtime.get("goal_tracker") is not None else None)
                    if isinstance(context_builder, UnifiedContextBuilder):
                        model_messages, view_stats = context_builder.build(
                            model_messages, goal_handoff=runtime.get("goal_handoff"),
                            active_correction=runtime.get("active_correction"),
                            assurance_context=assurance_context)
                    elif isinstance(context_builder, CacheFirstContextBuilder):
                        volatile_parts = []
                        if assurance_context:
                            volatile_parts.append(assurance_context)
                        handoff = runtime.get("goal_handoff")
                        if handoff:
                            volatile_parts.append("CURRENT GOAL HANDOFF (dynamic reference):\n" +
                                                  json.dumps(handoff, ensure_ascii=False, default=str))
                        correction = runtime.get("active_correction")
                        if correction:
                            correction_fields = {key: correction.get(key) for key in (
                                "category", "api_name", "reason", "observation_id", "attempts",
                                "uncertain_write", "official_schema") if key in correction}
                            volatile_parts.append("ACTIVE EXECUTOR ERROR (dynamic, not a new result):\n" +
                                                  json.dumps(correction_fields, ensure_ascii=False, default=str))
                        model_messages, view_stats = context_builder.build(
                            model_messages, step=step, max_steps=self.max_steps,
                            usage=state.get("usage"), max_cost_usd=self.max_cost_usd,
                            max_input_tokens=self.max_input_tokens,
                            max_output_tokens=self.max_output_tokens,
                            volatile_context="\n\n".join(volatile_parts) or None,
                            tool_definitions=available_tools)
                    else:
                        model_messages, view_stats = context_builder.build(model_messages)
                        if assurance_context:
                            model_messages = [*model_messages[:2],
                                {"role": "user", "content": assurance_context}, *model_messages[2:]]
                            view_stats["assurance_status_bytes"] = len(assurance_context.encode("utf-8"))
                    trace.append({"task_id": state["task_id"], "step": step,
                                  "event": "context_view", "visible_message_bytes": view_stats["visible_bytes"],
                                  "offloaded_count": len(view_stats["offloaded_ids"]),
                                  **({"unified": {key: value for key, value in view_stats.items()
                                                 if key != "offloaded_ids"}}
                                     if self.context_mode == "unified" else {}),
                                  **({"cache_first": {key: value for key, value in view_stats.items()
                                                      if key != "offloaded_ids"}}
                                     if self.context_mode == "cache_first" else {}),
                                  "timestamp": timestamp})
                    if (self.context_mode == "cache_first" and
                            view_stats.get("estimated_tokens", 0) >
                            view_stats.get("hard_capacity_tokens", 20000)):
                        trace.append({"task_id": state["task_id"], "step": step,
                                      "event": "context_capacity_stop",
                                      "estimated_tokens": view_stats["estimated_tokens"],
                                      "hard_capacity_tokens": view_stats.get("hard_capacity_tokens", 20000),
                                      "compact_level": view_stats.get("compact_level", 0)})
                        return {"step_count": state.get("step_count", 0), "finished": True,
                                "stop_reason": "context_capacity_exceeded",
                                "pending_tool_calls": [], "total_usage": state.get("total_usage", _zero_usage()),
                                "usage": _zero_usage()}
                elif runtime.get("goal_tracker") is not None:
                    assurance_context = runtime["goal_tracker"].render_context()
                    model_messages = [*model_messages[:2],
                        {"role": "user", "content": assurance_context}, *model_messages[2:]]
                overflow_recoveries = runtime.setdefault("overflow_recoveries", 0)
                for overflow_attempt in range(3):
                    try:
                        assistant_message, usage = response_model.generate_tool_calls(
                            model_messages, available_tools, tool_choice=self.tool_choice)
                        break
                    except SIWCContextOverflowError:
                        if not isinstance(context_builder, CacheFirstContextBuilder) or overflow_attempt >= 2:
                            raise
                        context_builder.force_overflow_compaction(state["messages"])
                        model_messages, view_stats = context_builder.build(
                            state["messages"], step=step, max_steps=self.max_steps,
                            usage=state.get("usage"), max_cost_usd=self.max_cost_usd,
                            max_input_tokens=self.max_input_tokens,max_output_tokens=self.max_output_tokens,
                            volatile_context="\n\n".join(volatile_parts) or None,
                            tool_definitions=available_tools)
                        overflow_recoveries += 1
                        runtime["overflow_recoveries"] = overflow_recoveries
                        trace.append({"task_id":state["task_id"],"step":step,
                            "event":"overflow_recovery","attempt":overflow_attempt+1,
                            "estimated_tokens":view_stats.get("estimated_tokens"),
                            "compact_level":view_stats.get("compact_level")})
                transport = getattr(response_model, "last_transport", None)
                if isinstance(transport, dict):
                    trace.append({"task_id": state["task_id"], "step": step,
                                  "event": "adapter_transport", **transport})
                pending_tool_calls = assistant_message.get("tool_calls") or []
                response = assistant_message.get("content") or ""
                if not pending_tool_calls or any(
                    not isinstance(call, dict)
                    or not isinstance(call.get("id"), str)
                    or not isinstance(call.get("function"), dict)
                    or not isinstance(call["function"].get("name"), str)
                    or not isinstance(call["function"].get("arguments"), str)
                    for call in pending_tool_calls
                ):
                    protocol_error = "The model did not return a valid tool call with an id, name, and JSON arguments."

            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                budget_blocked = type(exc).__name__ == "BudgetLimitReached"
                trace.append({"task_id": state["task_id"], "step": step,
                              "event": "budget_stop" if budget_blocked else "agent_error",
                              "model_response": response,
                              "usage": usage, "error": error,
                              **({"adapter_diagnostic": exc.diagnostic}
                                 if isinstance(getattr(exc, "diagnostic", None), dict) else {}),
                              "timestamp": timestamp})
                return {"step_count": step, "model_response": "", "error": error,
                        "finished": True,
                        "stop_reason": ("global_pre_request_budget_exhausted" if budget_blocked else
                                        "model_adapter_error" if isinstance(exc, Exception) and
                                        any(cls.__name__ == "SIWCError" for cls in type(exc).__mro__) else
                                        "model_or_format_error"),
                        "usage": usage,
                        "total_usage": _add_usage(state.get("total_usage", _zero_usage()), usage), "step_started_at": timestamp,
                        "step_duration_seconds": perf_counter() - started}

            total = _add_usage(state.get("total_usage", _zero_usage()), usage)
            stop_reason = None
            if total["cost_usd"] >= self.max_cost_usd:
                stop_reason = "cost_budget_exceeded"
            elif self.max_input_tokens is not None and total["input_tokens"] >= self.max_input_tokens:
                stop_reason = "input_token_budget_exceeded"
            elif total["output_tokens"] >= self.max_output_tokens:
                stop_reason = "output_token_budget_exceeded"
            if stop_reason:
                trace.append({"task_id": state["task_id"], "step": step, "event": "budget_stop", "model_response": response,
                              "usage": usage, "total_usage": total, "stop_reason": stop_reason, "timestamp": timestamp})
                return {"step_count": step, "model_response": response, "usage": usage,
                        "total_usage": total, "finished": True, "stop_reason": stop_reason,
                        "step_started_at": timestamp, "step_duration_seconds": perf_counter() - started}
            if protocol_error:
                retry_count = state.get("protocol_retry_count", 0)
                retry = retry_count < self.max_protocol_retries
                trace.append({"task_id": state["task_id"], "step": step, "event": "protocol_error",
                              "model_response": assistant_message,
                              "usage": usage, "error": protocol_error,
                              "retry_count": retry_count, "retry_allowed": retry, "timestamp": timestamp})
                return {"step_count": step, "model_response": response, "assistant_message": assistant_message,
                        "pending_tool_calls": [], "protocol_error": protocol_error,
                        "protocol_retry_count": retry_count + 1, "usage": usage, "total_usage": total,
                        "finished": not retry, "stop_reason": None if retry else "invalid_tool_protocol",
                        "step_started_at": timestamp, "step_duration_seconds": perf_counter() - started}

            return {"step_count": step, "model_response": response, "assistant_message": assistant_message,
                    "pending_tool_calls": pending_tool_calls, "protocol_error": None,
                    "usage": usage, "total_usage": total, "error": None, "finished": False,
                    "stop_reason": None, "step_started_at": timestamp, "step_started_perf": started}

        def execute_node(state: AgentState) -> dict[str, Any]:
            assistant_message = state.get("assistant_message") or {"role": "assistant", "content": state.get("model_response")}
            calls = state.get("pending_tool_calls") or []
            if not calls:
                assistant_message = {key: value for key, value in assistant_message.items() if key != "tool_calls"}
            messages = [*state["messages"], assistant_message]
            results = []
            python_count = state.get("python_execution_count", 0)
            repeated_error = False
            new_business_evidence = False
            new_tool_information = False
            error_recovered = False
            execution_total = state.get("total_usage", _zero_usage())
            probe_action = bool(reflection and reflection.probe_pending)
            reflection_triggered_this_step = False
            tool.last_operation_fingerprints = []
            for call in calls:
                function = call["function"]
                tool.last_assurance_events = []
                tool.assurance_call_id = None
                capture = runtime.get("lite_request_parser") is not None and function["name"] == "execute_python"
                requester = getattr(world, "requester", None) if capture else None
                requests_before = getattr(requester, "requests", None)
                request_start = len(requests_before) if isinstance(requests_before, list) else 0
                correction = runtime.get("active_correction") if runtime.get("lite") else None
                action_digest, _ = _failed_python_action(function["arguments"])
                blocked_reason = None
                completion_blocked = False
                if (reflection and reflection.authorized_completion
                        and function["name"] != "execute_python"):
                    reflection.authorized_completion = False
                if probe_action and function["name"] != "execute_python":
                    blocked_reason = "Reflection probe accepts only execute_python with a read-only AppWorld request."
                if correction and function["name"] == "execute_python":
                    if (action_digest == correction["code_digest"]
                            and not runtime.get("correction_information_changed")):
                        blocked_reason = ("Identical failed Python action blocked; change the call or gather new information. "
                                          f"Prior state: {correction.get('execution_state', 'UNKNOWN')}.")
                    elif correction.get("uncertain_write"):
                        try:
                            proposed_code = json.loads(function["arguments"]).get("code", "")
                        except (TypeError, ValueError, AttributeError):
                            proposed_code = ""
                        if not _safe_during_uncertain_write(proposed_code, correction, tool.api_retriever):
                            blocked_reason = ("A prior write outcome is uncertain. This action could repeat or "
                                              "conflict with that write. First use a documented readback, a safe "
                                              "authentication recovery, or an operation in a clearly separate app; "
                                              "no Python code was executed.")
                if runtime.get("lite") and function["name"] == "execute_python" and not blocked_reason:
                    try:
                        proposed_code = json.loads(function["arguments"]).get("code", "")
                    except (TypeError, ValueError, AttributeError):
                        proposed_code = ""
                    completion_form = AppWorldTool._completion_form(proposed_code)
                    if (reflection and reflection.authorized_completion
                            and completion_form not in {"direct", "print"}):
                        reflection.authorized_completion = False
                    if completion_form == "invalid":
                        blocked_reason = ("Completion protocol error: after writes and readback, submit only "
                                          "apis.supervisor.complete_task(answer=...) as a single statement "
                                          "(one print wrapper is allowed). Use a plain answer value without extra units "
                                          "unless requested, or answer=None for action-only tasks. No submission was executed.")
                        completion_blocked = True
                        trace.append({"task_id": state["task_id"], "step": state["step_count"],
                                      "event": "completion_protocol_rejected",
                                      "tool_call_id": call["id"], "reason": "non_standalone_completion"})
                    elif completion_form in {"direct", "print"}:
                        if isinstance(context_builder, CacheFirstContextBuilder):
                            context_builder.close_extraction()
                        reflection_triggered_this_step = True
                        if probe_action:
                            blocked_reason = ("A completion call cannot serve as the uncertainty probe. "
                                              "Use one documented read-only query first; no submission was executed.")
                            completion_blocked = True
                        elif reflection and reflection.authorized_completion:
                            reflection.authorized_completion = False
                            trace.append({"task_id": state["task_id"], "step": state["step_count"],
                                          "event": "terminal_reflection_route",
                                          "route": "consume_post_probe_completion_authorization"})
                        elif reflection:
                            if isinstance(context_builder, CacheFirstContextBuilder):
                                context_builder.close_extraction()
                            decision, route = review_terminal(state.get("step_count", 0), "submission",
                                                              pending_submission=proposed_code)
                            execution_total = _add_usage(execution_total, reflection.last_usage)
                            if route == "allow_existing_checks":
                                reflection.authorized_completion = True
                            elif route == "repair_budget_exhausted":
                                reflection.terminal_blocked = False
                                reflection.authorized_completion = True
                                trace.append({"task_id": state["task_id"], "step": state["step_count"],
                                              "event": "repair_budget_exhausted_allow_submit",
                                              "status": decision["status"],
                                              "repair_rounds": reflection.repair_rounds})
                            else:
                                completion_blocked = True
                                if route == "one_read_only_probe":
                                    blocked_reason = read_only_probe_feedback(decision)
                                elif route == "one_bounded_repair":
                                    blocked_reason = (audit_feedback(decision)
                                                      if self.assurance_mode == "reflection_v2"
                                                      else repair_feedback(decision))
                                else:
                                    reflection.terminal_blocked = True
                                    blocked_reason = (repair_feedback(decision) +
                                                      "\nNo further Reflection repair or probe is allowed; task was not submitted.")
                        if not blocked_reason:
                            check = runtime["completion_guard"].review(
                                runtime["lite_evidence"],
                                unresolved_error=_has_unresolved_business_error(runtime),
                                batch_uncertain=bool(runtime.get("lite_unverified_batch")),
                                batch_gap=bool(runtime.get("lite_batch_gap")),
                                assurance_state=runtime.get("assurance_state"))
                            trace.append({"task_id": state["task_id"], "step": state["step_count"],
                                          "event": "completion_guard", "status": check.status,
                                          "allowed": check.allow, "attempt": check.attempt,
                                          "observed": check.observed, "missing": check.missing,
                                          "goal_status": ({goal_id: goal.status.value for goal_id, goal in
                                              runtime["assurance_state"].goals.items()}
                                              if runtime.get("assurance_state") is not None else None)})
                            if not check.allow:
                                blocked_reason = ("Completion check failed.\nUser requirement: " +
                                                  world.task.instruction[:500] + "\n" + check.feedback())
                                completion_blocked = True
                                if check.attempt > runtime["completion_guard"].max_repairs:
                                    runtime["completion_guard_exhausted"] = True
                if blocked_reason:
                    outcome = {"tool_call_id": call["id"], "name": function["name"],
                               "ok": False, "validation": "blocked", "error": blocked_reason}
                    if runtime.get("lite") and not completion_blocked:
                        block_signature = hashlib.sha256(json.dumps(
                            [action_digest, blocked_reason], ensure_ascii=False).encode()).hexdigest()
                        previous_block = runtime.get("last_local_block_signature")
                        if previous_block == block_signature and not runtime.get("correction_information_changed"):
                            runtime["repeated_local_block_count"] = runtime.get("repeated_local_block_count", 1) + 1
                            repeated_error = True
                        else:
                            runtime["repeated_local_block_count"] = 1
                            runtime["last_local_block_signature"] = block_signature
                        if runtime.get("lite"):
                            runtime["lite_repeat_blocks"] = runtime["repeated_local_block_count"]
                        runtime["correction_information_changed"] = False
                        trace.append({"task_id": state["task_id"], "step": state["step_count"],
                                      "event": "local_recovery_block",
                                      "source": "runtime_policy",
                                      "execution_state": (correction or {}).get("execution_state", "UNKNOWN"),
                                      "action_digest": action_digest,
                                      "repeat_count": runtime["repeated_local_block_count"],
                                      "same_block_repeated": repeated_error,
                                      "python_executed": False})
                elif world.task_completed():
                    outcome = {"tool_call_id": call["id"], "name": function["name"],
                               "ok": False, "validation": "not_dispatched",
                               "error": "Skipped because AppWorld already signaled task completion."}
                else:
                    if function["name"] == "execute_python" and runtime.get("assurance_state") is not None:
                        tool.set_assurance_action(state["step_count"], call["id"])
                    outcome = tool.registry.dispatch(name=function["name"],
                                                    raw_arguments=function["arguments"],
                                                    tool_call_id=call["id"])
                    if runtime.get("lite") and outcome.get("ok") is True:
                        runtime["lite_repeat_blocks"] = 0
                        runtime["repeated_local_block_count"] = 0
                        runtime["last_local_block_signature"] = None
                boundary = runtime.get("requester_boundary") if reflection_enabled else None
                if boundary is not None:
                    for blocked in boundary.drain_blocked_attempts():
                        trace.append({"task_id": state["task_id"], "step": state["step_count"],
                                      "event": "reflection_read_only_block", **blocked})
                executed_requests = None
                if capture:
                    requests_after = getattr(requester, "requests", None)
                    if (isinstance(requests_before, list) and requests_after is requests_before
                            and len(requests_after) >= request_start):
                        parser = runtime["lite_request_parser"]
                        executed_requests = parser.request_evidence(requests_after[request_start:])
                    if runtime.get("lite") and executed_requests:
                        evidence_base = len(runtime["lite_evidence"])
                        runtime["lite_evidence"].extend({
                            "step": state["step_count"], "method": row.get("method"),
                            "api_name": row.get("api_name"), "ok": outcome.get("ok") is True,
                            "order": evidence_base + index,
                            "empty_result": (str(outcome.get("result") or "").strip() in {"[]", "{}"}
                                             if row.get("method") in {"GET", "HEAD"} and
                                             len(executed_requests) == 1 else False),
                        } for index, row in enumerate(executed_requests))
                        writes = [row for row in executed_requests
                                  if row.get("method") not in {"GET", "HEAD"} and
                                  not str(row.get("api_name") or "").endswith(".login") and
                                  not str(row.get("api_name") or "").endswith(".complete_task")]
                        if len(writes) > 1:
                            runtime["lite_unverified_batch"] = True
                            outcome["batch_verification"] = {
                                "writes_observed": len(writes), "status": "UNVERIFIED",
                                "next_step": ("Check each target's result and read back the final state; "
                                              "outer execute_python success is not proof of full batch success.")}
                        elif runtime.get("lite_unverified_batch") and any(
                                row.get("method") in {"GET", "HEAD"} for row in executed_requests):
                            try:
                                receipt_code = json.loads(function["arguments"]).get("code", "")
                            except (TypeError, ValueError, AttributeError):
                                receipt_code = ""
                            receipt = (batch_set_difference(str(outcome.get("result") or ""))
                                       if "AGENTLAB_BATCH_CHECK=" in receipt_code else None)
                            if receipt is not None:
                                runtime["lite_batch_gap"] = not receipt["match"]
                                outcome["batch_verification"] = {
                                    **receipt, "status": "REPORTED_MATCH" if receipt["match"] else "MISSING",
                                    "source": "post-write AppWorld readback output"}
                                runtime["lite_unverified_batch"] = not receipt["match"]
                if self.assurance_mode == "reflection_v2" and function["name"] == "execute_python":
                    failure = ((outcome.get("execution_summary") or {}).get("first_failure") or {})
                    for event in tool.last_assurance_events:
                        operation = str(getattr(getattr(event, "operation_type", None), "value",
                                                getattr(event, "operation_type", "")))
                        if operation not in {"READ", "WRITE", "DESTRUCTIVE"}:
                            continue
                        status = getattr(event, "execution_status", "unknown")
                        if operation == "READ" and status != "succeeded":
                            continue
                        api_name = getattr(event, "api_name", "unknown")
                        http_status = getattr(event, "http_status", None)
                        if http_status is None and failure.get("endpoint") == api_name:
                            http_status = failure.get("status_code")
                        runtime["audit_ledger"].append({
                            "step": getattr(event, "step", state["step_count"]),
                            "app.endpoint": api_name,
                            "operation": operation, "status": status,
                            "http_status": http_status,
                            "key_parameters": getattr(event, "arguments", {}),
                            "evidence_ref": getattr(event, "result_reference", None) or call["id"],
                            "readback_summary": (str(outcome.get("result") or "")[:240]
                                                  if operation == "READ" else ""),
                        })
                    if outcome.get("ok") is True:
                        if runtime.get("lite") and executed_requests:
                            error_recovered |= bool(runtime.get("failure_counts"))
                            runtime.setdefault("failure_counts", {}).clear()
                    if (outcome.get("ok") is True and executed_requests
                            and not outcome.get("execution_error")
                            and outcome.get("result") not in (None, "", "Execution successful.")):
                        # Repeating the same query and result is not new evidence.
                        # Only retain a digest in memory; observations may contain credentials.
                        evidence = json.dumps([executed_requests, outcome.get("result")],
                                              ensure_ascii=False, sort_keys=True, default=str)
                        digest = hashlib.sha256(evidence.encode("utf-8")).hexdigest()
                        seen = runtime.setdefault("business_evidence_digests", set())
                        if digest not in seen:
                            seen.add(digest)
                            new_business_evidence = True
                    empty_success = executed_requests == () and outcome.get("result") == "Execution successful."
                    runtime["empty_execution_streak"] = (
                        runtime.get("empty_execution_streak", 0) + 1 if empty_success else 0)
                if reflection_v1_enabled and function["name"] == "execute_python":
                    hint = pagination_hints(tool.last_assurance_events, tool.api_retriever)
                    if hint:
                        outcome["result"] = f"{outcome.get('result') or ''}\n{hint}"
                if function["name"] == "execute_python" and outcome.get("validation") == "accepted":
                    python_count += 1
                results.append(outcome)
                error_text = outcome.get("execution_error") or outcome.get("error")
                runtime_policy_block = bool(
                    function["name"] == "execute_python" and
                    (error_text == "write_replay_blocked" or any(
                        getattr(event, "gate_blocked", False) for event in tool.last_assurance_events)))
                new_tool_information = False
                if runtime.get("lite") and outcome.get("ok") is True:
                    digest = hashlib.sha256(json.dumps(
                        [function["name"], outcome.get("result")], ensure_ascii=False,
                        sort_keys=True, default=str).encode("utf-8")).hexdigest()
                    known = runtime.setdefault("known_tool_result_digests", set())
                    new_tool_information = digest not in known
                    known.add(digest)
                if (runtime.get("lite") and function["name"] == "execute_python"
                        and isinstance(error_text, str) and error_text
                        and outcome.get("validation") == "accepted" and not runtime_policy_block):
                    match = re.search(r"([A-Za-z_]+Error|[A-Za-z_]+Exception|[a-z_]+_error)", error_text)
                    category = match.group(1) if match else error_text[:120]
                    failure_counts = runtime.setdefault("failure_counts", {})
                    failure_counts[category] = failure_counts.get(category, 0) + 1
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps(outcome, ensure_ascii=False, default=str)})
                if function["name"] in tool.registry.tool_names:
                    messages[-1]["name"] = function["name"]
                if observation_store is not None:
                    record = observation_store.add(
                        tool_call_id=call["id"], tool_name=function["name"],
                        step=state["step_count"], content=messages[-1]["content"],
                        arguments=function["arguments"], outcome=outcome,
                        executed_requests=executed_requests)
                    if isinstance(context_builder, CacheFirstContextBuilder):
                        context_builder.record_request_events(call["id"], tool.last_assurance_events)
                    if function["name"] == "execute_python" and runtime.get("assurance_state") is not None:
                        linked = tool.link_assurance_observation(call["id"], record.id)
                        for event in linked:
                            trace.append({"task_id": state["task_id"], "step": event.step,
                                "event": "execution_assurance_event", **event.as_dict()})
                    if reflection_v1_enabled and function["name"] == "execute_python":
                        stdout = str(outcome.get("result") or "")
                        error_output = str(outcome.get("execution_error") or outcome.get("error") or "")
                        reflection.python_outputs.append({
                            "step": state["step_count"], "stdout": stdout[:700],
                            "error": error_output[:350]})
                        for event in tool.last_assurance_events:
                            reflection.actions.append({
                                "step": event.step, "api_name": event.api_name,
                                "arguments": event.arguments,
                                "status": event.execution_status,
                                "status_code": event.http_status,
                                "evidence_ref": event.result_reference or record.id,
                                "result_summary": stdout or error_output,
                            })
                    if runtime.get("lite"):
                        outcome["observation_id"] = record.id
                        messages[-1]["content"] = json.dumps(outcome, ensure_ascii=False, default=str)
                        if runtime_policy_block:
                            reason_code = ("write_replay_blocked" if error_text == "write_replay_blocked"
                                           else "action_gate_blocked")
                            block_signature = hashlib.sha256(json.dumps(
                                [reason_code, action_digest], ensure_ascii=False).encode()).hexdigest()
                            previous_block = runtime.get("last_policy_block_signature")
                            same_block = previous_block == block_signature
                            count = runtime.get("policy_block_count", 0) + 1 if same_block else 1
                            runtime["last_policy_block_signature"] = block_signature
                            runtime["policy_block_count"] = count
                            if runtime.get("lite"):
                                runtime["lite_repeat_blocks"] = count
                            repeated_error |= same_block
                            runtime["correction_information_changed"] = False
                            trace.append({"task_id": state["task_id"], "step": state["step_count"],
                                          "event": "local_recovery_block", "source": "runtime_policy",
                                          "reason": reason_code, "execution_state": "NOT_EXECUTED",
                                          "action_digest": action_digest, "repeat_count": count,
                                          "same_block_repeated": same_block, "python_executed": False})
                        elif (function["name"] == "execute_python" and error_text
                                and outcome.get("validation") == "accepted"):
                            previous = runtime.get("active_correction")
                            current = _local_correction(error_text, raw_arguments=function["arguments"],
                                                        executed_requests=executed_requests,
                                                        retriever=tool.api_retriever,
                                                        request_events=tool.last_assurance_events)
                            current.update(task_id=state["task_id"], observation_id=record.id,
                                           step=state["step_count"], goal_id=state.get("active_goal_id"))
                            if previous and (previous["category"], previous["api_name"],
                                             previous["code_digest"]) == (
                                    current["category"], current["api_name"], current["code_digest"]):
                                current["attempts"] = previous["attempts"] + 1
                                repeated_error |= not runtime.get("correction_information_changed")
                            runtime["active_correction"] = current
                            runtime["correction_information_changed"] = False
                            if _guard_blocking_correction(current):
                                runtime.setdefault("business_error_ledger", []).append(current)
                            trace.append({"task_id": state["task_id"], "step": state["step_count"],
                                          "event": "executor_local_error", "category": current["category"],
                                          "api_name": current["api_name"], "observation_id": record.id,
                                          "execution_state": current["execution_state"],
                                          "request_outcomes": current["request_outcomes"],
                                          "request_events": current["request_events"],
                                          "api_attempts": current["api_attempts"],
                                          "uncertain_write": current["uncertain_write"],
                                          "error_source": current["error_source"],
                                          "business_write_error": current["business_write_error"],
                                          "attempts": current["attempts"]})
                        elif outcome.get("ok") is True and runtime.get("active_correction"):
                            active_error = runtime["active_correction"]
                            for pending_error in runtime.get("business_error_ledger", []):
                                if not pending_error.get("resolved") and _correction_recovered(
                                        pending_error, function_name=function["name"],
                                        raw_arguments=function["arguments"], outcome=outcome,
                                        executed_requests=executed_requests):
                                    pending_error["resolved"] = True
                                    pending_error["resolution"] = "matching_api_success"
                                    trace.append({"task_id": state["task_id"], "step": state["step_count"],
                                                  "event": "business_error_resolved",
                                                  "failed_observation_id": pending_error.get("observation_id"),
                                                  "resolved_by_observation_id": record.id,
                                                  "resolution": "matching_api_success"})
                                elif not pending_error.get("resolved") and same_target_readback(
                                        pending_error, executed_requests, outcome):
                                    pending_error["uncertain_write"] = False
                                    pending_error["business_write_error"] = False
                                    pending_error["resolved"] = True
                                    pending_error["resolution"] = "same_target_readback_observed"
                                    trace.append({"task_id": state["task_id"], "step": state["step_count"],
                                                  "event": "business_error_resolved",
                                                  "failed_observation_id": pending_error.get("observation_id"),
                                                  "resolved_by_observation_id": record.id,
                                                  "resolution": "same_target_readback_observed"})
                            if _correction_recovered(active_error, function_name=function["name"],
                                                     raw_arguments=function["arguments"],
                                                     outcome=outcome, executed_requests=executed_requests):
                                runtime["active_correction"] = None
                                trace.append({"task_id": state["task_id"], "step": state["step_count"],
                                              "event": "executor_local_error_resolved",
                                              "observation_id": record.id,
                                              "resolution": "matching_api_success"})
                            elif same_target_readback(active_error, executed_requests, outcome):
                                active_error["uncertain_write"] = False
                                active_error["business_write_error"] = False
                                active_error["resolution"] = "readback_observed"
                                for pending_error in runtime.get("business_error_ledger", []):
                                    if pending_error.get("observation_id") == active_error.get("observation_id"):
                                        pending_error["uncertain_write"] = False
                                        pending_error["business_write_error"] = False
                                        pending_error["resolved"] = True
                                        pending_error["resolution"] = "same_target_readback_observed"
                                active_error["reason"] = ("A read-only request for the same target was observed. "
                                                          "Inspect its returned state before deciding whether a write is needed.")
                                runtime["correction_information_changed"] = True
                            else:
                                runtime["correction_information_changed"] |= new_tool_information
                        elif outcome.get("ok") is True and runtime.get("business_error_ledger"):
                            for pending_error in runtime["business_error_ledger"]:
                                if not pending_error.get("resolved") and _correction_recovered(
                                        pending_error, function_name=function["name"],
                                        raw_arguments=function["arguments"], outcome=outcome,
                                        executed_requests=executed_requests):
                                    pending_error["resolved"] = True
                                    pending_error["resolution"] = "matching_api_success"
                                    trace.append({"task_id": state["task_id"], "step": state["step_count"],
                                                  "event": "business_error_resolved",
                                                  "failed_observation_id": pending_error.get("observation_id"),
                                                  "resolved_by_observation_id": record.id,
                                                  "resolution": "matching_api_success"})
                    trace.append({"task_id": state["task_id"], "step": state["step_count"],
                                  "event": "observation_ref", "observation_id": record.id,
                                  "tool_call_id": call["id"], "source_tool": function["name"],
                                  "size_bytes": record.size_bytes, "api_names": record.api_names})
                    if (working_memory is not None and self.unified_config.memory
                            and function["name"] == "execute_python"):
                        extracted = working_memory.ingest(record.id, outcome, function["arguments"])
                        if extracted:
                            trace.append({"task_id": state["task_id"], "step": state["step_count"],
                                          "event": "memory_auto_extract", "count": len(extracted),
                                          "memory_ids": [row["memory_id"] for row in extracted],
                                          "source_observation_id": record.id})
                trace.append({"task_id": state["task_id"], "step": state["step_count"],
                              "event": "tool_call", "tool_call_id": call["id"],
                              "tool_name": function["name"], "arguments": function["arguments"],
                              "validation": outcome.get("validation"), "tool_result": outcome,
                              **({"request_outcomes": [
                                  {key: (row.as_dict() if hasattr(row, "as_dict") else row).get(key)
                                   for key in ("api_name", "execution_status", "http_status", "reason")}
                                  for row in tool.last_assurance_events]}
                                 if function["name"] == "execute_python" and tool.last_assurance_events else {}),
                              **({"executed_requests": executed_requests} if capture else {}),
                              "timestamp": datetime.now(timezone.utc).isoformat()})
            if state.get("protocol_error"):
                if not assistant_message.get("tool_calls"):
                    feedback = (
                        f"Tool protocol error: {state['protocol_error']}\n"
                        "The runtime rejects plain text and Python that only prints an explanation. Choose exactly one: "
                        "(1) call a registered tool to continue pagination or verify remaining data; "
                        "(2) if you have confirmed no more data is available, call execute_python with only "
                        "apis.supervisor.complete_task(...) to submit the best result. Return a valid tool call now."
                    )
                else:
                    feedback = f"Tool protocol error: {state['protocol_error']} Return one valid registered tool call."
                messages.append({"role": "user", "content": feedback})
            observation = json.dumps(results, ensure_ascii=False, default=str) if calls else (state.get("protocol_error") or "No tool call was returned.")
            completed = bool(world.task_completed())
            if reflection and probe_action:
                if isinstance(context_builder, CacheFirstContextBuilder):
                    context_builder.close_extraction()
                reflection.probe_pending = False
                decision, route = review_terminal(state.get("step_count", 0), "post_probe")
                execution_total = _add_usage(execution_total, reflection.last_usage)
                boundary_hook = runtime.get("requester_boundary")
                if boundary_hook is not None:
                    boundary_hook.set_read_only(False)
                if route == "allow_existing_checks":
                    reflection.authorized_completion = True
                    messages.append({"role": "user", "content": (
                        "The single read-only uncertainty probe has been reviewed and Reflection returned PASS. "
                        "You may submit if the task is complete; the existing Completion Guard still applies.")})
                elif route == "one_bounded_repair":
                    messages.append({"role": "user", "content": repair_feedback(decision)})
                elif route == "one_read_only_probe":
                    messages.append({"role": "user", "content": read_only_probe_feedback(decision)})
                else:
                    reflection.terminal_blocked = False
                    reflection.authorized_completion = True
                    trace.append({"task_id": state["task_id"], "step": state["step_count"],
                                  "event": "repair_budget_exhausted_allow_submit",
                                  "status": decision["status"],
                                  "repair_rounds": reflection.repair_rounds,
                                  "trigger": "post_probe", "route": route})
                    messages.append({"role": "user", "content": (
                        "The Reflection repair budget is exhausted after the read-only probe. "
                        "If the task is complete, submit now; the existing Completion Guard still applies. "
                        "Do not repeat the probe or repeat a side effect.")})
                if reflection.buffer_guard_pending:
                    check = run_buffer_guard(state)
                    reflection.buffer_guard_pending = False
                    if not check.allow:
                        messages.append({"role": "user", "content": check.feedback()})
            no_progress_stop = False
            if runtime.get("lite"):
                if new_business_evidence or error_recovered or new_tool_information:
                    runtime["no_progress_streak"] = 0
                else:
                    runtime["no_progress_streak"] = runtime.get("no_progress_streak", 0) + 1
                no_progress_stop = runtime["no_progress_streak"] >= self.max_no_progress_steps
                if no_progress_stop:
                    trace.append({"task_id": state["task_id"], "step": state["step_count"],
                                  "event": "no_progress_stop", "streak": runtime["no_progress_streak"],
                                  "limit": self.max_no_progress_steps,
                                  "reason": "No new distinct business evidence, tool information, or error recovery."})
            buffer_threshold = max(1, self.max_steps - 15)
            if (reflection_v1_enabled and reflection and not completed and not reflection_triggered_this_step
                    and not reflection.buffer_done
                    and state.get("step_count", 0) >= buffer_threshold):
                reflection.buffer_done = True
                if isinstance(context_builder, CacheFirstContextBuilder):
                    context_builder.close_extraction()
                decision, route = review_terminal(state.get("step_count", 0), "reserved_step_buffer")
                execution_total = _add_usage(execution_total, reflection.last_usage)
                feedback = []
                if route == "one_read_only_probe":
                    feedback.append(read_only_probe_feedback(decision))
                    reflection.buffer_guard_pending = True
                else:
                    if route == "one_bounded_repair":
                        feedback.append(repair_feedback(decision))
                    elif route == "repair_budget_exhausted":
                        reflection.terminal_blocked = False
                        reflection.authorized_completion = True
                        trace.append({"task_id": state["task_id"], "step": state["step_count"],
                                      "event": "repair_budget_exhausted_allow_submit",
                                      "status": decision["status"],
                                      "repair_rounds": reflection.repair_rounds,
                                      "trigger": "reserved_step_buffer", "route": route})
                        feedback.append("Reflection repair budget is exhausted. Submit if the task is complete; "
                                        "the existing Completion Guard still applies.")
                    elif route == "allow_existing_checks":
                        feedback.append("Reserved-step evidence review passed; continue only if a user requirement remains, then submit.")
                    check = run_buffer_guard(state)
                    if not check.allow:
                        feedback.append(check.feedback())
                if feedback:
                    messages.append({"role": "user", "content": "\n\n".join(feedback)})
            boundary = (no_progress_stop or runtime.get("lite_repeat_blocks", 0) >= 2 or
                        runtime.get("completion_guard_exhausted") or
                (reflection and reflection.terminal_blocked)) if runtime.get("lite") else False
            reason = "appworld_task_completed" if completed else (
                "no_progress_streak" if no_progress_stop else
                "reflection_repair_exhausted" if reflection and reflection.terminal_blocked else
                "completion_guard_exhausted" if runtime.get("completion_guard_exhausted") else
                "repeated_failed_action" if runtime.get("lite") and boundary else
                "max_steps_reached" if state.get("step_count", 0) >= self.max_steps else None
            )
            trace.append({"task_id": state["task_id"], "step": state["step_count"],
                          "event": "step", "assistant_message": assistant_message, "observation": observation,
                          "usage": state.get("usage", _zero_usage()),
                          "total_usage": execution_total,
                          "tool_call_count": len(calls), "python_execution_count": python_count,
                          "started_at": state.get("step_started_at"),
                          "ended_at": datetime.now(timezone.utc).isoformat(), "stop_reason": reason})
            return {"messages": messages, "pending_tool_calls": [],
                    "observation": observation, "tool_call_count": state.get("tool_call_count", 0) + len(calls),
                    "python_execution_count": python_count,
                    "total_usage": execution_total,
                    "finished": completed or boundary or reason == "max_steps_reached",
                    "stop_reason": reason}


        graph = StateGraph(AgentState)
        graph.add_node("agent_node", agent_node)
        graph.add_node("execute_node", execute_node)
        graph.add_edge(START, "agent_node")
        graph.add_conditional_edges("agent_node", lambda s: "end" if s.get("finished") else (
            "execute" if (s.get("pending_tool_calls") or s.get("protocol_error")) else "end"
        ),
                                    {"end": END, "execute": "execute_node"})
        graph.add_conditional_edges("execute_node", lambda s: "end" if s.get("finished") else "agent",
                                    {"end": END, "agent": "agent_node"})
        return graph.compile()


    def solve(self, world: Any, *, trace_path: Path) -> AgentState:
        task_id = world.task_id
        api_retriever = None
        if (self.enable_api_retrieval and hasattr(world.task, "api_docs")
                and hasattr(world.task, "allowed_apps")):
            api_retriever = ApiRetriever(world.task.api_docs, world.task.allowed_apps,
                                         schema_aware=self.schema_aware_retrieval)
        context = ContextManager(task_id, world.task.instruction,
                                 schema_aware_retrieval=self.schema_aware_retrieval,
                                 programmatic_execution=self.programmatic_execution,
                                 context_mode=self.context_mode,
                                 unified_config=self.unified_config if self.context_mode == "unified" else None,
                                 assurance_mode=self.assurance_mode,
                                 app_catalog=app_capability_catalog(api_retriever)
                                 if api_retriever else None)
        trace = TraceLogger(trace_path)
        planner_usage = _zero_usage()
        planner_calls = 0
        plan = None
        if self.plan_guided_react:
            plan, planner_usage, plan_error = one_shot_plan(self.model, world.task.instruction)
            planner_calls = 1
            trace.append({"task_id": task_id, "event": "one_shot_plan", "plan": plan,
                          "fallback_reason": plan_error, "usage": planner_usage})
            if plan is not None:
                context.messages[1]["content"] += ("\nAdvisory high-level plan (original task and "
                    "actual tool results take precedence):\n" + json.dumps(plan, ensure_ascii=False))
        initial: AgentState = {"task_id": task_id, "messages": context.messages, "step_count": 0,
                               "finished": False, "error": None,
                               "observation": "", "stop_reason": None, "usage": _zero_usage(),
                               "total_usage": planner_usage, "model_response": "",
                               "pending_tool_calls": [], "protocol_retry_count": 0,
                               "tool_call_count": 0, "python_execution_count": 0}
        run_runtime = {"api_retriever": api_retriever} if api_retriever else None
        if self.max_input_tokens is not None and planner_usage.get("input_tokens", 0) >= self.max_input_tokens:
            result = {**initial, "stop_reason": "input_token_budget_exceeded", "finished": True}
        elif planner_usage.get("cost_usd", 0) >= self.max_cost_usd or planner_usage.get("output_tokens", 0) >= self.max_output_tokens:
            result = {**initial, "stop_reason": "planner_budget_exceeded", "finished": True}
        else:
            if self.plan_guided_react:
                run_runtime = {**(run_runtime or {}), "lite": True, "assurance_enabled": True,
                               "assurance_plan": plan, "assurance_mode": self.assurance_mode,
                               "plan": plan}
            graph = self.build_graph(world, trace, runtime=run_runtime)
            try:
                result = graph.invoke(initial,
                    config={"recursion_limit": max(25, self.max_steps * 3)})
            finally:
                boundary_hook = (run_runtime or {}).get("requester_boundary")
                if boundary_hook is not None:
                    boundary_hook.restore()
        if not result.get("stop_reason"):
            result["stop_reason"] = "max_steps_reached" if result.get("step_count", 0) >= self.max_steps else "graph_ended"
        trace.write_summary(trace_path.with_name("summary.json"), {
            "task_id": task_id, "model": getattr(self.model, "model_name", type(self.model).__name__),
            "context_mode": self.context_mode,
            "efficient_tool_use": {"schema_aware_retrieval": self.schema_aware_retrieval,
                                   "programmatic_execution": self.programmatic_execution},
            "unified_config": self.unified_config.__dict__ if self.context_mode == "unified" else None,
            "architecture": "plan_guided_react" if self.plan_guided_react else "react",
            "plan": plan, "planner_calls": planner_calls,
            **({"terminal_reflection": {
                "mode": self.assurance_mode,
                **((run_runtime or {})["terminal_reflection"].summary()
                   if (run_runtime or {}).get("terminal_reflection") else {}),
            }} if self.assurance_mode in {"reflection_v1", "reflection_v2"} else {}),
            "execution_assurance": (run_runtime["assurance_state"].snapshot()
                                     if self.plan_guided_react and (run_runtime or {}).get("assurance_state") is not None
                                     else None),
            "api_boundary_coverage": ((run_runtime or {}).get("api_boundary_coverage")
                                      if self.plan_guided_react else None),
            "steps": result.get("step_count", 0), "llm_request_count": (
                result.get("step_count", 0) + planner_calls +
                ((run_runtime or {}).get("terminal_reflection").calls
                 if (run_runtime or {}).get("terminal_reflection") else 0)),
            "async_extraction": ((run_runtime or {}).get("context_builder").extraction_summary()
                if isinstance((run_runtime or {}).get("context_builder"), CacheFirstContextBuilder)
                else None),
            "overflow_recoveries": (run_runtime or {}).get("overflow_recoveries", 0),
            "stop_reason": result.get("stop_reason"),
            "task_completed_signal": bool(world.task_completed()), "error": result.get("error"),
            "total_usage": result.get("total_usage", _zero_usage()),
            "tool_call_count": result.get("tool_call_count", 0),
            "python_execution_count": result.get("python_execution_count", 0),
            "limits": {"max_steps": self.max_steps, "max_cost_usd": self.max_cost_usd,
                       "max_input_tokens": self.max_input_tokens, "max_output_tokens": self.max_output_tokens},
        })
        return result
