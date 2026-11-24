from __future__ import annotations

import pytest

from agentlab.execution_assurance.assurance import (
    ActionGateBlocked, GoalStatus, OperationType, OutcomeGate, RequesterBoundary,
    GoalTracker, RequesterEventParser, classify_operation,
)
from agentlab.execution_assurance.completion_guard import CompletionGuard
from agentlab.environments.appworld import AppWorldTool


class FakeRequester:
    def request(self, _app_name, _api_name, **kwargs):
        if _api_name == "fail_timeout":
            raise TimeoutError("read timed out")
        if _api_name == "list_records":
            page = kwargs.get("page_index", 0)
            return ([{"id": 1}, {"id": 2}] if page == 0 else [{"id": 3}])
        if _api_name == "show_file":
            return {"path": kwargs.get("path"), "content": "Title,Artists\nSong,Artist\n"}
        return {"ok": True, **kwargs}


def make_tracker(*, mode="shadow", plan=None):
    instruction = "Export all rows to a valid CSV before deleting the account."
    plan = plan or {
        "completion_checks": ["CSV is valid", "Account is deleted"],
        "goals": [
            {"id": "Gsrc", "description": "Collect all source rows",
             "source_requirement": "Export all rows", "constraints": {},
             "dependencies": [], "completion_criteria": ["All pages are collected"],
             },
            {"id": "Gcsv", "description": "Read back and validate the CSV backup",
             "source_requirement": "valid CSV", "constraints": {},
             "dependencies": ["Gsrc"], "completion_criteria": ["CSV is valid"]},
            {"id": "Gdel", "description": "Delete the account",
             "source_requirement": "deleting the account", "constraints": {},
             "dependencies": ["Gcsv"], "completion_criteria": ["Account is deleted"]},
        ],
        "destructive_preconditions": ["Gsrc", "Gcsv"],
    }
    tracker = GoalTracker.from_plan("task-1", instruction, plan, mode=mode)
    metadata = {
        "music.list_records": {"method": "GET", "parameters": [
            {"name": "page_index", "default": 0}, {"name": "page_limit", "default": 2}]},
        "files.show_file": {"method": "GET", "parameters": []},
        "admin.delete_account": {"method": "DELETE", "parameters": []},
        "music.fail_timeout": {"method": "POST", "parameters": []},
        "music.set_preference": {"method": "POST", "parameters": []},
    }
    gate = OutcomeGate(tracker.state, tracker, metadata)
    return tracker, gate, metadata


def run_action(boundary, gate, requester, step, call_id, api_calls):
    gate.begin_action(step, call_id)
    try:
        result = [requester.request(_app_name=app, _api_name=api, **args)
                  for app, api, args in api_calls]
        return result
    finally:
        gate.end_action(call_id, None)


def test_goal_tracker_preserves_source_and_does_not_trust_model_status():
    tracker, _gate, _metadata = make_tracker()
    assert all(goal.status == GoalStatus.PENDING for goal in tracker.state.goals.values())
    assert tracker.state.goals["Gsrc"].source_requirement_matched


def test_invalid_goal_dependencies_fall_back_without_losing_original_instruction():
    plan = {"completion_checks": ["Preserve the user's request"], "goals": [
        {"id": "G1", "description": "broken", "source_requirement": "absent phrase",
         "constraints": {}, "dependencies": ["G404"], "completion_criteria": ["check"]}]}
    tracker = GoalTracker.from_plan("task-1", "Original requirement", plan)
    assert not tracker.state.plan_valid
    assert tracker.state.goals["G1"].source_requirement == "Original requirement"
    assert tracker.state.unresolved_issues[0]["kind"] == "invalid_planner_goal_contract"


def test_goal_source_must_be_exact_original_task_text():
    plan = {"completion_checks": ["Keep task requirements"], "goals": [
        {"id": "G1", "description": "not grounded", "source_requirement": "invented requirement",
         "constraints": {}, "dependencies": [], "completion_criteria": ["check"]}]}
    tracker = GoalTracker.from_plan("task-1", "Original requirement", plan)
    assert not tracker.state.plan_valid
    assert tracker.state.goals["G1"].source_requirement == "Original requirement"
    assert tracker.state.goals["G1"].source_requirement_matched


def test_classification_uses_official_method_metadata_only():
    assert classify_operation("thing.delete", {"method": "GET"}) == OperationType.READ
    assert classify_operation("thing.lookup", {"method": "DELETE"}) == OperationType.DESTRUCTIVE
    assert classify_operation("thing.pay", {}) == OperationType.UNKNOWN


def test_shadow_mode_records_would_block_but_does_not_change_business_execution():
    tracker, gate, metadata = make_tracker(mode="shadow")
    requester = FakeRequester()
    hook = RequesterBoundary(requester, gate); hook.install()
    try:
        run_action(hook, gate, requester, 1, "call-1", [("admin", "delete_account", {})])
    finally:
        hook.restore()
    event = tracker.state.events[0]
    assert event.execution_status == "succeeded"
    # metadata lookup used by the fixture; no state transition is inferred from the call.
    decision = gate.action_gate.check("admin.delete_account")
    assert decision["would_block"] and not decision["blocked"]


def test_enforce_blocks_destructive_call_until_explicit_preconditions_verified():
    tracker, gate, _metadata = make_tracker(mode="enforce")
    requester = FakeRequester(); hook = RequesterBoundary(requester, gate); hook.install()
    gate.begin_action(1, "blocked-delete")
    with pytest.raises(ActionGateBlocked):
        requester.request(_app_name="admin", _api_name="delete_account")
    gate.end_action("blocked-delete", "obs-blocked")
    assert tracker.state.events[-1].execution_status == "blocked"
    assert tracker.state.goals["Gdel"].status == GoalStatus.PENDING
    assert any(issue["kind"] == "action_gate_precondition" and issue["blocked"]
               for issue in tracker.state.unresolved_issues)
    for goal_id in ("Gsrc", "Gcsv"):
        tracker.state.goals[goal_id].status = GoalStatus.VERIFIED
    gate.begin_action(2, "allowed-delete")
    assert requester.request(_app_name="admin", _api_name="delete_account")["ok"]
    gate.end_action("allowed-delete", "obs-delete")
    hook.restore()
    assert tracker.state.events[-1].execution_status == "succeeded"


def test_enforce_blocks_destructive_call_when_planner_only_provided_fallback_checks():
    tracker = GoalTracker.from_plan("task-1", "Delete the selected account.",
                                    {"completion_checks": ["Account removed"]}, mode="enforce")
    gate = OutcomeGate(tracker.state, tracker, {"admin.delete_account": {"method": "DELETE"}})
    decision = gate.action_gate.check("admin.delete_account")
    assert not tracker.state.plan_valid
    assert decision["blocked"]
    assert decision["missing_preconditions"] == ["valid_structured_plan"]


def test_completion_guard_reads_goal_state_and_does_not_keep_stale_gate_blocker():
    tracker, _gate, _metadata = make_tracker(mode="enforce")
    guard = CompletionGuard(tracker.state.instruction)
    evidence = [{"ok": True, "api_name": "admin.delete_account", "method": "DELETE", "step": 2}]
    blocked = guard.review(evidence, unresolved_error=False, assurance_state=tracker.state)
    assert not blocked.allow and blocked.status == "MISSING"
    for goal_id in ("Gsrc", "Gcsv"):
        tracker.state.goals[goal_id].status = GoalStatus.VERIFIED
    allowed = guard.review(evidence, unresolved_error=False, assurance_state=tracker.state)
    assert allowed.allow


def test_partial_batch_records_success_and_failure_per_api_call():
    class PartialRequester(FakeRequester):
        def request(self, _app_name, _api_name, **kwargs):
            if kwargs.get("user_id") == 2:
                raise ValueError("invalid target")
            return {"ok": True, **kwargs}

    tracker, gate, _metadata = make_tracker()
    requester = PartialRequester(); hook = RequesterBoundary(requester, gate); hook.install()
    gate.begin_action(5, "partial-batch")
    try:
        requester.request(_app_name="music", _api_name="set_preference", user_id=1)
        with pytest.raises(ValueError):
            requester.request(_app_name="music", _api_name="set_preference", user_id=2)
    finally:
        gate.end_action("partial-batch", "obs-partial")
        hook.restore()
    assert [event.execution_status for event in tracker.state.events] == ["succeeded", "failed"]
    assert all(event.result_reference == "obs-partial" for event in tracker.state.events)


def test_redacts_bare_id_token_and_unknown_metadata_is_recorded_conservatively():
    tracker, gate, _metadata = make_tracker()
    requester = FakeRequester(); hook = RequesterBoundary(requester, gate); hook.install()
    run_action(hook, gate, requester, 1, "token", [
        ("music", "unlisted", {"id_token": "sensitive-value", "record_id": 4})])
    hook.restore()
    event = tracker.state.events[-1]
    assert event.operation_type == OperationType.UNKNOWN
    assert event.arguments["id_token"] == "[REDACTED]"
    assert event.arguments["record_id"] == 4


def test_enforce_refuses_python_when_requester_boundary_is_unavailable():
    class BareWorld:
        task = type("Task", (), {"api_docs": {}, "allowed_apps": []})()
        def __init__(self): self.calls = []
        def execute(self, code): self.calls.append(code); return "executed"

    tracker, gate, _metadata = make_tracker(mode="enforce")
    world = BareWorld()
    tool = AppWorldTool(world, enable_api_retrieval=False, outcome_gate=gate)
    observation, error = tool.execute_python("mutate()")
    assert "Requester interception boundary" in observation
    assert error
    assert world.calls == []


def test_multiple_internal_calls_are_each_recorded_and_credentials_redacted():
    tracker, gate, _metadata = make_tracker()
    requester = FakeRequester(); hook = RequesterBoundary(requester, gate); hook.install()
    run_action(hook, gate, requester, 4, "batch", [
        ("music", "set_preference", {"user_id": 7, "access_token": "secret-token"}),
        ("music", "set_preference", {"user_id": 8, "password": "secret-password"}),
    ])
    hook.restore()
    assert len(tracker.state.events) == 2
    assert tracker.state.events[0].arguments["access_token"] == "[REDACTED]"
    assert tracker.state.events[1].arguments["password"] == "[REDACTED]"
    assert all(row.result_reference is None for row in tracker.state.events)
    assert all(row.step == 4 for row in tracker.state.events)


def test_write_timeout_is_unknown_and_unrelated_success_does_not_clear_it():
    tracker, gate, _metadata = make_tracker()
    requester = FakeRequester(); hook = RequesterBoundary(requester, gate); hook.install()
    with pytest.raises(TimeoutError):
        run_action(hook, gate, requester, 2, "timeout", [("music", "fail_timeout", {})])
    failed = tracker.state.events[-1]
    assert failed.execution_status == "unknown"
    run_action(hook, gate, requester, 3, "unrelated", [("music", "set_preference", {"user_id": 1})])
    assert tracker.state.events[-1].execution_status == "succeeded"
    assert any(issue.get("event_id") == failed.event_id for issue in tracker.state.unresolved_issues)
    hook.restore()


def test_no_llm_or_task_specific_shortcut_is_part_of_gate():
    tracker, gate, _metadata = make_tracker()
    assert gate.state is tracker.state
    assert all(goal_id not in {"60d0b5b_3", "6104387_1"} for goal_id in tracker.state.goals)


def test_request_parser_maps_real_calls_without_exposing_credentials():
    parser = RequesterEventParser({"music.change_record": {
        "method": "POST", "path": "/music/records/{record_id}"}})
    rows = parser.request_evidence([{
        "method": "POST", "url": "https://music.appworld.dev/music/records/17?access_token=secret&page_index=2",
        "data": {"value": "new"},
    }])
    assert len(rows) == 1
    assert rows[0]["api_name"] == "music.change_record"
    assert rows[0]["page_index"] == 2
    assert "access_token" not in rows[0]
    assert rows[0]["query_key"] and rows[0]["target_key"]
