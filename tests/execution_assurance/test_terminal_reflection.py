from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agent.react import LangGraphAgent
from execution_assurance.assurance import read_only_probe_allowed
from execution_assurance.reflection import TerminalReflection, read_only_probe_feedback


PLAN = {"phases": ["Perform the requested work", "Submit"],
        "constraints": [], "completion_checks": ["Complete the explicit task"]}


def _call(name, arguments, call_id):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": call_id, "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]}


def _reflection(status, *, issue="", evidence=None, next_action=""):
    return _call("submit_terminal_reflection", {
        "status": status, "issue": issue, "evidence": evidence or [],
        "next_action": next_action,
    }, "reflection")


def _action(code, call_id):
    return _call("execute_python", {"code": code}, call_id)


class MockModel:
    model_name = "offline-reflection-mock"

    def __init__(self, actions, reflections):
        self.actions = iter(actions)
        self.reflections = iter(reflections)
        self.requests = []
        self.last_transport = None

    def generate_tool_calls(self, messages, tools, *, tool_choice="required"):
        names = [row["function"]["name"] for row in tools]
        self.requests.append((messages, names))
        if names == ["submit_plan"]:
            response = _call("submit_plan", PLAN, "plan")
        elif names == ["submit_terminal_reflection"]:
            response = next(self.reflections)
        else:
            response = next(self.actions)
        return response, {"input_tokens": 5, "input_cache_hit_tokens": 1,
                          "output_tokens": 2, "cost_usd": 0.0}


class FakeRequester:
    def __init__(self):
        self.requests = []
        self.empty_list = False

    def request(self, app, api, **kwargs):
        self.requests.append({"app": app, "api": api, **kwargs})
        if api == "list_items":
            if self.empty_list or kwargs.get("page_index", 0) >= 1:
                return []
            return [{"item_id": item_id} for item_id in range(5)]
        return {"ok": True, "item_id": kwargs.get("item_id", 1)}


class FakeWorld:
    task_id = "reflection-fake-task"
    task = SimpleNamespace(
        instruction="Read the requested item and submit.",
        allowed_apps=["phone"],
        api_docs={"phone": {
            "show_item": {"method": "GET", "path": "/phone/items/{item_id}",
                          "description": "Read one item", "parameters": []},
            "search_items": {"method": "POST", "path": "/phone/items/search",
                             "description": "Search items", "parameters": []},
            "login": {"method": "POST", "path": "/phone/login",
                      "description": "Authenticate", "parameters": []},
            "delete_item": {"method": "DELETE", "path": "/phone/items/{item_id}",
                             "description": "Delete one item", "parameters": []},
            "list_items": {"method": "GET", "path": "/phone/items",
                           "description": "List items by page.", "parameters": [
                               {"name": "page_index", "type": "integer", "required": False},
                               {"name": "page_limit", "type": "integer", "required": False},
                           ]},
        }},
    )

    def __init__(self):
        self.requester = FakeRequester()
        self.finished = False
        self.executed = []
        self.fail_first_completion = False

    def execute(self, code):
        self.executed.append(code)
        if "complete_task" in code:
            if self.fail_first_completion:
                self.fail_first_completion = False
                raise ValueError("mock completion parameter error")
            self.finished = True
            return "Task completed"
        if code == "probe()":
            self.requester.request("phone", "show_item", item_id=1)
            return "Observed item 1"
        if code == "probe_write()":
            self.requester.request("phone", "delete_item", item_id=1)
            return "Unexpected write reached"
        if code == "page_zero()":
            return self.requester.request("phone", "list_items", page_index=0, page_limit=5)
        if code == "page_zero_then_one()":
            page_zero = self.requester.request("phone", "list_items", page_index=0, page_limit=5)
            page_one = self.requester.request("phone", "list_items", page_index=1, page_limit=5)
            return page_zero + page_one
        if code == "empty_page()":
            self.requester.empty_list = True
            return self.requester.request("phone", "list_items", page_index=0, page_limit=5)
        return "Action completed"

    def task_completed(self):
        return self.finished


def _run(tmp_path: Path, actions, reflections, *, max_steps=40, assurance_mode="reflection_v1"):
    model, world = MockModel(actions, reflections), FakeWorld()
    result = LangGraphAgent(
        model, max_steps=max_steps, plan_guided_react=True,
        assurance_mode=assurance_mode,
    ).solve(world, trace_path=tmp_path / "trace.jsonl")
    events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text(
        encoding="utf-8").splitlines()]
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    return model, world, result, events, summary


def test_reflection_pass_allows_existing_completion_guard(tmp_path):
    model, world, result, events, summary = _run(
        tmp_path, [_action("apis.supervisor.complete_task()", "finish")],
        [_reflection("PASS", evidence=["No objective gap in observed evidence."])])

    assert result["stop_reason"] == "appworld_task_completed"
    assert world.finished
    assert [event["event"] for event in events].count("terminal_reflection") == 1
    assert any(event.get("event") == "completion_guard" and event.get("allowed") for event in events)
    assert summary["terminal_reflection"]["calls"] == 1
    assert summary["llm_request_count"] == 3  # planner + executor + reflection
    assert summary["total_usage"]["input_tokens"] == 15
    # Reflection receives a short independent request, not the accumulated ReAct history.
    reflection_messages = next(messages for messages, names in model.requests
                               if names == ["submit_terminal_reflection"])
    assert len(reflection_messages) == 2
    assert "original_task" in reflection_messages[1]["content"]


def test_reflection_v1_deterministic_answer_precheck_has_separate_budget():
    class Model:
        def __init__(self): self.calls = 0
        def generate_tool_calls(self, *_args, **_kwargs):
            self.calls += 1
            return _reflection("MISSING", issue="One evidence gap remains."), {
                "input_tokens": 7, "input_cache_hit_tokens": 0, "output_tokens": 2, "cost_usd": 0.0}

    for answer in ("Unable to verify the total", "$12 USD"):
        class Trace:
            events = []
            def append(self, event): self.events.append(event)
        model, trace = Model(), Trace()
        reviewer = TerminalReflection(model, trace, "format-test", "What is the total amount?")
        code = f"apis.supervisor.complete_task(answer={answer!r})"
        decision, route = reviewer.review(1, "submission", pending_submission=code)
        assert decision["status"] == "MISSING" and route == "one_bounded_repair"
        assert trace.events[-1]["deterministic_precheck"] is True
        assert reviewer.format_repair_rounds == 1
        assert reviewer.repair_rounds == reviewer.calls == model.calls == 0
        if answer != "Unable to verify the total":
            continue
        # A later well-formed answer still gets its semantic Reflection opportunity.
        reviewer.instruction = "Read the item and finish the requested action."
        decision, route = reviewer.review(
            2, "submission", pending_submission="apis.supervisor.complete_task(answer=None)")
        assert model.calls == reviewer.calls == 1
        assert reviewer.repair_rounds == 1 and reviewer.format_repair_rounds == 1
        assert route == "one_bounded_repair" and decision["status"] == "MISSING"


def test_missing_and_conflict_each_get_one_repair_then_pass(tmp_path):
    for status in ("MISSING", "CONFLICT"):
        case_dir = tmp_path / status
        case_dir.mkdir()
        model, world, result, events, summary = _run(
            case_dir,
            [_action("apis.supervisor.complete_task()", "early"),
             _action("safe_action()", "repair"),
             _action("apis.supervisor.complete_task()", "finish")],
            [_reflection(status, issue="A concrete requested item is not evidenced.",
                         evidence=["task requirement: item cleanup"], next_action="Check and correct that item."),
             _reflection("PASS", evidence=["Repair evidence is now present."])],
        )
        assert result["stop_reason"] == "appworld_task_completed"
        assert world.executed == ["safe_action()", "apis.supervisor.complete_task()"]
        assert summary["terminal_reflection"]["repair_rounds"] == 1
        assert len([event for event in events if event.get("event") == "terminal_reflection"]) == 2
        assert any("bounded repair" in str(message.get("content", "")).lower()
                   for messages, names in model.requests if "submit_plan" not in names
                   and "submit_terminal_reflection" not in names
                   for message in messages)


def test_uncertain_gets_one_read_only_probe_and_second_adjudication(tmp_path):
    model, world, result, events, summary = _run(
        tmp_path,
        [_action("apis.supervisor.complete_task()", "early"),
         _action("probe()", "probe"),
         _action("apis.supervisor.complete_task()", "finish")],
        [_reflection("UNCERTAIN", issue="The item state needs a readback.",
                     next_action="Read the item state once."),
         _reflection("PASS", evidence=["phone.show_item", "obs_2"])],
    )
    assert result["stop_reason"] == "appworld_task_completed"
    assert world.requester.requests and world.requester.requests[0]["api"] == "show_item"
    probe_request = next((names for _messages, names in model.requests
                          if "submit_terminal_reflection" not in names
                          and "submit_plan" not in names and "search_api" not in names), None)
    assert probe_request == ["execute_python"]
    assert summary["terminal_reflection"]["probe_rounds"] == 1
    assert summary["terminal_reflection"]["repair_rounds"] == 0
    assert len([event for event in events if event.get("event") == "terminal_reflection"]) == 2
    assert summary["total_usage"]["input_tokens"] == 30


def test_probe_read_only_gate_allows_reads_and_blocks_writes(tmp_path):
    assert read_only_probe_allowed("GET", "phone.show_item")
    assert read_only_probe_allowed("POST", "phone.search_items")
    assert read_only_probe_allowed("POST", "phone.login")
    assert not read_only_probe_allowed("POST", "phone.create_item")
    assert not read_only_probe_allowed("DELETE", "phone.search_items")

    _, world, result, events, _ = _run(
        tmp_path,
        [_action("apis.supervisor.complete_task()", "early"),
         _action("probe_write()", "probe-write"),
         _action("apis.supervisor.complete_task()", "finish")],
        [_reflection("UNCERTAIN", issue="Check the item without changing it."),
         _reflection("PASS", evidence=["No write was executed during the probe."])],
    )
    assert result["stop_reason"] == "appworld_task_completed"
    assert world.requester.requests == []
    blocked = [event for event in events if event.get("event") == "reflection_read_only_block"]
    assert blocked and blocked[0]["api_name"] == "phone.delete_item"
    assert "Reflection probe is read-only" in str(events)


def test_probe_and_repair_budgets_are_hard_bounded(tmp_path):
    _, world, result, events, summary = _run(
        tmp_path,
        [_action("apis.supervisor.complete_task()", "early"),
         _action("probe()", "probe"),
         _action("safe_action()", "repair"),
         _action("apis.supervisor.complete_task()", "finish")],
        [_reflection("UNCERTAIN", issue="Needs a read."),
         _reflection("UNCERTAIN", issue="Still unsure; do not request another probe."),
         _reflection("PASS", evidence=["The repair step resolved the gap."])],
    )
    assert result["stop_reason"] == "appworld_task_completed"
    assert summary["terminal_reflection"]["probe_rounds"] == 1
    assert summary["terminal_reflection"]["repair_rounds"] == 1
    second = [event for event in events if event.get("event") == "terminal_reflection"][1]
    assert second["decision"]["status"] == "MISSING"
    assert len([event for event in events if event.get("event") == "terminal_reflection"]) == 3


def test_repeated_missing_allows_submit_after_one_repair(tmp_path):
    _, world, result, events, summary = _run(
        tmp_path,
        [_action("apis.supervisor.complete_task()", "early"),
         _action("safe_action()", "repair"),
         _action("apis.supervisor.complete_task()", "second-finish")],
        [_reflection("MISSING", issue="A requirement remains."),
         _reflection("MISSING", issue="The same requirement remains.")],
    )
    assert result["stop_reason"] == "appworld_task_completed"
    assert world.finished
    assert summary["terminal_reflection"]["repair_rounds"] == 1
    assert summary["terminal_reflection"]["calls"] == 2
    assert any(event.get("event") == "repair_budget_exhausted_allow_submit"
               and event.get("status") == "MISSING" for event in events)


def test_pass_is_not_repeated_after_adjacent_completion_retry(tmp_path):
    model = MockModel([
        _action('apis.supervisor.complete_task("bad answer")', "bad-finish"),
        _action("apis.supervisor.complete_task()", "retry-finish"),
    ], [_reflection("PASS", evidence=["No objective gap is visible."])])
    world = FakeWorld()
    world.fail_first_completion = True
    result = LangGraphAgent(model, max_steps=40, plan_guided_react=True,
                            assurance_mode="reflection_v1").solve(
        world, trace_path=tmp_path / "trace.jsonl")
    events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert result["stop_reason"] == "appworld_task_completed"
    assert world.finished
    assert len([event for event in events if event.get("event") == "terminal_reflection"]) == 1
    assert any(event.get("event") == "terminal_reflection_route"
               and event.get("route") == "consume_post_probe_completion_authorization"
               for event in events)


def test_post_probe_missing_repairs_when_available_and_allows_submit_when_exhausted(tmp_path):
    assert "every page" in read_only_probe_feedback({"status": "UNCERTAIN"})

    repair_dir = tmp_path / "repair_available"
    repair_dir.mkdir()
    _, world, result, events, summary = _run(
        repair_dir,
        [_action("apis.supervisor.complete_task()", "finish-1"),
         _action("probe()", "probe-1"), _action("probe_write()", "repair-1"),
         _action("apis.supervisor.complete_task()", "finish-2")],
        [_reflection("UNCERTAIN", issue="Read the current item."),
         _reflection("MISSING", issue="A required item still needs correction."),
         _reflection("PASS", evidence=["The correction and readback are recorded."])],
    )
    post_probe = next(event for event in events
                      if event.get("event") == "terminal_reflection"
                      and event.get("trigger") == "post_probe")
    first_reflection = next(event for event in events
                            if event.get("event") == "terminal_reflection")
    assert first_reflection["decision"]["status"] == "UNCERTAIN"
    assert first_reflection["probe_rounds"] == 1
    assert first_reflection["repair_rounds"] == 0
    assert post_probe["route"] == "one_bounded_repair"
    assert [request["api"] for request in world.requester.requests] == ["show_item", "delete_item"]
    assert result["stop_reason"] == "appworld_task_completed"
    assert summary["terminal_reflection"]["probe_rounds"] == 1
    assert summary["terminal_reflection"]["repair_rounds"] == 1


def test_reflection_hints_when_only_page_zero_returns_a_full_list(tmp_path):
    _, _world, result, events, _ = _run(
        tmp_path,
        [_action("page_zero()", "page-0"), _action("apis.supervisor.complete_task()", "finish")],
        [_reflection("PASS", evidence=["The task's list was checked."])],
    )
    assert result["stop_reason"] == "appworld_task_completed"
    call = next(event for event in events
                if event.get("event") == "tool_call" and event.get("step") == 1)
    assert "[Pagination Hint] phone.list_items returned 5 items on page 0" in call["tool_result"]["result"]


def test_pagination_hint_suppressed_after_page_one_empty_page_and_in_shadow(tmp_path):
    cases = [
        ("page_zero_then_one()", "reflection_v1"),
        ("empty_page()", "reflection_v1"),
        ("page_zero()", "shadow"),
    ]
    for index, (code, mode) in enumerate(cases):
        case_dir = tmp_path / str(index)
        case_dir.mkdir()
        reflections = ([_reflection("PASS", evidence=["The task is ready."])]
                       if mode == "reflection_v1" else [])
        _, _world, result, events, _ = _run(
            case_dir, [_action(code, "page-action"), _action("apis.supervisor.complete_task()", "finish")],
            reflections, assurance_mode=mode,
        )
        assert result["stop_reason"] == "appworld_task_completed"
        calls = [event for event in events if event.get("event") == "tool_call"]
        assert not any("[Pagination Hint]" in str(event.get("tool_result")) for event in calls)

    exhausted_dir = tmp_path / "repair_exhausted"
    exhausted_dir.mkdir()
    _, world, result, events, summary = _run(
        exhausted_dir,
        [_action("apis.supervisor.complete_task()", "finish-1"),
         _action("safe_action()", "repair-1"),
         _action("apis.supervisor.complete_task()", "finish-2"),
         _action("probe()", "probe-1"),
         _action("apis.supervisor.complete_task()", "finish-3")],
        [_reflection("MISSING", issue="One concrete requirement is missing."),
         _reflection("UNCERTAIN", issue="Read the current item."),
         _reflection("MISSING", issue="The probe did not resolve the gap.")],
    )
    exhausted = next(event for event in events
                     if event.get("event") == "repair_budget_exhausted_allow_submit"
                     and event.get("trigger") == "post_probe")
    probe_reflection = next(event for event in events
                            if event.get("event") == "terminal_reflection"
                            and event.get("decision", {}).get("status") == "UNCERTAIN")
    assert probe_reflection["probe_rounds"] == 1
    assert probe_reflection["repair_rounds"] == 1
    assert exhausted["status"] == "MISSING"
    assert result["stop_reason"] == "appworld_task_completed"
    assert world.finished
    assert any(event.get("event") == "completion_guard" and event.get("allowed")
               for event in events)
    assert not any(event.get("stop_reason") == "reflection_repair_exhausted" for event in events)
    assert summary["terminal_reflection"]["probe_rounds"] == 1
    assert summary["terminal_reflection"]["repair_rounds"] == 1


def test_invalid_json_fails_open_but_keeps_completion_guard(tmp_path):
    invalid = _call("submit_terminal_reflection", {"bad": "json"}, "invalid")
    invalid["tool_calls"][0]["function"]["arguments"] = "{"
    _, world, result, events, summary = _run(
        tmp_path, [_action("apis.supervisor.complete_task()", "finish")], [invalid])
    reflection = next(event for event in events if event.get("event") == "terminal_reflection")
    assert result["stop_reason"] == "appworld_task_completed"
    assert world.finished
    assert reflection["fallback"] is True
    assert reflection["decision"]["status"] == "PASS"
    assert any(event.get("event") == "completion_guard" for event in events)
    assert summary["terminal_reflection"]["calls"] == 1


def test_reserved_step_buffer_runs_reflection_and_completion_guard(tmp_path):
    _, world, result, events, summary = _run(
        tmp_path,
        [_action("safe_action()", "work"), _action("apis.supervisor.complete_task()", "finish")],
        [_reflection("PASS", evidence=["No objective gap is visible."]),
         _reflection("PASS", evidence=["The task is ready to submit."])],
        max_steps=16,
    )
    assert result["stop_reason"] == "appworld_task_completed"
    assert world.finished
    assert any(event.get("event") == "terminal_reflection"
               and event.get("trigger") == "reserved_step_buffer" for event in events)
    assert any(event.get("event") == "completion_guard"
               and event.get("trigger") == "reserved_step_buffer" for event in events)
    assert summary["terminal_reflection"]["buffer_checkpoint_triggered"] is True


def test_shadow_does_not_issue_reflection_or_change_request_count(tmp_path):
    model = MockModel([_action("apis.supervisor.complete_task()", "finish")], [])
    world = FakeWorld()
    result = LangGraphAgent(model, plan_guided_react=True, assurance_mode="shadow",
                            max_steps=40).solve(world, trace_path=tmp_path / "trace.jsonl")
    events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text(
        encoding="utf-8").splitlines()]
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert result["stop_reason"] == "appworld_task_completed"
    assert len(model.requests) == 2  # planner + one ReAct call
    assert not any(event.get("event") == "terminal_reflection" for event in events)
    assert "terminal_reflection" not in summary
    assert summary["llm_request_count"] == 2


def test_reflection_mode_requires_task_local_plan_guidance():
    import pytest

    with pytest.raises(ValueError, match="plan_guided_react"):
        LangGraphAgent(MockModel([], []), assurance_mode="reflection_v1")
