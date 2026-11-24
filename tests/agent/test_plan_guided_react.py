import json
from pathlib import Path
from types import SimpleNamespace

from agentlab.agent.react import LangGraphAgent
from agentlab.environments.appworld import AppWorldTool
from agentlab.agent.prompt import ContextManager
from agentlab.execution_assurance.recovery import (_correction_recovered,
                           _guard_blocking_correction, _local_correction,
                           _same_target_readback)
from agentlab.execution_assurance.completion_guard import CompletionGuard, batch_set_difference
from agentlab.agent.planner import PLAN_TOOL, one_shot_plan
from agentlab.models.siwc_responses import SIWCTransportError
from agentlab.tool_engineering.tools import ToolRegistry
from agentlab.context_engineering.unified import UnifiedContextConfig


def action(name, arguments, call_id):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": call_id, "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments)}}]}


class Model:
    model_name = "offline-plan-react"

    def __init__(self, actions):
        self.actions = iter(actions)
        self.requests = []

    def generate_tool_calls(self, messages, tools, *, tool_choice="required"):
        self.requests.append((messages, [row["function"]["name"] for row in tools]))
        return next(self.actions), {"input_tokens": 9, "input_cache_hit_tokens": 2,
                                    "output_tokens": 3, "cost_usd": 0.0}


class World:
    task_id = "lite-fake-task"
    task = SimpleNamespace(instruction="Perform one safe action and submit.",
                           allowed_apps=[], api_docs={})

    def __init__(self):
        self.executed = []
        self.finished = False

    def execute(self, code):
        self.executed.append(code)
        if code == "bad()":
            return "Execution failed. TypeError: bad argument"
        if code == "apis.supervisor.complete_task()":
            self.finished = True
            return "Task completed"
        return "Business action succeeded"

    def task_completed(self):
        return self.finished


PLAN = {"phases": ["Perform the requested action", "Submit"],
        "constraints": ["Use tool feedback"], "completion_checks": ["Action succeeded"]}


def test_v3_runtime_omits_v31_verifier_and_v32b_plan_guidance():
    properties = PLAN_TOOL[0]["function"]["parameters"]["properties"]
    goal_properties = properties["goals"]["items"]["properties"]
    assert "verification_contract" not in goal_properties
    context = ContextManager("task", "Do the requested work.")
    assert "verify_goal" not in context.messages[0]["content"]
    registry = ToolRegistry(search_api=lambda **_: [], get_api_schema=lambda **_: {},
                            execute_python=lambda **_: "ok")
    assert "verify_goal" not in registry.tool_names
    assert not hasattr(UnifiedContextConfig(), "enhanced_plan_guidance")


def test_planner_propagates_siwc_transport_failure():
    class FailingModel:
        def generate_tool_calls(self, *_args, **_kwargs):
            raise SIWCTransportError(phase="awaiting_headers", error_class="ReadTimeout")

    import pytest

    with pytest.raises(SIWCTransportError):
        one_shot_plan(FailingModel(), "A task")


def test_one_shot_plan_and_react_recovery(tmp_path: Path):
    model = Model([
        action("submit_plan", PLAN, "plan"),
        action("execute_python", {"code": "bad()"}, "bad"),
        action("execute_python", {"code": "safe_action()"}, "good"),
        action("execute_python", {"code": "apis.supervisor.complete_task()"}, "finish"),
    ])
    world = World()
    result = LangGraphAgent(model, context_mode="unified",
                            plan_guided_react=True, max_steps=5).solve(
                                world, trace_path=tmp_path / "trace.jsonl")
    assert result["stop_reason"] == "appworld_task_completed"
    assert world.executed == ["bad()", "safe_action()", "apis.supervisor.complete_task()"]
    assert len(model.requests) == 4
    assert "Advisory high-level plan" in model.requests[1][0][1]["content"]
    assert "TypeError" in str(model.requests[2][0])
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["planner_calls"] == 1
    assert summary["llm_request_count"] == 4
    assert summary["architecture"] == "plan_guided_react"


def test_v3_shadow_plan_tool_assurance_and_completion_guard_smoke(tmp_path: Path):
    instruction = "Delete the selected message."
    plan = {
        "phases": ["Delete the selected message", "Submit"],
        "constraints": [],
        "completion_checks": [instruction],
        "goals": [{"id": "delete-message", "description": instruction,
                   "source_requirement": instruction, "constraints": {},
                   "dependencies": [], "completion_criteria": [instruction]}],
        "destructive_preconditions": [],
    }

    class Requester:
        def __init__(self):
            self.requests = []

        def request(self, _app, _api, *, method, url):
            self.requests.append({"method": method, "url": url})
            return {"status_code": 204 if method == "DELETE" else 200}

    class AssuredWorld(World):
        task = SimpleNamespace(instruction=instruction, allowed_apps=["phone"], api_docs={"phone": {
            "show_message": {"method": "GET", "path": "/phone/messages/{id}",
                             "description": "Read a message", "parameters": []},
            "delete_message": {"method": "DELETE", "path": "/phone/messages/{id}",
                               "description": "Delete the selected message", "parameters": []},
        }})

        def __init__(self):
            super().__init__()
            self.requester = Requester()

        def execute(self, code):
            if code == "delete()":
                self.requester.request("phone", "delete_message", method="DELETE",
                                       url="/phone/messages/7")
            elif code == "readback()":
                self.requester.request("phone", "show_message", method="GET",
                                       url="/phone/messages/7")
            return super().execute(code)

    model = Model([action("submit_plan", plan, "plan"),
                   action("execute_python", {"code": "delete()"}, "delete"),
                   action("execute_python", {"code": "readback()"}, "readback"),
                   action("execute_python", {"code": "apis.supervisor.complete_task()"}, "finish")])
    world = AssuredWorld()
    result = LangGraphAgent(model, context_mode="unified",
                            plan_guided_react=True, assurance_mode="shadow", max_steps=6).solve(
                                world, trace_path=tmp_path / "trace.jsonl")

    assert result["stop_reason"] == "appworld_task_completed"
    assert len(world.requester.requests) == 2
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["planner_calls"] == 1
    assert summary["execution_assurance"]["mode"] == "shadow"
    assert len(summary["execution_assurance"]["events"]) == 2
    rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row.get("event") == "execution_assurance_initialized" for row in rows)
    assert any(row.get("event") == "completion_guard" and row.get("status") == "SATISFIED"
               and row.get("allowed") for row in rows)
    delete_messages = model.requests[2][0]
    assert any(message.get("role") == "tool" and message.get("tool_call_id") == "delete"
               for message in delete_messages)
    assert any(message.get("role") == "assistant" and any(
        call.get("id") == "delete" for call in message.get("tool_calls", []))
        for message in delete_messages)


def test_repeated_failed_action_does_not_execute_again(tmp_path: Path):
    model = Model([
        action("submit_plan", PLAN, "plan"),
        action("execute_python", {"code": "bad()"}, "bad-1"),
        action("execute_python", {"code": "bad()"}, "bad-2"),
        action("execute_python", {"code": "bad()"}, "bad-3"),
    ])
    world = World()
    result = LangGraphAgent(model, context_mode="unified",
                            plan_guided_react=True, max_steps=10).solve(
                                world, trace_path=tmp_path / "trace.jsonl")
    assert result["stop_reason"] == "repeated_failed_action"
    assert world.executed == ["bad()"]


def test_invalid_plan_falls_back_to_react(tmp_path: Path):
    model = Model([action("unknown", {}, "invalid"),
                   action("execute_python", {"code": "apis.supervisor.complete_task()"}, "finish")])
    result = LangGraphAgent(model, context_mode="unified",
                            plan_guided_react=True, max_steps=2).solve(
                                World(), trace_path=tmp_path / "trace.jsonl")
    assert result["stop_reason"] == "appworld_task_completed"
    rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["fallback_reason"] == "invalid_plan_action"


def test_completion_guard_statuses_and_bounded_unknown():
    assert CompletionGuard("Make a list of my messages.").intent == ()
    assert CompletionGuard("Make payment requests for coworkers.").intent == ("create",)
    guard = CompletionGuard("Delete the selected messages.")
    assert guard.review([], unresolved_error=False).status == "MISSING"
    assert guard.review([], unresolved_error=False).allow is False
    last = guard.review([], unresolved_error=False)
    assert (last.status, last.allow) == ("MISSING", False)
    empty = [{"step": 1, "method": "GET", "api_name": "phone.search_messages",
              "ok": True, "empty_result": True}]
    no_op = CompletionGuard("Delete messages.")
    no_op.review(empty, unresolved_error=False)
    no_op.review(empty, unresolved_error=False)
    uncertain = no_op.review(empty, unresolved_error=False)
    assert (uncertain.status, uncertain.allow) == ("UNKNOWN", True)
    evidence = [
        {"step": 2, "method": "DELETE", "api_name": "phone.delete_message", "ok": True},
        {"step": 3, "method": "GET", "api_name": "phone.show_message", "ok": True},
    ]
    assert CompletionGuard("Delete messages.").review(evidence, unresolved_error=False).status == "SATISFIED"
    assert CompletionGuard("Delete messages.").review(evidence, unresolved_error=True).status == "MISSING"
    alarm_evidence = [
        {"step": 2, "order": 0, "method": "PATCH", "api_name": "phone.update_alarm", "ok": True},
        {"step": 2, "order": 1, "method": "GET", "api_name": "phone.show_alarms", "ok": True},
    ]
    renamed_action = CompletionGuard("Move the alarm one hour later.").review(
        alarm_evidence, unresolved_error=False)
    assert (renamed_action.status, renamed_action.allow) == ("UNKNOWN", True)
    batch_readback = CompletionGuard("Add songs to a playlist.").review(
        [{"step": 2, "order": 0, "method": "POST", "api_name": "spotify.add_song_to_playlist", "ok": True},
         {"step": 2, "order": 1, "method": "GET", "api_name": "spotify.show_playlist", "ok": True}],
        unresolved_error=False, batch_uncertain=True)
    assert (batch_readback.status, batch_readback.allow) == ("UNKNOWN", True)


def test_batch_set_difference_is_bounded_and_does_not_invent_success():
    receipt = batch_set_difference('AGENTLAB_BATCH_CHECK={"expected":["a","b"],"observed":["a","c"],"failed":["b"]}')
    assert receipt["match"] is False
    assert receipt["missing"] == ["b"]
    assert receipt["unexpected"] == ["c"]
    assert batch_set_difference("arbitrary tool output") is None


def test_lite_completion_is_blocked_until_write_and_readback(tmp_path: Path):
    class DeletionWorld(World):
        task = SimpleNamespace(instruction="Delete the selected message.", allowed_apps=["phone"],
                               api_docs={"phone": {
                                   "show_message": {"method": "GET", "path": "/phone/messages/{id}",
                                                    "description": "Read message", "parameters": []},
                                   "delete_message": {"method": "DELETE", "path": "/phone/messages/{id}",
                                                      "description": "Delete message", "parameters": []}}})

        def __init__(self):
            super().__init__()
            class Requester:
                def __init__(self):
                    self.requests = []

                def request(self, app, api, **request):
                    self.requests.append(request)
                    return {"status_code": 200}

            self.requester = Requester()

        def execute(self, code):
            if code == "query()":
                self.requester.request("phone", "show_message", method="GET", url="/phone/messages/7")
            elif code == "delete()":
                self.requester.request("phone", "delete_message", method="DELETE", url="/phone/messages/7")
            elif code == "readback()":
                self.requester.request("phone", "show_message", method="GET", url="/phone/messages/7")
            return super().execute(code)

    model = Model([action("submit_plan", PLAN, "plan"),
                   action("execute_python", {"code": "query()"}, "q"),
                   action("execute_python", {"code": "apis.supervisor.complete_task()"}, "early"),
                   action("execute_python", {"code": "delete()"}, "d"),
                   action("execute_python", {"code": "readback()"}, "r"),
                   action("execute_python", {"code": "apis.supervisor.complete_task()"}, "done")])
    world = DeletionWorld()
    result = LangGraphAgent(model, context_mode="unified",
                            plan_guided_react=True, max_steps=8).solve(
                                world, trace_path=tmp_path / "trace.jsonl")
    assert result["stop_reason"] == "appworld_task_completed"
    assert world.executed.count("apis.supervisor.complete_task()") == 1
    checks = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
              if '"event": "completion_guard"' in line]
    assert [(row["status"], row["allowed"]) for row in checks] == [
        ("MISSING", False), ("SATISFIED", True)]


def test_batch_mismatch_blocks_completion_until_corrected_readback(tmp_path: Path):
    class BatchWorld(World):
        task = SimpleNamespace(instruction="Move the selected files.", allowed_apps=["file_system"],
                               api_docs={"file_system": {
                                   "move_file": {"method": "POST", "path": "/fs/files/{id}",
                                                 "description": "Move file", "parameters": []},
                                   "show_file": {"method": "GET", "path": "/fs/files/{id}",
                                                 "description": "Read file", "parameters": []}}})

        def __init__(self):
            super().__init__()
            class Requester:
                def __init__(self):
                    self.requests = []

                def request(self, app, api, **request):
                    self.requests.append(request)
                    return {"status_code": 200}

            self.requester = Requester()

        def execute(self, code):
            if code == "batch()":
                self.requester.request("file_system", "move_file", method="POST", url="/fs/files/1")
                self.requester.request("file_system", "move_file", method="POST", url="/fs/files/2")
                self.executed.append(code)
                return "Batch submitted"
            if code.startswith("readback"):
                self.requester.request("file_system", "show_file", method="GET", url="/fs/files/1")
                self.executed.append(code)
                observed = '["1"]' if code.startswith("readback_bad") else '["1","2"]'
                failed = '["2"]' if code.startswith("readback_bad") else '[]'
                return ('AGENTLAB_BATCH_CHECK={"expected":["1","2"],"observed":' +
                        observed + ',"failed":' + failed + '}')
            return super().execute(code)

    model = Model([action("submit_plan", PLAN, "plan"),
                   action("execute_python", {"code": "batch()"}, "batch"),
                   action("execute_python", {"code": "readback_bad() # AGENTLAB_BATCH_CHECK="}, "bad-readback"),
                   action("execute_python", {"code": "apis.supervisor.complete_task()"}, "early"),
                   action("execute_python", {"code": "readback_good() # AGENTLAB_BATCH_CHECK="}, "good-readback"),
                   action("execute_python", {"code": "apis.supervisor.complete_task()"}, "done")])
    world = BatchWorld()
    result = LangGraphAgent(model, context_mode="unified",
                            plan_guided_react=True, max_steps=8).solve(
                                world, trace_path=tmp_path / "trace.jsonl")
    assert result["stop_reason"] == "appworld_task_completed"
    assert world.executed.count("apis.supervisor.complete_task()") == 1
    rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    checks = [row for row in rows if row.get("event") == "completion_guard"]
    assert [(row["status"], row["allowed"]) for row in checks] == [
        ("MISSING", False), ("SATISFIED", True)]
    tool_results = [row["tool_result"] for row in rows if row.get("event") == "tool_call"]
    assert any(result.get("batch_verification", {}).get("status") == "MISSING" for result in tool_results)


def test_missing_write_evidence_stops_after_bounded_repairs(tmp_path: Path):
    class MissingWorld(World):
        task = SimpleNamespace(instruction="Delete the selected messages.",
                               allowed_apps=[], api_docs={})

    model = Model([action("submit_plan", PLAN, "plan"),
                   action("execute_python", {"code": "apis.supervisor.complete_task()"}, "first"),
                   action("execute_python", {"code": "apis.supervisor.complete_task()"}, "second"),
                   action("execute_python", {"code": "apis.supervisor.complete_task()"}, "third")])
    world = MissingWorld()
    result = LangGraphAgent(model, context_mode="unified",
                            plan_guided_react=True, max_steps=8).solve(
                                world, trace_path=tmp_path / "trace.jsonl")
    assert result["stop_reason"] == "completion_guard_exhausted"
    assert world.executed == []


def test_rejected_completion_parameters_can_be_corrected(tmp_path: Path):
    class AnswerWorld(World):
        task = SimpleNamespace(instruction="How much money was received?",
                               allowed_apps=[], api_docs={})

        def execute(self, code):
            self.executed.append(code)
            if code == "apis.supervisor.complete_task('answer')":
                return ("Execution failed. TypeError: _api_request() takes 0 positional "
                        "arguments but 1 was given")
            if code == "apis.supervisor.complete_task(answer='answer')":
                self.finished = True
                return "Task completed"
            raise AssertionError(code)

    rejected = _local_correction("Unexpected parameter 'task_id' passed to complete_task",
                                 raw_arguments=json.dumps({"code": "apis.supervisor.complete_task(task_id='x')"}),
                                 executed_requests=({"method": "POST", "api_name": "supervisor.complete_task"},),
                                 retriever=None)
    assert rejected["category"] == "parameter_validation_error"
    assert rejected["uncertain_write"] is False
    model = Model([action("submit_plan", PLAN, "plan"),
                   action("execute_python", {"code": "apis.supervisor.complete_task('answer')"}, "wrong"),
                   action("execute_python", {"code": "apis.supervisor.complete_task(answer='answer')"}, "fixed")])
    world = AnswerWorld()
    result = LangGraphAgent(model, context_mode="unified",
                            plan_guided_react=True, max_steps=4).solve(
                                world, trace_path=tmp_path / "trace.jsonl")
    assert result["stop_reason"] == "appworld_task_completed"
    assert len(world.executed) == 2


def test_finish_protocol_accepts_only_direct_or_single_print_wrapper():
    world = World()
    for code in ("apis.supervisor.complete_task()",
                 "print(apis.supervisor.complete_task(answer='ok'))"):
        assert AppWorldTool._completion_form(code) in {"direct", "print"}
    assert world.executed == []

    rejected = (
        "print(print(apis.supervisor.complete_task()))",
        "print(apis.supervisor.complete_task(), 1)",
        "value = apis.supervisor.complete_task()",
        "finish = apis.supervisor.complete_task; finish()",
        "getattr(apis.supervisor, 'complete_task')()",
        "apis.supervisor.complete_task(); print('done')",
        "print(str(apis.supervisor.complete_task()))",
        "read_business_state(); apis.supervisor.complete_task()",
        "write_business_state(); apis.supervisor.complete_task()",
    )
    for code in rejected:
        assert AppWorldTool._completion_form(code) == "invalid"
    assert world.executed == []


def test_single_print_completion_passes_guard_and_submits_once(tmp_path: Path):
    class PrintWorld(World):
        def execute(self, code):
            self.executed.append(code)
            if code == "print(apis.supervisor.complete_task())":
                self.finished = True
                return "Task completed\n"
            return "Business action succeeded"

    model = Model([action("submit_plan", PLAN, "plan"),
                   action("execute_python", {"code": "print(apis.supervisor.complete_task())"}, "finish")])
    world = PrintWorld()
    result = LangGraphAgent(model, context_mode="unified",
                            plan_guided_react=True, max_steps=4).solve(
                                world, trace_path=tmp_path / "trace.jsonl")
    rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    guards = [row for row in rows if row.get("event") == "completion_guard"]
    assert result["stop_reason"] == "appworld_task_completed"
    assert [(row["status"], row["allowed"]) for row in guards] == [("SATISFIED", True)]
    assert world.executed == ["print(apis.supervisor.complete_task())"]


def test_duplicate_completion_calls_in_one_response_submit_once(tmp_path: Path):
    class TwoCallModel:
        model_name = "offline-two-call"
        def __init__(self):
            self.planned = False
        def generate_tool_calls(self, messages, tools, *, tool_choice="required"):
            names = {row["function"]["name"] for row in tools}
            if "submit_plan" in names and not self.planned:
                self.planned = True
                response = action("submit_plan", PLAN, "plan")
            else:
                response = {"role": "assistant", "content": None, "tool_calls": [
                    {"id": f"finish-{index}", "type": "function", "function": {
                        "name": "execute_python", "arguments": json.dumps({
                            "code": "apis.supervisor.complete_task()"})}}
                    for index in (1, 2)]}
            return response, {"input_tokens": 1, "input_cache_hit_tokens": 0,
                              "output_tokens": 1, "cost_usd": 0.0}

    world = World()
    result = LangGraphAgent(TwoCallModel(), context_mode="unified",
                            plan_guided_react=True, max_steps=3).solve(
                                world, trace_path=tmp_path / "trace.jsonl")
    assert result["stop_reason"] == "appworld_task_completed"
    assert world.executed == ["apis.supervisor.complete_task()"]


def test_mixed_completion_protocol_errors_do_not_consume_guard_budget(tmp_path: Path):
    class PrintWorld(World):
        def execute(self, code):
            self.executed.append(code)
            if code == "apis.supervisor.complete_task()":
                self.finished = True
                return "Task completed"
            return "Business action succeeded"

    model = Model([action("submit_plan", PLAN, "plan"),
                   action("execute_python", {"code": "read_business_state(); apis.supervisor.complete_task()"}, "bad-read"),
                   action("execute_python", {"code": "write_business_state(); apis.supervisor.complete_task()"}, "bad-write"),
                   action("execute_python", {"code": "apis.supervisor.complete_task()"}, "finish")])
    world = PrintWorld()
    result = LangGraphAgent(model, context_mode="unified",
                            plan_guided_react=True, max_steps=5).solve(
                                world, trace_path=tmp_path / "trace.jsonl")
    rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    guards = [row for row in rows if row.get("event") == "completion_guard"]
    rejected = [row for row in rows if row.get("event") == "completion_protocol_rejected"]
    assert result["stop_reason"] == "appworld_task_completed"
    assert len(rejected) == 2
    assert len(guards) == 1 and guards[0]["attempt"] == 1
    assert world.executed == ["apis.supervisor.complete_task()"]


def test_completion_argument_error_stays_on_official_execution_path(tmp_path: Path):
    class AnswerWorld(World):
        task = SimpleNamespace(instruction="What is the answer?", allowed_apps=[], api_docs={})

        def execute(self, code):
            self.executed.append(code)
            if "bad=" in code:
                return "Execution failed. TypeError: unexpected parameter 'bad'"
            if code == "print(apis.supervisor.complete_task(answer='ok'))":
                self.finished = True
                return "Task completed"
            return "Execution successful."

    world = AnswerWorld()
    model = Model([action("submit_plan", PLAN, "plan"),
                   action("execute_python", {"code": "apis.supervisor.complete_task(bad='x')"}, "wrong"),
                   action("execute_python", {"code": "print(apis.supervisor.complete_task(answer='ok'))"}, "fixed")])
    result = LangGraphAgent(model, context_mode="unified",
                            plan_guided_react=True, max_steps=4).solve(
                                world, trace_path=tmp_path / "trace.jsonl")
    assert result["stop_reason"] == "appworld_task_completed"
    assert world.executed == ["apis.supervisor.complete_task(bad='x')",
                              "print(apis.supervisor.complete_task(answer='ok'))"]


def test_guard_blocking_errors_are_scoped_and_matching_recovery_only():
    retriever = SimpleNamespace(get_api_schema=lambda name: {
        "method": "POST" if name.endswith("create_transaction") else "GET"})
    failed_write = _local_correction(
        "Execution failed. Response status code is 422", raw_arguments=json.dumps(
            {"code": "apis.venmo.create_transaction(amount=1)"}),
        executed_requests=({"api_name": "venmo.create_transaction", "method": "POST",
                            "target_key": "target-a", "query_key": "query-a",
                            "entity_key": "entity-a"},), retriever=retriever)
    assert _guard_blocking_correction(failed_write)
    same_api = ({"api_name": "venmo.create_transaction", "method": "POST",
                 "target_key": "target-a", "query_key": "query-b",
                 "entity_key": "entity-a"},)
    assert _correction_recovered(failed_write, function_name="execute_python",
                                 raw_arguments=json.dumps({"code": "fixed_write()"}),
                                 outcome={"ok": True}, executed_requests=same_api)
    assert not _correction_recovered(failed_write, function_name="execute_python",
                                     raw_arguments=json.dumps({"code": "other_record_write()"}),
                                     outcome={"ok": True}, executed_requests=(
                                         {"api_name": "venmo.create_transaction", "method": "POST",
                                          "entity_key": "different-entity"},))
    unrelated = ({"api_name": "venmo.show_transactions", "method": "GET",
                  "target_key": "other", "query_key": "query-a"},)
    assert not _correction_recovered(failed_write, function_name="execute_python",
                                     raw_arguments=json.dumps({"code": "unrelated_read()"}),
                                     outcome={"ok": True}, executed_requests=unrelated)

    failed_read = _local_correction(
        "Execution failed. Response status code is 401", raw_arguments=json.dumps(
            {"code": "apis.venmo.show_account(access_token='redacted')"}),
        executed_requests=({"api_name": "venmo.show_account", "method": "GET",
                            "target_key": "target-x", "query_key": "query-x"},), retriever=retriever)
    assert not _guard_blocking_correction(failed_read)
    payment_evidence = [
        {"step": 28, "method": "POST", "api_name": "venmo.create_transaction", "ok": True},
        {"step": 31, "method": "GET", "api_name": "venmo.show_transactions", "ok": True},
    ]
    assert CompletionGuard("Send money to the selected person.").review(
        payment_evidence, unresolved_error=_guard_blocking_correction(failed_read)).status == "SATISFIED"


def test_local_computation_recovery_and_uncertain_write_readback_scope():
    failed_code = ("pls=apis.spotify.show_playlist_library(); "
                   "print(songs[str(i)]['duration'] for i in pls)")
    local_error = _local_correction(
        "Execution failed. KeyError: '23'", raw_arguments=json.dumps({"code": failed_code}),
        executed_requests=({"api_name": "spotify.show_playlist_library", "method": "GET",
                            "target_key": "playlists", "query_key": "all"},), retriever=None)
    assert local_error["error_source"] == "python_local"
    assert {"pls", "songs"}.issubset(set(local_error["recovery_names"]))
    assert not _guard_blocking_correction(local_error)
    assert _correction_recovered(local_error, function_name="execute_python",
                                 raw_arguments=json.dumps({"code": "print(songs[i]['duration'] for i in pls)"}),
                                 outcome={"ok": True}, executed_requests=())
    assert not _correction_recovered(local_error, function_name="execute_python",
                                     raw_arguments=json.dumps({"code": "print(1+1)"}),
                                     outcome={"ok": True}, executed_requests=())

    uncertain = {"api_name": "venmo.create_transaction", "uncertain_write": True,
                 "business_write_error": True, "target_keys": ("same-target",),
                 "entity_keys": ("same-target",)}
    assert not _correction_recovered(uncertain, function_name="execute_python",
                                     raw_arguments=json.dumps({"code": "unrelated()"}),
                                     outcome={"ok": True}, executed_requests=())
    assert not _correction_recovered(uncertain, function_name="execute_python",
                                     raw_arguments=json.dumps({"code": "retry_write()"}),
                                     outcome={"ok": True}, executed_requests=(
                                         {"api_name": "venmo.create_transaction", "method": "POST"},))
    assert _same_target_readback(
        uncertain, ({"api_name": "venmo.show_transaction", "method": "GET",
                     "target_key": "different", "entity_key": "same-target"},),
        outcome={"ok": True, "result": '{"id":"same-target","status":"complete"}'},
        request_events=[SimpleNamespace(api_name="venmo.show_transaction",
                                        execution_status="succeeded")])
    assert not _same_target_readback(uncertain, ({"method": "GET", "target_key": "different",
                                                   "entity_key": "unrelated"},))
    assert not _guard_blocking_correction({**uncertain, "uncertain_write": False,
                                           "business_write_error": False})
