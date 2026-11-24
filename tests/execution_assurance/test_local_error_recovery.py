import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from agentlab.environments.appworld import AppWorldTool
from agentlab.execution_assurance.recovery import (
    _correction_recovered,
    _executed_write_fingerprints,
    _local_correction,
    _safe_during_uncertain_write,
    _same_target_readback,
)
from agentlab.memory.store import ObservationStore


class Retriever:
    def get_api_schema(self, api_name):
        if api_name.endswith((".signup", ".create_payment", ".login")):
            return {"method": "POST", "parameters": []}
        if api_name.endswith((".show_account", ".show_payment")):
            return {"method": "GET", "parameters": []}
        return None


def _event(api_name, status, http_status=None, reason=None):
    return {"api_name": api_name, "execution_status": status,
            "http_status": http_status, "reason": reason}


def test_local_python_error_before_request_does_not_blame_later_api():
    correction = _local_correction(
        "Execution failed. Usage of the following function is not allowed: csv.writer.",
        raw_arguments=json.dumps({"code": "writer=csv.writer(buf); apis.file_system.login(email=x)"}),
        executed_requests=(), retriever=Retriever())
    assert correction["execution_state"] == "LOCAL_ERROR"
    assert correction["error_source"] == "python_local"
    assert correction["api_name"] is None
    assert correction["uncertain_write"] is False
    assert correction["api_attempts"] == [{"api_name": "file_system.login",
                                             "execution_state": "NOT_EXECUTED"}]


def test_partial_api_activity_is_retained_before_later_python_error():
    requests = ({"api_name": "files.show_account", "method": "GET"},
                {"api_name": "files.create_file", "method": "POST"})
    events = (_event("files.show_account", "succeeded"),
              _event("files.create_file", "succeeded"))
    correction = _local_correction(
        "Execution failed. TypeError: unsupported local value",
        raw_arguments=json.dumps({"code": "a=apis.files.show_account(); b=apis.files.create_file()"}),
        executed_requests=requests, retriever=Retriever(), request_events=events)
    assert correction["execution_state"] == "LOCAL_ERROR"
    assert correction["api_name"] is None
    assert [row["execution_state"] for row in correction["request_outcomes"]] == ["SUCCESS", "SUCCESS"]
    assert correction["uncertain_write"] is False


def test_repeated_api_calls_are_matched_to_actual_request_count():
    correction = _local_correction(
        "Execution failed. TypeError: local formatting failed",
        raw_arguments=json.dumps({"code": "apis.files.create_file(name='a'); apis.files.create_file(name='b')"}),
        executed_requests=({"api_name": "files.create_file", "method": "POST"},),
        request_events=(_event("files.create_file", "succeeded"),), retriever=Retriever())
    assert correction["execution_state"] == "LOCAL_ERROR"
    assert correction["request_outcomes"] == [{"api_name": "files.create_file", "method": "POST",
                                                 "execution_state": "SUCCESS", "http_status": None}]
    assert correction["api_attempts"] == [{"api_name": "files.create_file",
                                             "execution_state": "NOT_EXECUTED"}]


def test_explicit_signup_conflict_is_definite_and_login_is_recovery():
    correction = _local_correction(
        'Execution failed. Response status code is 409: {"message":"User with this email already exists."}',
        raw_arguments=json.dumps({"code": "apis.file_system.signup(email=email)"}),
        executed_requests=({"api_name": "file_system.signup", "method": "POST"},),
        request_events=(_event("file_system.signup", "failed", 409),),
        retriever=Retriever())
    assert correction["execution_state"] == "DEFINITE_FAILURE"
    assert correction["api_name"] == "file_system.signup"
    assert correction["business_write_error"] is True
    assert correction["uncertain_write"] is False
    assert _correction_recovered(
        correction, function_name="execute_python",
        raw_arguments=json.dumps({"code": "apis.file_system.login(email=email)"}),
        outcome={"ok": True},
        executed_requests=({"api_name": "file_system.login", "method": "POST"},))


def test_timeout_after_write_request_remains_uncertain():
    correction = _local_correction(
        "Execution failed. ReadTimeout: request timed out",
        raw_arguments=json.dumps({"code": "apis.payments.create_payment(amount=4)"}),
        executed_requests=({"api_name": "payments.create_payment", "method": "POST"},),
        request_events=(_event("payments.create_payment", "unknown", reason="write_outcome_uncertain"),),
        retriever=Retriever())
    assert correction["execution_state"] == "UNCERTAIN_WRITE"
    assert correction["uncertain_write"] is True


def test_unexplained_conflict_or_failed_write_response_stays_uncertain():
    correction = _local_correction(
        "Execution failed. Response status code is 409",
        raw_arguments=json.dumps({"code": "apis.payments.create_payment(amount=4)"}),
        executed_requests=({"api_name": "payments.create_payment", "method": "POST"},),
        request_events=(_event("payments.create_payment", "failed", 409),),
        retriever=Retriever())
    assert correction["execution_state"] == "UNCERTAIN_WRITE"
    no_status = _local_correction(
        "Execution failed. Requester raised an exception",
        raw_arguments=json.dumps({"code": "apis.payments.create_payment(amount=4)"}),
        executed_requests=({"api_name": "payments.create_payment", "method": "POST"},),
        request_events=(_event("payments.create_payment", "failed"),),
        retriever=Retriever())
    assert no_status["execution_state"] == "UNCERTAIN_WRITE"


def test_partial_success_then_api_failure_preserves_both_request_outcomes():
    requests = ({"api_name": "files.show_account", "method": "GET"},
                {"api_name": "file_system.signup", "method": "POST"})
    correction = _local_correction(
        'Execution failed. Response status code is 409: {"message":"User with this email already exists."}',
        raw_arguments=json.dumps({"code": "a=apis.files.show_account(); apis.file_system.signup(email=e)"}),
        executed_requests=requests,
        request_events=(_event("files.show_account", "succeeded"),
                        _event("file_system.signup", "failed", 409)),
        retriever=Retriever())
    assert correction["api_name"] == "file_system.signup"
    assert correction["execution_state"] == "DEFINITE_FAILURE"
    assert [row["execution_state"] for row in correction["request_outcomes"]] == [
        "SUCCESS", "DEFINITE_FAILURE"]


def test_uncertain_write_allows_readback_and_authentication_recovery():
    correction = {"api_name": "payments.create_payment", "uncertain_write": True,
                  "target_keys": ("target",), "entity_keys": ("entity",)}
    retriever = Retriever()
    assert _safe_during_uncertain_write(
        "print(apis.payments.show_payment(payment_id='p1'))", correction, retriever)
    assert _safe_during_uncertain_write(
        "print(apis.payments.login(email=email, password=password))", correction, retriever)


def test_uncertain_write_blocks_same_app_mutation_but_allows_separate_app():
    correction = {"api_name": "payments.create_payment", "uncertain_write": True}
    retriever = Retriever()
    assert not _safe_during_uncertain_write(
        "apis.payments.create_payment(amount=4)", correction, retriever)
    assert _safe_during_uncertain_write(
        "apis.calendar.create_event(title='meeting')", correction, retriever)
    assert _safe_during_uncertain_write("value = 1 + 2", correction, retriever)


def test_unrelated_success_does_not_clear_uncertain_write():
    correction = {"api_name": "payments.create_payment", "uncertain_write": True,
                  "target_keys": ("target-a",), "entity_keys": ("entity-a",)}
    assert not _correction_recovered(
        correction, function_name="execute_python", raw_arguments=json.dumps({"code": "other()"}),
        outcome={"ok": True},
        executed_requests=({"api_name": "calendar.show_events", "method": "GET"},))
    assert not _same_target_readback(
        correction, ({"api_name": "payments.show_payment", "method": "GET",
                     "target_key": "other", "entity_key": "other-entity"},),
        outcome={"ok": True, "result": '{"id":"other-entity"}'},
        request_events=[SimpleNamespace(api_name="payments.show_payment",
                                         execution_status="succeeded")])


def test_trusted_same_target_readback_matches_uncertain_write():
    correction = {"api_name": "payments.create_payment", "uncertain_write": True,
                  "target_keys": ("target",), "entity_keys": ("entity",)}
    assert _same_target_readback(
        correction, ({"api_name": "payments.show_payment", "method": "GET",
                     "target_key": "different", "entity_key": "entity"},),
        outcome={"ok": True, "result": '{"id":"entity","status":"paid"}'},
        request_events=[SimpleNamespace(api_name="payments.show_payment",
                                         execution_status="succeeded")])
    assert not _same_target_readback(
        correction, ({"api_name": "payments.show_payment", "method": "GET",
                     "target_key": "different", "entity_key": "entity"},),
        outcome={"ok": False, "result": '{"id":"entity"}'},
        request_events=[SimpleNamespace(api_name="payments.show_payment",
                                         execution_status="succeeded")])
    assert not _same_target_readback(
        correction, ({"api_name": "payments.show_payment", "method": "GET",
                     "target_key": "different", "entity_key": "entity"},),
        outcome={"ok": True, "result": '{"id":"entity"}'},
        request_events=[SimpleNamespace(api_name="payments.show_payment",
                                         execution_status="failed")])


def test_only_actual_request_events_record_write_attempts():
    planned = [("fp-create", "file_system.create_file")]
    assert _executed_write_fingerprints(planned, ()) == []
    assert _executed_write_fingerprints(planned, (
        {"api_name": "file_system.create_file", "method": "POST"},)) == [planned[0]]




def test_lite_tool_blocks_only_a_write_confirmed_by_requester():
    docs = {"phone": {"delete_record": {"method": "DELETE",
        "path": "/phone/records/{record_id}", "description": "Delete a record.",
        "parameters": [{"name": "record_id", "type": "string", "required": True}]}}}

    class World:
        task_id = "lite-ledger"
        task = SimpleNamespace(instruction="Delete a record", allowed_apps=["phone"], api_docs=docs)
        def __init__(self):
            self.requester = SimpleNamespace(requests=[])
            self.executed = []
        def execute(self, code):
            self.executed.append(code)
            self.requester.requests.append({"method": "DELETE",
                "url": "https://phone.appworld.dev/phone/records/r1",
                "data": {"record_id": "r1"}})
            return "deleted"

    world = World()
    tool = AppWorldTool(world)
    code = "apis.phone.delete_record(record_id='r1')"
    assert tool.execute_python(code) == ("deleted", None)
    result, error = tool.execute_python(code)
    assert error == "write_replay_blocked"
    assert "exact operation was already attempted" in result
    assert world.executed == [code]


def test_failed_authentication_can_be_retried_after_credentials_are_corrected():
    docs = {"phone": {"login": {"method": "POST", "path": "/phone/login",
        "description": "Authenticate.", "parameters": [
            {"name": "username", "type": "string", "required": True},
            {"name": "password", "type": "string", "required": True}]}}}

    class World:
        task_id = "auth-recovery"
        task = SimpleNamespace(instruction="Log in", allowed_apps=["phone"], api_docs=docs)
        def __init__(self):
            self.requester = SimpleNamespace(requests=[])
            self.calls = 0
        def execute(self, code):
            self.calls += 1
            self.requester.requests.append({"method": "POST",
                "url": "https://phone.appworld.dev/phone/login", "data": {}})
            if self.calls == 1:
                return 'Execution failed. Response status code is 401: {"message":"Invalid credentials"}'
            return "authenticated"

    world = World()
    tool = AppWorldTool(world)
    code = "apis.phone.login(username=user, password=password)"
    assert tool.execute_python(code)[1] is not None
    assert tool.execute_python(code) == ("authenticated", None)
    assert world.calls == 2


def test_historical_6104387_error_chain_offline_replay():
    trace = Path(r"D:\worldapp\runs\agentlab_v2_lite_v3_assurance_shadow_train15_20261124_131203\6104387_1\trace.jsonl")
    if not trace.exists():
        pytest.skip("Historical Train-15 trace is not present in this checkout.")
    events = {}
    for line in trace.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("event") == "tool_call" and row.get("step") in {45, 48, 49, 52, 53}:
            events[row["step"]] = row
    local = events[45]
    outcome = local.get("tool_result", {})
    try:
        json.loads(local.get("arguments", "{}"))
        local_arguments = local.get("arguments", "{}")
    except json.JSONDecodeError:
        # The historical logger's credential redaction made this serialized
        # argument string non-JSON. Replay only the recorded call site needed
        # for the NOT_EXECUTED classification; never expose its contents.
        local_arguments = json.dumps({"code": "writer=csv.writer(buf); apis.file_system.login(email=email)"})
    correction = _local_correction(
        outcome.get("execution_error") or outcome.get("error") or "",
        raw_arguments=local_arguments,
        executed_requests=tuple(local.get("executed_requests", ())), retriever=Retriever())
    assert correction["execution_state"] == "LOCAL_ERROR"
    assert correction["api_name"] is None
    assert correction["api_attempts"][0]["execution_state"] == "NOT_EXECUTED"

    signup = events[48]
    signup_result = signup.get("tool_result", {})
    try:
        json.loads(signup.get("arguments", "{}"))
        signup_arguments = signup.get("arguments", "{}")
    except json.JSONDecodeError:
        signup_arguments = json.dumps({"code": "apis.file_system.signup(email=email)"})
    signup_correction = _local_correction(
        signup_result.get("execution_error") or signup_result.get("error") or "",
        raw_arguments=signup_arguments,
        executed_requests=tuple(signup.get("executed_requests", ())), retriever=Retriever())
    assert signup_correction["execution_state"] == "DEFINITE_FAILURE"
    assert signup_correction["api_name"] == "file_system.signup"
    assert all(events[step].get("tool_result", {}).get("validation") != "accepted"
               for step in (49, 52, 53))
