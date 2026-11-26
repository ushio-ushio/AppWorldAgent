from __future__ import annotations

import json
from types import SimpleNamespace

from execution_assurance.reflection_v2 import (
    AUDITOR_SYSTEM_PROMPT, TerminalAudit, audit_feedback, build_audit_context,
    deterministic_answer_gap,
)
from environments.appworld import AppWorldTool


class Trace:
    def __init__(self): self.rows = []
    def append(self, row): self.rows.append(row)


class AuditModel:
    model_name = "offline-audit"
    def __init__(self): self.requests = []
    def generate_tool_calls(self, messages, tools, *, tool_choice="required"):
        self.requests.append((messages, tools, tool_choice))
        return {"tool_calls": [{"id": "audit", "type": "function", "function": {
            "name": "submit_terminal_audit", "arguments": json.dumps({
                "verdict": "PASS", "evidence_gaps": "", "next_verification": ""})}}]}, {
                    "input_tokens": 3, "input_cache_hit_tokens": 0, "output_tokens": 2, "cost_usd": 0.0}


class FixedVerdictModel(AuditModel):
    def __init__(self, verdict, evidence_gaps, next_verification):
        super().__init__()
        self.result = {"tool_calls": [{"id": "audit", "type": "function", "function": {
            "name": "submit_terminal_audit", "arguments": json.dumps({
                "verdict": verdict, "evidence_gaps": evidence_gaps,
                "next_verification": next_verification})}}]}
    def generate_tool_calls(self, messages, tools, *, tool_choice="required"):
        self.requests.append((messages, tools, tool_choice))
        return self.result, {"input_tokens": 3, "input_cache_hit_tokens": 0,
                             "output_tokens": 2, "cost_usd": 0.0}


def test_audit_context_is_independent_and_contains_only_four_inputs():
    ledger = [{"app.endpoint": "drive.create_file", "status": "succeeded",
               "key_parameters": {"file_name": "backup.txt"}, "evidence_ref": "obs_2"}]
    messages = build_audit_context("Create a backup file.", ledger,
                                   'apis.supervisor.complete_task(answer=None)')
    assert len(messages) == 2 and messages[0]["content"] == AUDITOR_SYSTEM_PROMPT
    pack = json.loads(messages[1]["content"])
    assert set(pack) == {"original_instruction", "requester_ledger", "proposed_answer"}
    assert pack["original_instruction"] == "Create a backup file."
    assert pack["requester_ledger"] == ledger
    assert pack["proposed_answer"] == {"python_type": "NoneType", "content": None}
    assert "executor private thought" not in messages[1]["content"]
    assert "search_api history" not in messages[1]["content"]


def test_deterministic_answer_precheck_blocks_refusal_and_units_only():
    instruction = "How much money was transferred?"
    assert deterministic_answer_gap(instruction, "Unable to verify the amount", "str")
    assert deterministic_answer_gap(instruction, "$34 USD", "str")
    assert deterministic_answer_gap(instruction, "The total was 34 after checking records.", "str")
    assert deterministic_answer_gap("Delete the requested item.", None, "NoneType") is None
    assert deterministic_answer_gap(instruction, 34, "int") is None
    assert deterministic_answer_gap(instruction, "34", "str") is None


def test_auditor_receives_no_executor_history_and_feedback_is_short_json():
    model, trace = AuditModel(), Trace()
    auditor = TerminalAudit(model, trace, "task", "What is the total amount?")
    decision, route = auditor.review(4, "submission", pending_submission=
        'apis.supervisor.complete_task(answer="34")', ledger=[{"app.endpoint": "venmo.search", "status": "succeeded"}])
    assert route == "allow_existing_checks" and decision["status"] == "PASS"
    messages, tools, choice = model.requests[0]
    assert len(messages) == 2 and choice == "required"
    assert [row["function"]["name"] for row in tools] == ["submit_terminal_audit"]
    assert "Executor traceback noise" not in json.dumps(messages)
    feedback = audit_feedback({"verdict": "FAIL", "evidence_gaps": "missing operation", "next_verification": "read back"})
    assert len(feedback) < 300
    assert set(json.loads(feedback)) == {"verdict", "evidence_gaps", "next_verification"}


def test_invalid_auditor_response_is_uncertain_not_implicit_pass():
    class BrokenModel(AuditModel):
        def generate_tool_calls(self, *args, **kwargs):
            return {"tool_calls": []}, {"input_tokens": 1, "input_cache_hit_tokens": 0,
                                         "output_tokens": 0, "cost_usd": 0.0}
    audit = TerminalAudit(BrokenModel(), Trace(), "task", "What is the answer?")
    decision, route = audit.review(1, "submission", pending_submission=
        'apis.supervisor.complete_task(answer="42")', ledger=[])
    assert decision["status"] == "UNCERTAIN"
    assert route == "one_bounded_repair"


def test_600_char_audit_json_is_parsed_and_only_executor_feedback_is_truncated():
    model, trace = FixedVerdictModel("FAIL", "Missing evidence: " + "x" * 520,
                                     "Read the missing date range and recompute the total.") , Trace()
    auditor = TerminalAudit(model, trace, "task", "What is the total amount?")
    decision, route = auditor.review(7, "submission",
        pending_submission="apis.supervisor.complete_task(answer=393)", ledger=[])
    feedback = audit_feedback(decision)
    row = next(item for item in trace.rows if item["event"] == "terminal_audit")
    assert len(json.dumps(model.result["tool_calls"][0]["function"]["arguments"])) > 600
    assert decision["verdict"] == "FAIL" and decision["status"] == "MISSING"
    assert decision["evidence_gaps"].startswith("Missing evidence:")
    assert row["fallback_reason"] is None
    assert route == "one_bounded_repair"
    assert len(feedback) <= 480
    assert json.loads(feedback)["evidence_gaps"].startswith("Missing evidence:")


def test_deterministic_format_retry_does_not_spend_semantic_repair_budget():
    model = FixedVerdictModel("UNCERTAIN", "Some queried dates are not evidenced.",
                              "Read the remaining months.")
    auditor = TerminalAudit(model, Trace(), "task", "What is the total amount this year?")
    first, first_route = auditor.review(1, "submission",
        pending_submission='apis.supervisor.complete_task(answer="$393.00 USD")', ledger=[])
    assert first["verdict"] == "FAIL" and first_route == "one_bounded_repair"
    assert auditor.format_repair_rounds == 1 and auditor.repair_rounds == 0
    assert not model.requests  # deterministic precheck required no semantic audit request

    second, second_route = auditor.review(2, "submission",
        pending_submission="apis.supervisor.complete_task(answer=393)", ledger=[])
    assert second["verdict"] == "UNCERTAIN"
    assert second["evidence_gaps"] == "Some queried dates are not evidenced."
    assert second_route == "one_bounded_repair"
    assert auditor.repair_rounds == 1
    assert len(model.requests) == 1


def test_app_401_counter_escalates_per_app_and_success_resets():
    def event(status):
        return SimpleNamespace(api_name="mail.send_message", operation_type=SimpleNamespace(value="WRITE"),
            execution_status="succeeded" if status == 200 else "failed", http_status=status,
            reason=None if status == 200 else "HTTPError", step=1, arguments={})

    class Gate:
        state = SimpleNamespace(mode="shadow")
        def __init__(self): self.statuses = [401, 401, 200, 401]
        def begin_action(self, *_): pass
        def end_action(self, *_): return [event(self.statuses.pop(0))]

    class World:
        task = SimpleNamespace(api_docs={"mail": {"send_message": {
            "method": "POST", "path": "/mail/send", "description": "Send a message.", "parameters": []}}},
            allowed_apps=["mail"])
        requester = SimpleNamespace(requests=[])
        def __init__(self): self.responses = [401, 401, 200, 401]
        def execute(self, _code):
            status = self.responses.pop(0)
            return "Execution successful." if status == 200 else f"Execution failed. Response status code is {status}"

    gate = Gate()
    tool = AppWorldTool(World(), outcome_gate=gate)
    def call(i):
        tool.set_assurance_action(i, str(i))
        return tool.registry.dispatch(name="execute_python", raw_arguments='{"code":"send()"}', tool_call_id=str(i))

    first, second = call(1), call(2)
    assert first["error_envelope"]["error_type"] == "http_401_auth"
    assert "show_account_passwords()" in first["error_envelope"]["recovery_hint"]
    assert "failed twice consecutively" in second["error_envelope"]["recovery_hint"]
    assert "show_account_passwords()" in second["error_envelope"]["recovery_hint"]
    assert call(3)["ok"] is True
    fourth = call(4)
    assert fourth["error_envelope"]["error_type"] == "http_401_auth"
    assert "failed twice consecutively" not in fourth["error_envelope"]["recovery_hint"]
