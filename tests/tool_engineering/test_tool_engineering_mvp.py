import json
from types import SimpleNamespace

from agentlab.context_engineering.context import CacheFirstContextBuilder
from agentlab.memory.store import ObservationStore
from agentlab.tool_engineering.tools import ApiRetriever, ToolRegistry
from agentlab.environments.appworld import AppWorldTool


DOCS = {"demo": {
    "update": {"path": "/demo/update", "method": "POST", "description": "Update a record.",
               "parameters": [{"name": "record_id", "type": "integer", "required": True}]},
    "list": {"path": "/demo/list", "method": "GET", "description": "List records.",
             "parameters": [{"name": "page_index", "type": "integer", "required": False},
                            {"name": "access_token", "type": "string", "required": True}]},
}}


def test_observation_default_page_and_no_second_offload():
    store = ObservationStore("task")
    record = store.add(tool_call_id="call", tool_name="read_observation", step=1,
                       content="x" * 3200, arguments="{}", outcome={"ok": True})
    assert len(store.read_observation(record.id)["content"]) == 1600
    builder = CacheFirstContextBuilder(store)
    assert builder._freeze_output({"role": "tool", "tool_call_id": "call", "content": record.content}) == record.content


def test_compact_search_and_pagination():
    docs = {"demo": dict(DOCS["demo"])}
    for i in range(3):
        docs["demo"][f"extra_{i}"] = {"path": f"/demo/extra{i}", "method": "GET",
            "description": "records " + "long description " * 100,
            "parameters": [{"name": "id", "type": "integer", "required": True}]}
    retriever = ApiRetriever(docs, ["demo"], schema_aware=True)
    rows = retriever.search_api("demo records list update extra")
    assert len(rows) == 5
    assert len(json.dumps(rows, ensure_ascii=False, separators=(",", ":"))) < 1800
    assert next(row for row in rows if row["api_name"] == "demo.list")["paginated"] is True
    assert retriever.get_api_schema("demo.list")["paginated"] is True
    short_rows = ApiRetriever(DOCS, ["demo"], schema_aware=True).search_api("demo list update")
    assert any("*" in p["param"] for row in short_rows for p in row.get("schema", {}).get("parameters", []))


def test_partial_success_summary_and_error_envelopes():
    success = SimpleNamespace(api_name="demo.update", execution_status="succeeded",
                              operation_type=SimpleNamespace(value="WRITE"), http_status=200, reason=None)
    failure = SimpleNamespace(api_name="demo.update", execution_status="failed",
                              operation_type=SimpleNamespace(value="WRITE"), http_status=422, reason="Exception")
    class Gate:
        state = SimpleNamespace(mode="shadow")
        def begin_action(self, *args): pass
        def end_action(self, *args): return [success, failure]
    class World:
        task = SimpleNamespace(api_docs=DOCS, allowed_apps=["demo"])
        requester = SimpleNamespace(requests=[])
        def execute(self, code):
            self.requester.requests.extend([{"method": "POST", "url": "/demo/update", "data": {}},
                                            {"method": "POST", "url": "/demo/update", "data": {}}])
            return "Execution failed. Response status code is 422"
    tool = AppWorldTool(World(), outcome_gate=Gate())
    tool.set_assurance_action(1, "call")
    result = tool.registry.dispatch(name="execute_python", raw_arguments='{"code":"print(1)"}', tool_call_id="call")
    assert result["execution_summary"]["succeeded_calls"] == 1
    assert result["execution_summary"]["failed_calls"] == 1
    assert result["execution_summary"]["succeeded_mutations"] == ["demo.update"]
    assert "record_id" in result["error_envelope"]["recovery_hint"]
    for error, expected in [("Response status code is 401", "http_401_auth"),
                            ("takes 0 positional arguments", "positional_arg_error"),
                            ("Usage of the following function is not allowed: csv.writer", "sandbox_restricted"),
                            ("Object of type function is not JSON serializable", "json_not_serializable")]:
        assert tool.registry._error_envelope(error, None)["error_type"] == expected
