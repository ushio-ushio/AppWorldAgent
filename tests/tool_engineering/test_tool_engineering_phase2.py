import json

from agent.react import LangGraphAgent
from tool_engineering.tools import ApiRetriever, ToolRegistry


def _call(call_id, name, args):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": call_id, "type": "function", "function": {
            "name": name, "arguments": json.dumps(args),
        },
    }]}


class _Model:
    model_name = "offline"

    def __init__(self):
        self.calls = [
            _call("search", "search_api", {"query": "alarms", "app": "phone"}),
            _call("finish", "execute_python", {"code": "apis.supervisor.complete_task(answer=None)"}),
        ]
        self.system_prompts = []
        self.tool_names = []

    def generate_tool_calls(self, messages, tools, *, tool_choice="required"):
        self.system_prompts.append(messages[0]["content"])
        self.tool_names.append([tool["function"]["name"] for tool in tools])
        return self.calls.pop(0), {"input_tokens": 1, "input_cache_hit_tokens": 0,
                                   "output_tokens": 1, "cost_usd": 0.0}


class _World:
    task_id = "catalog-test"
    task = type("Task", (), {
        "instruction": "Use the phone app.",
        "allowed_apps": ["phone"],
        "api_docs": {"phone": {
            "show_alarms": {"api_name": "show_alarms", "app_name": "phone",
                            "description": "Get a list of alarms.", "parameters": []},
        }},
    })()

    def __init__(self):
        self.finished = False

    def execute(self, code):
        assert "complete_task" in code
        self.finished = True
        return "completed"

    def task_completed(self):
        return self.finished


def test_task_app_catalog_is_frozen_in_l0_across_react_steps(tmp_path):
    model = _Model()
    result = LangGraphAgent(model, max_steps=3).solve(
        _World(), trace_path=tmp_path / "trace.jsonl")

    assert result["stop_reason"] == "appworld_task_completed"
    assert len(model.system_prompts) == 2
    assert model.system_prompts[0] == model.system_prompts[1]
    assert "Task-allowed AppWorld app capabilities (static task catalog)" in model.system_prompts[0]
    assert "- phone: APIs primarily cover alarms." in model.system_prompts[0]
    assert model.tool_names[0] == model.tool_names[1] == [
        "search_api", "get_api_schema", "execute_python"]


def test_synonym_expansion_retrieves_and_keeps_compact_paginated_search():
    docs = {"people": {}}
    for name in ("list_contacts", "update_contact", "delete_contact", "show_contact", "search_contact"):
        docs["people"][name] = {
            "api_name": name, "app_name": "people", "method": "GET",
            "description": "Contact record operation. " + ("Detailed documentation. " * 70),
            "parameters": ([{"name": "page_index", "type": "integer", "required": False}]
                           if name == "list_contacts" else []),
        }
    retriever = ApiRetriever(docs, ["people"], schema_aware=True)

    rows = retriever.search_api("coworker", top_k=5)
    serialized = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    assert rows
    assert any(row["api_name"] == "people.list_contacts" for row in rows)
    assert next(row for row in rows if row["api_name"] == "people.list_contacts")["paginated"] is True
    assert len(serialized) < 1800
    assert all("score" in row for row in rows)


def test_get_api_schema_is_deduplicated_per_registry_session():
    schema = {"app_name": "people", "api_name": "list_contacts",
              "description": "Long official schema " * 30,
              "parameters": [
                  {"name": "page_index", "type": "integer", "required": False},
                  {"name": "access_token", "type": "string", "required": True},
                  {"name": "query", "type": "string", "required": True},
              ], "paginated": True}
    registry = ToolRegistry(search_api=lambda **_: [], get_api_schema=lambda **_: schema,
                            execute_python=lambda **_: ("ok", None))

    first = registry.dispatch(name="get_api_schema", raw_arguments='{"api_name":"people.list_contacts"}',
                              tool_call_id="first")
    second = registry.dispatch(name="get_api_schema", raw_arguments='{"api_name":"people.list_contacts"}',
                               tool_call_id="second")

    assert first["ok"] and second["ok"]
    assert first["result"] == schema
    assert second["result"] == {
        "api_name": "people.list_contacts",
        "required_signature": "access_token: string*, query: string*",
        "already_fetched_in_session": True,
        "paginated": True,
    }
    assert len(json.dumps(second["result"])) < len(json.dumps(first["result"]))
    assert [item["function"]["name"] for item in registry.definitions] == [
        "search_api", "get_api_schema", "execute_python"]


def test_search_api_single_batch_and_app_index_are_compatible_and_compact():
    docs = {
        "people": {
            "list_contacts": {"method": "GET", "description": "List contact records.",
                "parameters": [
                    {"name": "access_token", "type": "string", "required": True},
                    {"name": "page_index", "type": "integer", "required": False},
                    {"name": "query", "type": "string", "required": False},
                ]},
            "show_contact": {"method": "GET", "description": "Show one contact record.",
                "parameters": [{"name": "contact_id", "type": "integer", "required": True}]},
        },
        "calendar": {"list_events": {"method": "GET", "description": "List calendar events.",
            "parameters": [{"name": "page_index", "type": "integer", "required": False}]}}
    }
    retriever = ApiRetriever(docs, ["people", "calendar"], schema_aware=True)
    registry = ToolRegistry(search_api=retriever.search_api, get_api_schema=retriever.get_api_schema,
                            execute_python=lambda **_: ("ok", None), schema_aware_retrieval=True)

    single = registry.dispatch(name="search_api", raw_arguments='{"query":"contact list"}', tool_call_id="one")
    batch = registry.dispatch(name="search_api", raw_arguments=json.dumps({
        "queries": ["contact list", "list friends"], "app_name": "people"}), tool_call_id="batch")
    app_index = registry.dispatch(name="search_api", raw_arguments='{"app_name":"people"}', tool_call_id="index")

    assert single["ok"] and any(row["api_name"] == "people.list_contacts" for row in single["result"])
    assert batch["ok"] and len({row["api_name"] for row in batch["result"]}) == len(batch["result"])
    row = next(row for row in batch["result"] if row["api_name"] == "people.list_contacts")
    assert "required access_token:string" in row["parameter_hint"]
    assert "optional page_index:integer,query:string" in row["parameter_hint"]
    assert app_index["ok"] and app_index["result"]["total_apis"] == 2
    assert all(item["api_name"].startswith("people.") for item in app_index["result"]["apis"])
    assert [item["function"]["name"] for item in registry.definitions] == [
        "search_api", "get_api_schema", "execute_python"]
