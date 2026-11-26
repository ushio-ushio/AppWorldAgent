import copy
import json
import tempfile
import unittest
from pathlib import Path

from agentlab.agent.react import LangGraphAgent
from agentlab.agent.prompt import ContextManager
from agentlab.tool_engineering.tools import ApiRetriever, compact_api_schema
from agentlab.execution_assurance.recovery import write_operation_fingerprints


class BatchWorld:
    """Small AppWorld-shaped fake; execute is the only Python action boundary."""
    task_id = "batch-task"
    docs = {
        "phone": {
            "list_records": {
                "app_name": "phone", "api_name": "list_records", "method": "GET",
                "description": "List records with page-based pagination.",
                "parameters": [
                    {"name": "page", "type": "integer", "required": True,
                     "description": "Page number, starting at 1.", "constraints": ["minimum: 1"]},
                    {"name": "access_token", "type": "string", "required": True,
                     "description": "Access token from login.", "constraints": []},
                    {"name": "limit", "type": "integer", "required": False,
                     "description": "Maximum records per page.", "constraints": ["maximum: 20"]},
                ],
            },
            "delete_record": {
                "app_name": "phone", "api_name": "delete_record", "method": "DELETE",
                "description": "Delete a record by ID.",
                "parameters": [{"name": "record_id", "type": "string", "required": True}],
            },
        }
    }
    task = type("Task", (), {"instruction": "Find matching records.",
                              "allowed_apps": ["phone"], "api_docs": docs})()

    def __init__(self):
        self.finished = False
        self.state = {"pages": 0, "results": None}
        self.apis = self
        self.phone = self

    def execute(self, code):
        # In the real runtime this call is delegated to the AppWorld Python
        # shell. The fake executes harmless test code to verify shell continuity.
        namespace = {"apis": self.apis, "state": self.state}
        exec(code, namespace, namespace)
        self.state.update({key: value for key, value in namespace.items()
                           if key in {"pages", "results"}})
        return namespace.get("_result", "")

    class _Response:
        def __init__(self, rows, has_more):
            self.rows, self.has_more = rows, has_more

    def list_records(self, *, page, access_token, limit=None):
        self.state["pages"] += 1
        if page == 1:
            return self._Response([{"id": "r1", "active": True}, {"id": "r2", "active": False}], True)
        if page == 2:
            return self._Response([{"id": "r1", "active": True}, {"id": "r3", "active": True}], False)
        raise AssertionError("Pagination did not stop at documented end condition")

    def task_completed(self):
        return self.finished


class FunctionModel:
    model_name = "offline-mock"

    def __init__(self, calls):
        self.calls = calls
        self.requests = []

    def generate_tool_calls(self, messages, tools, *, tool_choice="required"):
        self.requests.append((messages, tools))
        call = self.calls.pop(0)
        return {"role": "assistant", "content": None, "tool_calls": [call]}, {
            "input_tokens": 1, "input_cache_hit_tokens": 0,
            "output_tokens": 1, "cost_usd": 0.0}


def call(call_id, name, args):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(args)}}


class TestSchemaAwareRetrieval(unittest.TestCase):
    def setUp(self):
        self.docs = copy.deepcopy(BatchWorld.docs)

    def test_compact_schema_keeps_required_types_constraints_and_auth(self):
        result = compact_api_schema("phone.list_records", self.docs["phone"]["list_records"])
        self.assertEqual([p["name"] for p in result["parameters"] if p["required"]],
                         ["page", "access_token"])
        self.assertEqual(result["parameters"][0]["type"], "integer")
        self.assertIn("minimum: 1", result["parameters"][0]["constraints"])
        self.assertIn("Access token", result["parameters"][1]["description"])
        self.assertEqual(self.docs, BatchWorld.docs)  # index creation is read-only

    def test_search_attaches_two_compact_schemas_and_preserves_bm25(self):
        retriever = ApiRetriever(self.docs, ["phone"], schema_aware=True)
        rows = retriever.search_api("phone", top_k=2)
        self.assertEqual([r["rank"] for r in rows], [1, 2])
        self.assertTrue(all("schema" in r for r in rows))
        self.assertEqual(rows[0]["schema"]["schema_status"], "compact_official_schema")
        self.assertGreaterEqual(rows[0]["score"], rows[1]["score"])
        self.assertIn("phone.list_records", [row["api_name"] for row in rows])
        self.assertIsNone(retriever.get_api_schema("unknown"))

    def test_complex_or_oversized_schema_falls_back_to_exact_lookup(self):
        doc = {"parameters": [{"name": "nested", "type": {"object": ["x"]},
                               "required": True, "description": "x" * 200}]}
        compact = compact_api_schema("phone.complex", doc, max_chars=128)
        self.assertEqual(compact, {"api_name": "phone.complex",
                                   "schema_status": "use_get_api_schema",
                                   "schema_lookup": "phone.complex"})

    def test_disabled_schema_mode_preserves_search_result_shape(self):
        retriever = ApiRetriever(self.docs, ["phone"])
        self.assertEqual(set(retriever.search_api("list records")[0]),
                         {"api_name", "description", "score", "parameter_hint"})
        self.assertIn("required page:integer,access_token:string",
                      retriever.search_api("list records")[0]["parameter_hint"])
        self.assertIn("optional limit:integer",
                      retriever.search_api("list records")[0]["parameter_hint"])

    def test_distinct_loop_batches_have_distinct_write_fingerprints(self):
        retriever = ApiRetriever(self.docs, ["phone"])
        first = write_operation_fingerprints(
            "for x in first_batch:\n    apis.phone.delete_record(record_id=x)", retriever)
        second = write_operation_fingerprints(
            "for x in second_batch:\n    apis.phone.delete_record(record_id=x)", retriever)
        self.assertEqual(first[0][1], "phone.delete_record")
        self.assertEqual(second[0][1], "phone.delete_record")
        self.assertNotEqual(first[0][0], second[0][0])


class TestProgrammaticExecutionGuidance(unittest.TestCase):
    def test_prompt_flags_are_independent_and_baseline_prompt_is_unchanged(self):
        baseline = ContextManager("t", "task").messages[0]["content"]
        a_only = ContextManager("t", "task", schema_aware_retrieval=True).messages[0]["content"]
        ab = ContextManager("t", "task", schema_aware_retrieval=True,
                            programmatic_execution=True).messages[0]["content"]
        self.assertIn("Read it before calling that API.", baseline)
        self.assertIn("compact official schema", a_only)
        self.assertNotIn("Programmatic execution guidance", a_only)
        self.assertIn("Programmatic execution guidance", ab)
        self.assertIn("never use host filesystem access", ab)
        self.assertIn("do not blindly retry or repeat a payment", ab)

    def test_single_execute_python_call_batches_pagination_filter_and_dedup(self):
        code = """\nrows = []\nseen = set()\npage = 1\nwhile True:\n    response = apis.phone.list_records(page=page, access_token='fake', limit=20)\n    for row in response.rows:\n        if row['active'] and row['id'] not in seen:\n            rows.append(row)\n            seen.add(row['id'])\n    if not response.has_more:\n        break\n    page += 1\nstate['results'] = rows\n_result = {'pages': page, 'active_unique_ids': [r['id'] for r in rows]}\n"""
        model = FunctionModel([call("batch", "execute_python", {"code": code})])
        world = BatchWorld()
        with tempfile.TemporaryDirectory() as temp:
            result = LangGraphAgent(model, max_steps=1,
                                    schema_aware_retrieval=True,
                                    programmatic_execution=True).solve(
                world, trace_path=Path(temp) / "trace.jsonl")
        self.assertEqual(result["stop_reason"], "max_steps_reached")
        self.assertEqual(world.state["pages"], 2)
        self.assertEqual([r["id"] for r in world.state["results"]], ["r1", "r3"])
        # The graph still uses its existing registered execute_python node/tool.
        names = [definition["function"]["name"] for definition in model.requests[0][1]]
        self.assertIn("execute_python", names)
        self.assertEqual(len(model.requests), 1)


if __name__ == "__main__":
    unittest.main()
