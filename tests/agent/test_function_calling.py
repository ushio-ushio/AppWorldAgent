import json
import tempfile
import unittest
from pathlib import Path

from agentlab.agent.react import LangGraphAgent


def tool_call(call_id, name, arguments):
    return {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": call_id, "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments),
        }}],
    }


class FakeFunctionModel:
    model_name = "mock-function-model"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []

    def generate_tool_calls(self, messages, tools, *, tool_choice="required"):
        self.requests.append({"messages": list(messages), "tools": tools, "tool_choice": tool_choice})
        output = self.outputs.pop(0)
        return output, {"input_tokens": 5, "input_cache_hit_tokens": 1,
                        "output_tokens": 3, "cost_usd": 0.00001}


class FakeFunctionWorld:
    task_id = "fake-function-task"
    task = type("Task", (), {
        "instruction": "Use a harmless mock flow.",
        "allowed_apps": ["phone"],
        "api_docs": {"phone": {
            "show_alarms": {
                "app_name": "phone", "api_name": "show_alarms",
                "description": "Get a list of alarms.", "parameters": [],
                "response_schemas": {"success": {"alarms": []}},
            },
        }},
    })()

    def __init__(self):
        self.shell_state = {}
        self.executed = []
        self.finished = False

    def execute(self, code):
        self.executed.append(code)
        if code == "value = 40":
            self.shell_state["value"] = 40
            return "value stored"
        if code == "print(value + 2)":
            return str(self.shell_state["value"] + 2)
        if "complete_task" in code:
            self.finished = True
            return "completion accepted"
        raise AssertionError(f"Unexpected mock code: {code}")

    def task_completed(self):
        return self.finished


class TestFunctionCallingProtocol(unittest.TestCase):
    def test_bm25_schema_python_and_tool_message_chain(self):
        model = FakeFunctionModel([
            tool_call("call-1", "search_api", {"query": "list alarms", "app": "phone"}),
            tool_call("call-2", "get_api_schema", {"api_name": "phone.show_alarms"}),
            tool_call("call-3", "execute_python", {"code": "value = 40"}),
            tool_call("call-4", "execute_python", {"code": "print(value + 2)"}),
            tool_call("call-5", "execute_python", {"code": "apis.supervisor.complete_task()"}),
        ])
        world = FakeFunctionWorld()
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "trace.jsonl"
            result = LangGraphAgent(model, max_steps=6).solve(
                world, trace_path=trace_path
            )
            rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(result["stop_reason"], "appworld_task_completed")
        self.assertEqual(result["tool_call_count"], 5)
        self.assertEqual(result["python_execution_count"], 3)
        self.assertEqual(world.executed[1], "print(value + 2)")
        self.assertIn("42", model.requests[4]["messages"][-1]["content"])
        tool_messages = [m for m in model.requests[1]["messages"] if m["role"] == "tool"]
        self.assertEqual(tool_messages[0]["tool_call_id"], "call-1")
        self.assertIn("phone.show_alarms", tool_messages[0]["content"])
        self.assertEqual([r["event"] for r in rows if r["event"] == "tool_call"].count("tool_call"), 5)

    def test_invalid_arguments_are_observed_and_can_be_corrected(self):
        model = FakeFunctionModel([
            tool_call("bad-args", "search_api", {"query": "alarms", "unexpected": True}),
            tool_call("good-call", "execute_python", {"code": "value = 40"}),
            tool_call("finish", "execute_python", {"code": "apis.supervisor.complete_task()"}),
        ])
        world = FakeFunctionWorld()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = LangGraphAgent(model, max_steps=4).solve(
                world, trace_path=Path(temp_dir) / "trace.jsonl"
            )
        self.assertEqual(result["stop_reason"], "appworld_task_completed")
        self.assertIn("Unknown argument", model.requests[1]["messages"][-1]["content"])
        self.assertEqual(result["python_execution_count"], 2)

    def test_missing_tool_call_allows_only_one_protocol_correction(self):
        model = FakeFunctionModel([
            {"role": "assistant", "content": "plain text", "tool_calls": []},
            {"role": "assistant", "content": "still plain text", "tool_calls": []},
        ])
        world = FakeFunctionWorld()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = LangGraphAgent(model, max_steps=5).solve(
                world, trace_path=Path(temp_dir) / "trace.jsonl"
            )
        self.assertEqual(result["stop_reason"], "invalid_tool_protocol")
        self.assertEqual(len(model.requests), 2)
        self.assertEqual(world.executed, [])


if __name__ == "__main__":
    unittest.main()
