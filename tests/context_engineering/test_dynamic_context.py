import json
import tempfile
import unittest
from pathlib import Path

from context_engineering.dynamic import DynamicContextBuilder
from memory.store import ObservationStore
from agent.react import LangGraphAgent


class StoreTests(unittest.TestCase):
    def test_lossless_pages_index_and_task_scope(self):
        store = ObservationStore("task-a")
        raw = "A🚀" * 5000
        call_id = "call-1"
        record = store.add(tool_call_id=call_id, tool_name="execute_python", step=1,
                           content=raw, arguments=json.dumps({"code": "apis.venmo.show_transactions()"}),
                           outcome={"ok": True, "result": raw})
        self.assertEqual(record.content, raw)
        self.assertEqual(record.api_names, ("venmo.show_transactions",))
        self.assertEqual(store.search_history("venmo.show_transactions")[0]["observation_id"], record.id)
        self.assertEqual(store.search_history(record.id)[0]["source_tool_call_id"], call_id)
        self.assertEqual(store.search_history("unfindableword"), [])
        offset, parts = 0, []
        while True:
            page = store.read_observation(record.id, offset=offset, limit=137)
            parts.append(page["content"])
            if page["next_offset"] is None:
                break
            offset = page["next_offset"]
        self.assertEqual("".join(parts), raw)
        other = ObservationStore("task-b")
        self.assertEqual(other.search_history(record.id), [])
        with self.assertRaises(ValueError):
            other.read_observation(record.id)
        with self.assertRaises(ValueError):
            store.read_observation(record.id, limit=5000)

    def test_view_pairs_recent_and_full_unchanged(self):
        store = ObservationStore("task-a")
        messages = [{"role": "system", "content": "rules"},
                    {"role": "user", "content": "original task"}]
        for step, tool in enumerate(("get_api_schema", "execute_python", "search_api", "execute_python"), 1):
            call_id = f"call-{step}"
            arguments = json.dumps({"api_name": "venmo.login"} if tool == "get_api_schema" else
                                   {"query": "venmo login"} if tool == "search_api" else
                                   {"code": "apis.venmo.login()"})
            messages.append({"role": "assistant", "content": None, "tool_calls": [{
                "id": call_id, "type": "function", "function": {"name": tool, "arguments": arguments}}]})
            result = "large result " * 500 if step == 2 else f"result-{step}"
            content = json.dumps({"result": result})
            messages.append({"role": "tool", "name": tool, "tool_call_id": call_id,
                             "content": content})
            store.add(tool_call_id=call_id, tool_name=tool, step=step,
                      content=content, arguments=arguments, outcome={"result": result})
        before = json.dumps(messages)
        full, _ = DynamicContextBuilder(store, strategy="full").build(messages)
        self.assertIs(full, messages)
        view, stats = DynamicContextBuilder(store).build(messages)
        self.assertEqual(json.dumps(messages), before)
        self.assertEqual(view[:2], messages[:2])
        self.assertEqual(len(view), len(messages))
        self.assertEqual(view[-1], messages[-1])  # latest observation stays full
        self.assertEqual(view[-3], messages[-3])
        self.assertIn("observation_id", view[3]["content"])
        self.assertIn("observation_id", view[5]["content"])
        self.assertEqual(len(stats["offloaded_ids"]), 2)
        for assistant, tool in zip(view[2::2], view[3::2]):
            self.assertEqual(assistant["tool_calls"][0]["id"], tool["tool_call_id"])
            if "observation_id" in tool["content"]:
                obs_id = json.loads(tool["content"])["observation_id"]
                self.assertEqual(store.read_observation(obs_id, limit=4096)["source_tool_call_id"],
                                 tool["tool_call_id"])


class FakeWorld:
    task_id = "fake-dynamic"
    task = type("Task", (), {"instruction": "Finish safely", "allowed_apps": ["phone"],
                              "api_docs": {"phone": {"show_alarms": {
                                  "app_name": "phone", "api_name": "show_alarms",
                                  "description": "List alarms", "parameters": []}}}})()

    def __init__(self):
        self.finished = False

    def execute(self, code):
        if "complete_task" in code:
            self.finished = True
            return "completed"
        return "printed " + "x" * 6000

    def task_completed(self):
        return self.finished


class FakeModel:
    model_name = "fake"

    def __init__(self):
        self.requests = []
        self.observation_id = None

    def generate_tool_calls(self, messages, tools, *, tool_choice="required"):
        self.requests.append((messages, tools))
        turn = len(self.requests)
        if turn == 1:
            name, args = "search_api", {"query": "alarms"}
        elif turn == 2:
            name, args = "get_api_schema", {"api_name": "phone.show_alarms"}
        elif turn == 3:
            name, args = "execute_python", {"code": "print('large')"}
        elif turn == 4:
            name, args = "search_history", {"query": "phone.show_alarms"}
        elif turn == 5:
            old = next(m for m in messages if m.get("role") == "tool" and
                       "observation_id" in m["content"])
            self.observation_id = json.loads(old["content"])["observation_id"]
            name, args = "read_observation", {"observation_id": self.observation_id, "limit": 512}
        else:
            name, args = "execute_python", {"code": "apis.supervisor.complete_task()"}
        return {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"call-{turn}", "type": "function", "function": {
                "name": name, "arguments": json.dumps(args)}}]}, {
                    "input_tokens": 1, "input_cache_hit_tokens": 0,
                    "output_tokens": 1, "cost_usd": 0.0}


class GraphTests(unittest.TestCase):
    def test_dynamic_graph_retrieval_and_original_history(self):
        model = FakeModel()
        with tempfile.TemporaryDirectory() as directory:
            state = LangGraphAgent(model, context_mode="dynamic", max_steps=7).solve(
                                       FakeWorld(), trace_path=Path(directory) / "trace.jsonl")
            trace = [json.loads(line) for line in (Path(directory) / "trace.jsonl").read_text().splitlines()]
        self.assertEqual(state["stop_reason"], "appworld_task_completed")
        self.assertEqual(state["tool_call_count"], 6)
        self.assertIsNotNone(model.observation_id)
        self.assertTrue(any(row.get("event") == "context_view" and row.get("offloaded_count", 0)
                            for row in trace))
        self.assertEqual([m["role"] for m in state["messages"]].count("tool"), 6)
        self.assertIn("printed", state["messages"][7]["content"])
        self.assertIn("read_observation", [t["function"]["name"] for t in model.requests[0][1]])


if __name__ == "__main__":
    unittest.main()
