"""Replay redacted historical tool messages; never run AppWorld or a model."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

SCRIPT = Path(__file__).resolve()
ROOT = next((parent for parent in SCRIPT.parents
             if (parent / "runs/agentlab_codex_luna_frozen10_20261121_165531").exists()),
            SCRIPT.parents[1])
STAGED_SRC = ROOT / ".v1_2_b_stage/src"
sys.path.insert(0, str(STAGED_SRC if STAGED_SRC.exists() else ROOT / "src"))

from agentlab.context_engineering.dynamic import DynamicContextBuilder
from agentlab.memory.store import ObservationStore
from agentlab.agent.prompt import ContextManager
from agentlab.tool_engineering.tools import ToolRegistry

RUN = ROOT / "runs/agentlab_codex_luna_frozen10_20261121_165531"
BASELINE = ROOT / "runs/agentlab_v1_1_d_budget_relaxation/progress.json"
OUT = ROOT / "reports/v1_2_b_offline_metrics.json"


def definition_list(dynamic: bool) -> list[dict]:
    noop = lambda **_: None
    return ToolRegistry(search_api=noop, get_api_schema=noop, execute_python=noop,
                        search_history=noop if dynamic else None,
                        read_observation=noop if dynamic else None).definitions


def bytes_of(messages: list[dict], tools: list[dict]) -> int:
    return len(json.dumps({"messages": messages, "tools": tools},
                          ensure_ascii=False, default=str).encode("utf-8"))


def validate_pairs(messages: list[dict]) -> bool:
    pending: set[str] = set()
    for message in messages:
        if message.get("role") == "assistant":
            pending.update(call["id"] for call in message.get("tool_calls", []))
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                return False
            pending.remove(call_id)
    return not pending


def replay(task_id: str, instruction: str) -> dict:
    directory = RUN / task_id
    events = [json.loads(line) for line in (directory / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    steps = {event["step"]: event for event in events if event.get("event") == "step"}
    calls: dict[int, list[dict]] = {}
    for event in events:
        if event.get("event") == "tool_call":
            calls.setdefault(event["step"], []).append(event)
    store = ObservationStore(task_id)
    messages = ContextManager(task_id, instruction).messages
    dynamic_initial = ContextManager(task_id, instruction, context_mode="dynamic").messages
    modes = {mode: {"sum_bytes": 0, "max_bytes": 0, "offloaded_refs": 0,
                    "reduction_by_tool_bytes": Counter(), "valid_pairs": True,
                    "recent_full": True}
             for mode in ("full", "dynamic")}
    tools_full = definition_list(False)
    tools_dynamic = definition_list(True)
    offloaded_ids = set()
    for number in range(1, summary["steps"] + 1):
        for mode, metrics in modes.items():
            builder = DynamicContextBuilder(store, strategy=mode)
            if mode == "full":
                view, stats = builder.build(messages)
                tools = tools_full
            else:
                # Use the production dynamic system message, preserving original task.
                with_help = [dynamic_initial[0], *messages[1:]]
                view, stats = builder.build(with_help)
                tools = tools_dynamic
            size = bytes_of(view, tools)
            metrics["sum_bytes"] += size
            metrics["max_bytes"] = max(metrics["max_bytes"], size)
            metrics["offloaded_refs"] += len(stats["offloaded_ids"])
            metrics["valid_pairs"] &= validate_pairs(view)
            if messages[-1].get("role") == "tool":
                metrics["recent_full"] &= view[-1]["content"] == messages[-1]["content"]
            for record_id in stats["offloaded_ids"]:
                record = store._records[record_id]
                original = next(m["content"] for m in messages if m.get("role") == "tool"
                                and m.get("tool_call_id") == record.tool_call_id)
                compact = next(m["content"] for m in view if m.get("role") == "tool"
                               and m.get("tool_call_id") == record.tool_call_id)
                metrics["reduction_by_tool_bytes"][record.tool_name] += (
                    len(original.encode("utf-8")) - len(compact.encode("utf-8")))
                offloaded_ids.add(record_id)
        step = steps.get(number)
        if step is None:
            continue
        assistant = step.get("assistant_message")
        if assistant:
            messages.append(assistant)
        for event in calls.get(number, []):
            outcome = event["tool_result"]
            content = json.dumps(outcome, ensure_ascii=False, default=str)
            tool_message = {"role": "tool", "tool_call_id": event["tool_call_id"],
                            "name": event["tool_name"], "content": content}
            messages.append(tool_message)
            store.add(tool_call_id=event["tool_call_id"], tool_name=event["tool_name"],
                      step=number, content=content, arguments=event.get("arguments", "{}"),
                      outcome=outcome)
    restored, source_ok, search_ok = 0, 0, 0
    for record_id in offloaded_ids:
        record = store._records[record_id]
        position, parts = 0, []
        while True:
            page = store.read_observation(record_id, offset=position, limit=4096)
            parts.append(page["content"])
            source_ok += page["source_tool_call_id"] == record.tool_call_id
            if page["next_offset"] is None:
                break
            position = page["next_offset"]
        restored += "".join(parts) == record.content
        search_ok += bool(store.search_history(record_id)) and (
            store.search_history(record_id)[0]["observation_id"] == record_id)
    return {"task_id": task_id, "steps": summary["steps"],
            "modes": {mode: {**data, "reduction_by_tool_bytes": dict(data["reduction_by_tool_bytes"])}
                      for mode, data in modes.items()},
            "offloaded_unique": len(offloaded_ids), "restored_exact": restored,
            "source_verified_pages": source_ok, "search_exact_found": search_ok}


def main() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))["completed_attempts"]
    instructions = {row["task_id"]: row["instruction"] for row in baseline}
    selection = json.loads((RUN / "frozen_task_selection.json").read_text(encoding="utf-8"))
    tasks = [replay(item["task_id"], instructions[item["task_id"]]) for item in selection]
    OUT.write_text(json.dumps({"method": "redacted trace replay; UTF-8 bytes, not tokens",
                               "tasks": tasks}, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in tasks:
        full = row["modes"]["full"]["sum_bytes"]
        print(row["task_id"], "full", full,
              "dynamic=" + str(row["modes"]["dynamic"]["sum_bytes"]),
              "recovery", f"{row['restored_exact']}/{row['offloaded_unique']}")


if __name__ == "__main__":
    main()
