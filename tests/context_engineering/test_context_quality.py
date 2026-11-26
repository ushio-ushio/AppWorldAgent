"""V1.4-R quality tests: content-level dedup and lossless compaction."""
import json

import pytest

from memory.store import ObservationStore
from context_engineering.unified import UnifiedContextBuilder, UnifiedContextConfig
from runtime.task_state import TaskState
from context_engineering.compaction import TrajectoryCompactor
from memory.store import WorkingMemory


def add_exchange(store, messages, step, name, args, outcome):
    call_id = f"c{step}"
    raw = json.dumps(args)
    messages.append({"role": "assistant", "tool_calls": [{"id": call_id, "type": "function",
                     "function": {"name": name, "arguments": raw}}], "content": None})
    content = json.dumps(outcome)
    messages.append({"role": "tool", "name": name, "tool_call_id": call_id, "content": content})
    return store.add(tool_call_id=call_id, tool_name=name, step=step, content=content,
                     arguments=raw, outcome=outcome)


def setup():
    store = ObservationStore("quality-task")
    memory = WorkingMemory(store)
    state = TaskState("quality-task", "Pay amount for bill", store)
    messages = [{"role": "system", "content": "rules"},
                {"role": "user", "content": "Task: Pay amount for bill"}]
    config = UnifiedContextConfig(memory_selection=True, compaction=False,
                                  manual_retrieval=True, dependency_budget_bytes=0)
    return store, memory, state, messages, config


def test_exact_visible_value_dedup_then_pointer_restores_memory():
    store, memory, state, messages, config = setup()
    args = {"code": "print(apis.venmo.show_payment())"}
    outcome = {"ok": True, "result": json.dumps({"payment_id": "p-1", "amount": 50})}
    source = add_exchange(store, messages, 1, "execute_python", args, outcome)
    assert len(memory.ingest(source.id, outcome, json.dumps(args))) == 1
    builder = UnifiedContextBuilder(store, memory, state, config)
    view, stats = builder.build(messages)
    assert stats["memory_count"] == 0
    assert stats["memory_selection"]["duplicate_skipped"] == 1
    assert not any("WORKING MEMORY" in str(m.get("content", "")) for m in view)
    add_exchange(store, messages, 2, "search_api", {"query": "payment"}, {"ok": True, "result": []})
    add_exchange(store, messages, 3, "get_api_schema", {"api_name": "venmo.login"},
                 {"ok": True, "result": {"api_name": "venmo.login"}})
    # Force the old Python result into V1.2-B's reference-only view.
    assert source.size_bytes < 2048
    large = {"ok": True, "result": json.dumps({"payment_id": "p-2", "amount": 60,
                                                 "padding": "x" * 3000})}
    # The same source ID alone is not a reason to suppress the fact: replace
    # the original record content in a fresh store with a large source.
    store2, memory2, state2, messages2, config2 = setup()
    source2 = add_exchange(store2, messages2, 1, "execute_python", args, large)
    memory2.ingest(source2.id, large, json.dumps(args))
    add_exchange(store2, messages2, 2, "search_api", {"query": "payment"}, {"ok": True, "result": []})
    add_exchange(store2, messages2, 3, "get_api_schema", {"api_name": "venmo.login"},
                 {"ok": True, "result": {"api_name": "venmo.login"}})
    view2, stats2 = UnifiedContextBuilder(store2, memory2, state2, config2).build(messages2)
    assert source2.id in stats2["offloaded_ids"]
    assert stats2["memory_count"] >= 1
    assert "WORKING MEMORY" in json.dumps(view2)


def test_ordinary_metadata_loses_budget_priority_to_amount():
    store, memory, state, messages, config = setup()
    source = add_exchange(store, messages, 1, "execute_python",
                          {"code": "print(apis.venmo.show_payment())"},
                          {"ok": True, "result": json.dumps({"payment_id": "p-1", "pinned": True,
                                                               "amount": 50, "padding": "x" * 3000})})
    memory.ingest(source.id, {"ok": True, "result": json.dumps({"payment_id": "p-1",
                         "pinned": True, "amount": 50, "padding": "x" * 3000})},
                  json.dumps({"code": "print(apis.venmo.show_payment())"}))
    add_exchange(store, messages, 2, "search_api", {"query": "amount"}, {"ok": True, "result": []})
    add_exchange(store, messages, 3, "get_api_schema", {"api_name": "venmo.login"},
                 {"ok": True, "result": {"api_name": "venmo.login"}})
    config = UnifiedContextConfig(memory_selection=True, compaction=False,
                                  dependency_budget_bytes=0, memory_budget_bytes=750)
    view, _ = UnifiedContextBuilder(store, memory, state, config).build(messages)
    segment = next(m["content"] for m in view if str(m.get("content", "")).startswith("WORKING MEMORY"))
    entries = json.loads(segment.split("\n", 1)[1])
    assert entries[0]["key"] == "amount"


def test_compaction_preserves_errors_incomplete_calls_and_argument_pages():
    store = ObservationStore("quality-task")
    messages = [{"role": "system", "content": "rules"},
                {"role": "user", "content": "Task: investigate"}]
    first = add_exchange(store, messages, 1, "execute_python", {"code": "x" * 7000},
                         {"ok": False, "error": "ValueError: bad argument"})
    for step in range(2, 9):
        add_exchange(store, messages, step, "get_api_schema", {"api_name": "venmo.show_payment"},
                     {"ok": True, "result": {"schema": step}})
    # A pending call has no result and must remain in the visible view.
    messages.append({"role": "assistant", "content": None, "tool_calls": [{
        "id": "pending", "type": "function", "function": {
            "name": "execute_python", "arguments": "{}"}}]})
    before = json.dumps(messages)
    compactor = TrajectoryCompactor(store, recent_groups=3)
    view, stats = compactor.compact(messages)
    assert stats["compacted_groups"] == 5
    assert view[-1] == messages[-1]
    checkpoint = next(m["content"] for m in view if str(m.get("content", "")).startswith("Compacted prior"))
    assert "business-goal success" in checkpoint and "ValueError" in checkpoint
    assert first.id in stats["compacted_observation_ids"]
    offset, pages = 0, []
    while True:
        page = store.read_observation(first.id, field="arguments", offset=offset, limit=777)
        pages.append(page["content"])
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    assert "".join(pages) == first.arguments
    assert store.read_observation(first.id)["content"] == first.content[:2048]
    assert json.dumps(messages) == before
    with pytest.raises(ValueError):
        store.read_observation(first.id, field="code")
    # Rebuilding a longer history remains deterministic and does not create
    # dangling Tool Results or remove the pending action.
    repeat, _ = compactor.compact(messages)
    assert repeat == view


def test_long_history_repeated_compaction_keeps_pairs_and_evidence():
    store = ObservationStore("long-task")
    messages = [{"role": "system", "content": "rules"},
                {"role": "user", "content": "Task: long workflow"}]
    compactor = TrajectoryCompactor(store, recent_groups=4)
    for step in range(1, 41):
        record = add_exchange(store, messages, step, "execute_python",
                              {"code": f"print({step})"}, {"ok": step % 7 != 0,
                              "result": str(step), **({"error": "ValueError: retry"} if step % 7 == 0 else {})})
        view, stats = compactor.compact(messages)
        assert stats["compacted_groups"] == max(0, step - 4)
        assert view[-8:] == messages[-8:]
        pending = set()
        for message in view:
            if message.get("role") == "assistant":
                pending.update(call["id"] for call in message.get("tool_calls", []))
            elif message.get("role") == "tool":
                assert message["tool_call_id"] in pending
                pending.remove(message["tool_call_id"])
        assert not pending
        assert store.read_observation(record.id, field="arguments")["content"] == record.arguments
    assert len(messages) == 82  # original history never compacted in place
