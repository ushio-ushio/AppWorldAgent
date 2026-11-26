import json
from copy import deepcopy
from pathlib import Path
import tempfile

from agent.prompt import ContextManager
from agent.react import LangGraphAgent
from context_engineering.compaction import EpochCompactor, TrajectoryCompactor
from context_engineering.context import CacheFirstContextBuilder, CacheFirstContextConfig, TokenMeter
from context_engineering.extraction import AsyncContextExtractor
from context_engineering.dynamic import DynamicContextBuilder
from memory.store import ObservationStore
from models.siwc_responses import ChatGPTResponsesAdapter


TOOLS = [{
    "type": "function",
    "function": {"name": "execute_python", "description": "Run task code.",
                 "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                                "required": ["code"]}},
}]


def _input_bytes(items):
    return b",".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                    for item in items)


def _append(messages, store, step, *, content=None, outcome=None, executed_requests=()):
    call_id = f"call-{step}"
    args = json.dumps({"code": f"print({step})"})
    outcome = outcome or {"ok": True, "result": f"result-{step}"}
    content = content if content is not None else json.dumps(outcome, ensure_ascii=False)
    messages.extend([
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": call_id, "type": "function", "function": {
                "name": "execute_python", "arguments": args}}]},
        {"role": "tool", "name": "execute_python", "tool_call_id": call_id,
         "content": content},
    ])
    store.add(tool_call_id=call_id, tool_name="execute_python", step=step,
              content=content, arguments=args, outcome=outcome,
              executed_requests=executed_requests)
    return call_id


def _pairs(messages):
    pending = set()
    for message in messages:
        if message.get("role") == "assistant":
            pending.update(call["id"] for call in message.get("tool_calls", []))
        elif message.get("role") == "tool":
            call_id = message["tool_call_id"]
            assert call_id in pending
            pending.remove(call_id)
    assert not pending


def test_static_l0_and_append_only_prefix_with_volatile_assurance_tail():
    store = ObservationStore("prefix-task")
    messages = [{"role": "system", "content": "fixed system prompt"},
                {"role": "user", "content": "Task: keep original constraints. Plan: read-only."}]
    builder = CacheFirstContextBuilder(store, CacheFirstContextConfig(inline_observation_chars=40))
    payloads = []
    l0_views = []

    for step in range(1, 6):
        original = deepcopy(messages)
        view, stats = builder.build(messages, step=step, max_steps=30,
                                    usage={"input_tokens": step * 10},
                                    volatile_context=f"assurance state changes at step {step}")
        assert messages == original
        payload = [ChatGPTResponsesAdapter._tools(TOOLS), *ChatGPTResponsesAdapter._input(view)]
        payloads.append(payload)
        l0_views.append(_input_bytes(payload[:3]))  # additional_tools + system + task
        assert view[-1]["role"] == "user"
        assert f"step={step} / max_steps=30" in view[-1]["content"]
        assert stats["prefix_stable"]
        if step < 5:
            _append(messages, store, step,
                    outcome={"ok": True, "result": "x" * (100 if step == 2 else 8)})

    assert len(set(l0_views)) == 1
    for current, following in zip(payloads, payloads[1:]):
        prefix = _input_bytes(current[:-1])
        next_input = _input_bytes(following)
        assert next_input.startswith(prefix + b",")
        assert len(next_input) > len(prefix)


def test_large_observation_is_frozen_and_exactly_recoverable():
    store = ObservationStore("freeze-task")
    messages = [{"role": "system", "content": "fixed"}, {"role": "user", "content": "task"}]
    builder = CacheFirstContextBuilder(store, CacheFirstContextConfig(inline_observation_chars=80))
    _append(messages, store, 1)
    special = "[Pagination Hint] continue until []\nAGENTLAB_BATCH_CHECK={\"expected\":[1],\"observed\":[1],\"failed\":[]}"
    raw_result = "row-data," * 30 + special
    content = json.dumps({"ok": True, "result": raw_result}, ensure_ascii=False)
    call_id = _append(messages, store, 2, content=content,
                      outcome={"ok": True, "result": raw_result})

    frozen = []
    for step in range(3, 6):
        view, _ = builder.build(messages, step=step, max_steps=30)
        message = next(row for row in view if row.get("role") == "tool" and row.get("tool_call_id") == call_id)
        frozen.append(message["content"])
        if step < 5:
            _append(messages, store, step)

    assert frozen[0] == frozen[1] == frozen[2]
    assert "observation_id" in frozen[0]
    assert "[Pagination Hint] continue until []" in frozen[0]
    handle = json.loads(frozen[0].split("\n", 1)[1])
    assert "AGENTLAB_BATCH_CHECK={\"expected\":[1],\"observed\":[1],\"failed\":[]}" in handle["protected_lines"]
    record = store.for_call(call_id)
    assert record is not None
    assert "row-data," * 30 in store.read_observation(record.id, limit=4096)["content"]


def test_resolved_no_write_error_is_pruned_as_an_atomic_call_pair():
    store = ObservationStore("prune-task")
    messages = [{"role": "system", "content": "fixed"}, {"role": "user", "content": "task"}]
    builder = CacheFirstContextBuilder(store)
    _append(messages, store, 1)
    failed_id = _append(messages, store, 2, outcome={
        "ok": False, "execution_error": "TypeError: local shell error"}, executed_requests=())
    _append(messages, store, 3, outcome={"ok": True, "result": "repaired"})

    view, stats = builder.build(messages, step=4, max_steps=30)
    assert failed_id not in {call["id"] for row in view if row.get("role") == "assistant"
                             for call in row.get("tool_calls", [])}
    assert failed_id not in {row.get("tool_call_id") for row in view if row.get("role") == "tool"}
    assert stats["newly_pruned_error_groups"] == [[failed_id]]
    _pairs(view)
    # The audit trail remains intact even though the model view prunes the resolved pair.
    assert any(row.get("tool_call_id") == failed_id for row in messages if row.get("role") == "tool")


def test_error_with_successful_write_is_never_pruned():
    store = ObservationStore("write-protect-task")
    messages = [{"role": "system", "content": "fixed"}, {"role": "user", "content": "task"}]
    builder = CacheFirstContextBuilder(store)
    _append(messages, store, 1)
    failed_id = _append(messages, store, 2, outcome={
        "ok": False, "execution_error": "RuntimeError after request"},
        executed_requests=({"method": "POST", "api_name": "mail.send"},))
    builder.record_request_events(failed_id, [{"execution_status": "succeeded",
                                               "operation_type": "write", "method": "POST"}])
    _append(messages, store, 3, outcome={"ok": True, "result": "repaired"})

    view, stats = builder.build(messages, step=4, max_steps=30)
    assert failed_id in {call["id"] for row in view if row.get("role") == "assistant"
                         for call in row.get("tool_calls", [])}
    assert stats["newly_pruned_error_groups"] == []
    _pairs(view)


def test_legacy_modes_and_builders_remain_available():
    assert LangGraphAgent(object(), context_mode="full").context_mode == "full"
    assert LangGraphAgent(object(), context_mode="dynamic").context_mode == "dynamic"
    assert LangGraphAgent(object(), context_mode="unified").context_mode == "unified"
    store = ObservationStore("legacy-task")
    messages = [{"role": "system", "content": "fixed"}, {"role": "user", "content": "task"}]
    full, _ = DynamicContextBuilder(store, strategy="full").build(messages)
    assert full is messages
    dynamic, _ = DynamicContextBuilder(store).build(messages)
    assert dynamic[:2] == messages[:2]
    assert isinstance(TrajectoryCompactor(store), TrajectoryCompactor)
    assert "Cache-first history notice" not in ContextManager("t", "task", context_mode="dynamic").messages[0]["content"]
    assert "Cache-first history notice" not in ContextManager("t", "task", context_mode="unified").messages[0]["content"]


class _FakeWorld:
    task_id = "cache-first-fake"
    task = type("Task", (), {"instruction": "Finish the fake task.", "allowed_apps": ["phone"],
                              "api_docs": {"phone": {"show_alarms": {
                                  "app_name": "phone", "api_name": "show_alarms",
                                  "description": "List alarms", "parameters": []}}}})()

    def __init__(self):
        self.finished = False

    def execute(self, code):
        if "apis.supervisor.complete_task" in code:
            self.finished = True
            return "Task completed."
        return "2"

    def task_completed(self):
        return self.finished


class _FakeModel:
    model_name = "offline-fake"

    def __init__(self):
        self.messages = []

    def generate_tool_calls(self, messages, tools, *, tool_choice="required"):
        self.messages.append(messages)
        turn = len(self.messages)
        name, args = (("search_api", {"query": "alarms"}) if turn == 1 else
                      ("execute_python", {"code": "print(1 + 1)"}) if turn == 2 else
                      ("execute_python", {"code": "apis.supervisor.complete_task()"}))
        return {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"fake-{turn}", "type": "function", "function": {
                "name": name, "arguments": json.dumps(args)}}]}, {
                    "input_tokens": 5, "input_cache_hit_tokens": 1,
                    "output_tokens": 2, "cost_usd": 0.0}


def test_cache_first_mode_is_wired_through_react_without_live_model():
    model = _FakeModel()
    with tempfile.TemporaryDirectory() as directory:
        result = LangGraphAgent(model, context_mode="cache_first", max_steps=5).solve(
            _FakeWorld(), trace_path=Path(directory) / "trace.jsonl")
        trace = [json.loads(line) for line in (Path(directory) / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert result["stop_reason"] == "appworld_task_completed"
    assert len(model.messages) == 3
    assert "Cache-first history notice" in model.messages[0][0]["content"]
    assert model.messages[-1][-1]["content"].startswith("VOLATILE RUNTIME TAIL")
    assert any(row.get("event") == "context_view" and "cache_first" in row for row in trace)


def test_token_meter_calibration_and_lcp():
    meter = TokenMeter(CacheFirstContextConfig())
    first = meter.measure("abcdefgh", {"input_tokens": 4, "input_cache_hit_tokens": 1}, 0, 0)
    second = meter.measure("abcdefXY", {"input_tokens": 8, "input_cache_hit_tokens": 2}, 1, 1)
    assert first["estimated_tokens"] == 2
    assert second["lcp_chars"] == 6
    assert second["lcp_ratio"] == 0.75
    assert second["cached_tokens"] == 2
    assert second["remaining_capacity_tokens"] == 99996


def test_epoch_compactor_level1_then_level2_immutable_shards_and_tail():
    store = ObservationStore("epoch")
    messages = [{"role":"system","content":"s"},{"role":"user","content":"t"}]
    removed = set()
    for step in range(1, 9):
        cid=f"c{step}"
        messages.extend([{"role":"assistant","tool_calls":[{"id":cid}]},
                         {"role":"tool","tool_call_id":cid,"content":json.dumps({"ok":True,"result":step})}])
        store.add(tool_call_id=cid,tool_name="search_api",step=step,content=json.dumps({"ok":True,"result":step}),
                  arguments="{}",outcome={"ok":True,"result":step},executed_requests=())
    compactor=EpochCompactor(store,keep_recent_groups=4,min_compact_groups=2,min_growth_tokens=0)
    # A proven no-write error is removed first and does not produce a shard.
    removed,level,count=compactor.compact(messages,removed,level1_candidates={("c1",)})
    assert level == 1 and not compactor.shards and count == 0
    removed,level,count=compactor.compact(messages,removed,level1_candidates=set())
    assert level == 2 and len(compactor.shards)==1
    frozen=compactor.shards[0].rendered_text
    # Add a later complete group and commit a second contiguous checkpoint.
    for step in range(9, 14):
        cid=f"c{step}"; messages.extend([{"role":"assistant","tool_calls":[{"id":cid}]},
                                          {"role":"tool","tool_call_id":cid,"content":"{\"ok\":true}"}])
        store.add(tool_call_id=cid,tool_name="search_api",step=step,content='{"ok":true}',arguments="{}",outcome={"ok":True},executed_requests=())
    removed,level,count=compactor.compact(messages,removed,level1_candidates=set())
    assert level == 2 and compactor.shards[0].rendered_text == frozen
    assert compactor.shards[1].event_range[0] == compactor.shards[0].event_range[1] + 1
    assert not (removed & set(range(len(messages)-8,len(messages))))
    live_ids=[c["id"] for m in messages if m.get("role")=="assistant" for c in m.get("tool_calls",[]) if messages.index(m) not in removed]
    output_ids=[m["tool_call_id"] for i,m in enumerate(messages) if m.get("role")=="tool" and i not in removed]
    assert set(live_ids)==set(output_ids)


def test_epoch_compactor_hysteresis_prevents_shard_thrashing_and_soft_watermark_is_not_hard_stop():
    store=ObservationStore("hysteresis")
    messages=[{"role":"system","content":"s"},{"role":"user","content":"t"}]
    removed=set(); compact=EpochCompactor(store,keep_recent_groups=2)
    def add_group(step):
        cid=f"h{step}"; content='{"ok":true}'
        messages.extend([{"role":"assistant","tool_calls":[{"id":cid}]},
                         {"role":"tool","tool_call_id":cid,"content":content}])
        store.add(tool_call_id=cid,tool_name="search_api",step=step,content=content,arguments="{}",
                  outcome={"ok":True},executed_requests=())
    for step in range(1,6): add_group(step)
    removed,level,_=compact.compact(messages,removed,level1_candidates=set(),estimated_tokens=17000)
    assert level==0  # only three groups outside the protected tail
    add_group(6)
    removed,level,_=compact.compact(messages,removed,level1_candidates=set(),estimated_tokens=17000)
    assert level==2 and len(compact.shards)==1
    for step in range(7,10): add_group(step)
    removed,level,_=compact.compact(messages,removed,level1_candidates=set(),estimated_tokens=18000)
    assert level==0  # group count alone cannot bypass token hysteresis
    config=CacheFirstContextConfig()
    assert config.compact_trigger_tokens==16000 and config.max_context_tokens==100000
    meter=TokenMeter(config).measure("x"*80000,None,0,0)
    assert meter["estimated_tokens"]==20000 and meter["remaining_capacity_tokens"]==80000


def test_archived_groups_stay_out_of_l2_across_following_builds():
    store=ObservationStore("no-rebound")
    builder=CacheFirstContextBuilder(store,CacheFirstContextConfig(compact_trigger_tokens=1,
                                                                   keep_recent_groups=2))
    messages=[{"role":"system","content":"system"},{"role":"user","content":"task"}]
    def append(step):
        cid=f"arc-{step}"; args=json.dumps({"query":f"q{step}"}); result=json.dumps({"ok":True,"result":f"result-{step}"})
        messages.extend([{"role":"assistant","tool_calls":[{"id":cid,"type":"function","function":{"name":"search_api","arguments":args}}]},
                         {"role":"tool","tool_call_id":cid,"content":result}])
        store.add(tool_call_id=cid,tool_name="search_api",step=step,content=result,arguments=args,
                  outcome={"ok":True,"result":f"result-{step}"},executed_requests=())
    for i in range(1,9): append(i)
    turn_n,stats=builder.build(messages,step=9,tool_definitions=TOOLS)
    assert stats["compact_level"]==2 and stats["l1_shard_count"]==1
    archived=set(builder.compactor.archived_groups)
    size_n=stats["visible_bytes"]
    for i in (9,10):
        append(i)
        view,stats=builder.build(messages,step=i+1,tool_definitions=TOOLS)
        visible_ids={m.get("tool_call_id") for m in view if m.get("role")=="tool"}
        assert not any(cid in visible_ids for group in archived for cid in group)
        assert any("CP-01" in m.get("content","") for m in view)
        assert any("Context Boundary" in m.get("content","") for m in view)
        assert stats["visible_bytes"] <= size_n + 1500
        size_n=stats["visible_bytes"]


def test_async_extraction_validates_evidence_and_isolates_terminal_cancel():
    class Model:
        session_id = "task-session"
        def extract_context(self, messages, tools=None):
            assert messages[-1]["role"] == "user"
            return ({"covered_step_range": [1, 2],
                     "facts": [{"text":"supported", "evidence_ref":"obs-ok"},
                               {"text":"hallucinated", "evidence_ref":"obs-missing"}],
                     "entities": [], "progress": [], "unresolved": []},
                    {"input_tokens": 20, "cached_tokens": 3, "output_tokens": 4})
    from time import sleep
    extractor=AsyncContextExtractor(Model())
    assert extractor.submit([{"role":"developer","content":"stable"},
                             {"role":"user","content":"VOLATILE RUNTIME TAIL: step 1"}])
    for _ in range(100):
        result=extractor.poll({"obs-ok"})
        if result is not None: break
        sleep(.005)
    assert result["facts"] == [{"text":"supported", "evidence_ref":"obs-ok"}]
    assert extractor.usage["cached_tokens"] == 3
    extractor.cancel()


def test_compactor_caps_commit_to_extracted_cursor():
    store=ObservationStore("cursor")
    messages=[{"role":"system","content":"s"},{"role":"user","content":"t"}]
    for step in range(1,8):
        cid=f"cur-{step}"; text='{"ok":true}'
        messages.extend([{"role":"assistant","tool_calls":[{"id":cid}]},
                         {"role":"tool","tool_call_id":cid,"content":text}])
        store.add(tool_call_id=cid,tool_name="search_api",step=step,content=text,arguments="{}",
                  outcome={"ok":True},executed_requests=())
    compactor=EpochCompactor(store,keep_recent_groups=2,min_compact_groups=1,min_growth_tokens=0)
    removed,level,count=compactor.compact(messages,set(),level1_candidates=set(),
                                           estimated_tokens=20000,extracted_cursor=2,
                                           semantic_memory={"facts":[{"text":"f","evidence_ref":"obs"}]})
    assert level==2 and count==2 and compactor.shards[0].event_range==(1,2)
    assert '"facts"' in compactor.shards[0].rendered_text
    assert len(removed)==4


def test_soft_watermark_submits_clone_before_compaction_and_counts_fallback():
    from time import sleep
    class SlowModel:
        def extract_context(self, messages, tools=None):
            sleep(.15)
            return ({"covered_step_range":[1,1],"facts":[],"entities":[],"progress":[],"unresolved":[]},
                    {"input_tokens":8,"cached_tokens":2,"output_tokens":3})
    store=ObservationStore("watermark")
    config=CacheFirstContextConfig(soft_extract_tokens=12000,compact_trigger_tokens=16000,
                                   max_context_tokens=100000,keep_recent_groups=1)
    builder=CacheFirstContextBuilder(store,config,model=SlowModel())
    messages=[{"role":"system","content":"s"},{"role":"user","content":"t"}]
    # Cross soft watermark without compacting; submission must happen before any later compact.
    messages[1]["content"]="x"*50000
    _,first=builder.build(messages,tool_definitions=TOOLS)
    assert first["extraction"]["trigger_count"]==1
    for step in range(1,7):
        cid=f"wm-{step}"; result='{"ok":true,"result":"'+('z'*2500)+'"}'
        messages.extend([{"role":"assistant","tool_calls":[{"id":cid,"type":"function","function":{"name":"search_api","arguments":"{}"}}]},
                         {"role":"tool","tool_call_id":cid,"content":result}])
        store.add(tool_call_id=cid,tool_name="search_api",step=step,content=result,arguments="{}",
                  outcome={"ok":True},executed_requests=())
    _,second=builder.build(messages,step=7,tool_definitions=TOOLS)
    assert second["compact_level"]==2
    assert second["extraction"]["fallback_count"] > 0
    assert second["extraction"]["trigger_count"] > 0
    builder.close_extraction()


def test_level3_consolidates_shards_and_preserves_evidence():
    store=ObservationStore("master")
    compactor=EpochCompactor(store)
    from context_engineering.compaction import CheckpointShard
    compactor.shards=[CheckpointShard(f"CP-{i:02d}",(1+(i-1)*2,i*2),
        ({"api":f"app.write{i}","status":"succeeded","evidence_ref":f"obs{i}"},),
        ({"obs_id":f"obs{i}"},),f"old narrative {i}"*100,
        facts=({"text":f"f{i}","evidence_ref":f"obs{i}"},),
        entities=({"id":f"entity{i}","evidence_ref":f"obs{i}"},),
        unresolved=({"text":f"open{i}","evidence_ref":f"obs{i}"},)) for i in range(1,4)]
    master=compactor.consolidate_checkpoints()
    assert len(compactor.shards)==1 and master.shard_id=="CP-MASTER-01"
    assert master.event_range==(1,6) and len(master.mutations)==3
    assert len(master.key_observations)==3 and len(master.entities)==3 and len(master.unresolved)==3
    assert len(master.rendered_text)<sum(len(x.rendered_text) for x in [*compactor.shards]) + 100000
    assert compactor.level3_compactions==1


def test_replay_reports_lcp_cache_and_compaction_counts():
    from context_engineering.replay import analyze_trace
    rows=[{"event":"context_view","step":1,"cache_first":{"lcp_chars":0,"lcp_ratio":0,"cached_tokens":0,"compact_level":0}},
          {"event":"context_view","step":2,"cache_first":{"lcp_chars":8,"lcp_ratio":.8,"cached_tokens":2,"compact_level":3,"extraction":{"triggers":1}}},
          {"event":"overflow_recovery","step":2}]
    report=analyze_trace(rows)
    assert report["adjacent_steps"][0]["lcp_chars"]==8
    assert report["adjacent_steps"][0]["cache_efficiency"]==1
    assert report["compact_counts"]["3"]==1 and report["overflow_recoveries"]==1


def test_clone_incremental_throttle_requires_group_and_token_growth():
    class Model:
        def extract_context(self,messages,tools=None): return ({}, {})
    from context_engineering.extraction import AsyncContextExtractor
    extractor=AsyncContextExtractor(Model())
    assert extractor.submit([{"role":"user","content":"x"}],total_groups=5,estimated_tokens=12000)
    assert not extractor.submit([{"role":"user","content":"x"}],total_groups=8,estimated_tokens=20000)
    # Once the first job finishes, both independent growth limits are enforced.
    import time
    for _ in range(100):
        extractor.poll(set())
        if extractor.future is None: break
        time.sleep(.005)
    assert not extractor.submit([{"role":"user","content":"x"}],total_groups=8,estimated_tokens=16000)
    assert extractor.submit([{"role":"user","content":"x"}],total_groups=9,estimated_tokens=16000)
    extractor.cancel()


def test_react_replays_same_step_after_context_overflow():
    from models.siwc_responses import SIWCContextOverflowError
    class OverflowModel(_FakeModel):
        def __init__(self):
            super().__init__(); self.failed_once=False
        def generate_tool_calls(self,messages,tools,*,tool_choice="required"):
            if not self.failed_once:
                self.failed_once=True
                raise SIWCContextOverflowError("context_length_exceeded")
            return super().generate_tool_calls(messages,tools,tool_choice=tool_choice)
    model=OverflowModel()
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/"trace.jsonl"
        result=LangGraphAgent(model,context_mode="cache_first",max_steps=5).solve(_FakeWorld(),trace_path=path)
        rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert result["stop_reason"]=="appworld_task_completed"
    assert any(row.get("event")=="overflow_recovery" for row in rows)
