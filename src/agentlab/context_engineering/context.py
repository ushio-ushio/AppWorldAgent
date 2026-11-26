"""Cache-stable context views for the opt-in ``cache_first`` mode."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

from agentlab.memory.store import ObservationStore
from agentlab.runtime.trace import _redact
from agentlab.context_engineering.compaction import EpochCompactor


@dataclass(frozen=True)
class CacheFirstContextConfig:
    inline_observation_chars: int = 2500
    preview_chars: int = 220
    keep_recent_groups: int = 4
    compact_trigger_tokens: int = 16000
    max_context_tokens: int = 100000
    enable_async_extraction: bool = True
    soft_extract_tokens: int = 12000
    level3_trigger_ratio: float = 0.80
    min_extract_groups: int = 4
    min_extract_tokens: int = 3500

    def __post_init__(self) -> None:
        if (self.inline_observation_chars < 0 or self.preview_chars < 1 or
                self.keep_recent_groups < 1 or self.compact_trigger_tokens < 1 or
                self.max_context_tokens < self.compact_trigger_tokens or
                not 0 < self.level3_trigger_ratio <= 1 or self.min_extract_groups < 1 or
                self.min_extract_tokens < 1):
            raise ValueError("Cache-first context limits must be non-negative")


class TokenMeter:
    def __init__(self, config: CacheFirstContextConfig):
        self.config = config
        self.previous_serialized: str | None = None
        self.calibration_ratio: float | None = None
        self.actual_cached_tokens = 0

    @staticmethod
    def _lcp(left: str, right: str) -> int:
        i = 0
        while i < min(len(left), len(right)) and left[i] == right[i]: i += 1
        return i

    def measure(self, serialized: str, usage: dict[str, Any] | None, shard_count: int,
                compact_level: int) -> dict[str, Any]:
        chars = len(serialized)
        estimated = int(chars / 4 * (self.calibration_ratio or 1.0))
        actual_input = int((usage or {}).get("input_tokens", 0))
        cached = int((usage or {}).get("input_cache_hit_tokens", (usage or {}).get("cached_tokens", 0)))
        if actual_input and chars:
            self.calibration_ratio = actual_input / max(1, chars / 4)
        lcp = self._lcp(self.previous_serialized, serialized) if self.previous_serialized is not None else 0
        lcp_ratio = round(lcp/max(1, min(len(serialized), len(self.previous_serialized or serialized))), 6)
        self.previous_serialized = serialized
        self.actual_cached_tokens = cached
        return {"estimated_tokens": estimated, "actual_input_tokens_previous": actual_input,
                "cached_tokens": cached, "remaining_capacity_tokens": self.config.max_context_tokens-estimated,
                "lcp_chars": lcp, "lcp_ratio": lcp_ratio,
                "compact_level": compact_level, "l1_shard_count": shard_count,
                "soft_watermark_tokens": self.config.compact_trigger_tokens,
                "hard_capacity_tokens": self.config.max_context_tokens}


class CacheFirstContextBuilder:
    """Freeze the task prefix and each observation on first model-visible entry.

    The source message list remains untouched. Phase 1 keeps its L2 in memory and
    uses the task-local ObservationStore as the exact-value retrieval backend.
    """

    _PROTECTED_MARKERS = ("[Pagination Hint]", "AGENTLAB_BATCH_CHECK")
    _KNOWN_LIST_KEYS = ("items", "results", "records", "data", "messages", "files")
    _KNOWN_REQUEST_STATUSES = {"succeeded", "failed", "blocked"}

    def __init__(self, store: ObservationStore,
                 config: CacheFirstContextConfig | None = None, *, model: Any = None):
        self.store = store
        self.config = config or CacheFirstContextConfig()
        self._l0: list[dict[str, Any]] | None = None
        self._frozen_outputs: dict[str, str] = {}
        self._request_events: dict[str, tuple[dict[str, str], ...]] = {}
        self._pruned_groups: set[tuple[str, ...]] = set()
        self.meter = TokenMeter(self.config)
        self.compactor = EpochCompactor(store, keep_recent_groups=self.config.keep_recent_groups)
        self._archived_ranges: list[tuple[int, int]] = []
        from agentlab.context_engineering.extraction import AsyncContextExtractor
        self.extractor = (AsyncContextExtractor(model) if model is not None and
                          self.config.enable_async_extraction and
                          callable(getattr(model, "extract_context", None)) else None)
        self._extracted_cursor: int | None = None

    def record_request_events(self, tool_call_id: str, events: Iterable[Any] | None) -> None:
        """Keep only non-sensitive outcome fields needed by the zero-mutation guard."""
        rows = []
        for event in events or ():
            if isinstance(event, dict):
                status = event.get("execution_status", event.get("status", ""))
                operation = event.get("operation_type", "")
                method = event.get("method", "")
            else:
                status = getattr(event, "execution_status", "")
                operation = getattr(event, "operation_type", "")
                method = getattr(event, "method", "")
            status = getattr(status, "value", status)
            operation = getattr(operation, "value", operation)
            rows.append({"status": str(status).lower(),
                         "operation": str(operation).lower(),
                         "method": str(method).upper()})
        self._request_events[tool_call_id] = tuple(rows)

    @staticmethod
    def _decode(value: Any) -> Any:
        for _ in range(2):
            if not isinstance(value, str):
                break
            try:
                value = json.loads(value)
            except (TypeError, ValueError):
                break
        return value

    @classmethod
    def _strings(cls, value: Any) -> Iterable[str]:
        value = cls._decode(value)
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from cls._strings(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from cls._strings(item)

    @classmethod
    def _item_count(cls, content: str) -> int | None:
        value = cls._decode(content)
        if isinstance(value, dict) and "result" in value:
            value = cls._decode(value["result"])
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict):
            for key in cls._KNOWN_LIST_KEYS:
                rows = value.get(key)
                if isinstance(rows, list):
                    return len(rows)
        return None

    def _large_handle(self, record: Any) -> str:
        content = record.content
        protected = []
        seen = set()
        for text in [content, *self._strings(content)]:
            for line in text.splitlines():
                if any(marker in line for marker in self._PROTECTED_MARKERS) and line not in seen:
                    seen.add(line)
                    protected.append(line)
        handle = {
            "kind": "observation_handle",
            "observation_id": record.id,
            "source_tool_call_id": record.tool_call_id,
            "source_tool": record.tool_name,
            "step": record.step,
            "total_chars": len(content),
            "size_bytes": record.size_bytes,
            "item_count_estimate": self._item_count(content),
            "head_preview": _redact(content[:self.config.preview_chars]),
            "tail_preview": _redact(content[-self.config.preview_chars:]),
            "protected_lines": protected,
            "recovery": "Use read_observation(observation_id, offset, limit) to recover exact pages; do not repeat the source action.",
        }
        return "Large tool observation stored by reference (not a new result):\n" + json.dumps(
            handle, ensure_ascii=False, separators=(",", ":"), default=str)

    def _freeze_output(self, message: dict[str, Any]) -> str:
        call_id = str(message.get("tool_call_id", ""))
        if call_id in self._frozen_outputs:
            return self._frozen_outputs[call_id]
        content = str(message.get("content", ""))
        record = self.store.for_call(call_id)
        source = record.content if record is not None else content
        frozen = (self._large_handle(record) if record is not None and record.tool_name != "read_observation" and
                  len(source) > self.config.inline_observation_chars else content)
        self._frozen_outputs[call_id] = frozen
        return frozen

    @staticmethod
    def _tool_outcome(message: dict[str, Any]) -> dict[str, Any] | None:
        try:
            result = json.loads(str(message.get("content", "")))
        except (TypeError, ValueError):
            return None
        return result if isinstance(result, dict) else None

    def _has_successful_write(self, call_id: str) -> bool:
        record = self.store.for_call(call_id)
        events = self._request_events.get(call_id, ())
        if events:
            for event in events:
                status = event["status"]
                read_only = event["operation"] in {"read", "operationtype.read"} or event["method"] in {"GET", "HEAD"}
                if status == "succeeded" and not read_only:
                    return True
                if status not in self._KNOWN_REQUEST_STATUSES:
                    return True  # An unknown outcome is protected from pruning.
            return False
        if record is None or record.tool_name != "execute_python":
            return False
        requests = record.executed_requests
        if requests is None:
            return True  # No evidence that the failed Python action was mutation-free.
        return any(str(row.get("method", "")).upper() not in {"GET", "HEAD"}
                   for row in requests)

    def _interaction_groups(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups = []
        index = 2
        while index < len(messages):
            message = messages[index]
            calls = message.get("tool_calls", []) if message.get("role") == "assistant" else []
            call_ids = tuple(str(call.get("id", "")) for call in calls)
            if not call_ids or any(not call_id for call_id in call_ids) or len(set(call_ids)) != len(call_ids):
                index += 1
                continue
            expected = set(call_ids)
            outputs = []
            end = index + 1
            while end < len(messages) and messages[end].get("role") == "tool":
                output = messages[end]
                if output.get("tool_call_id") not in expected:
                    break
                outputs.append(output)
                end += 1
            actual_ids = [str(output.get("tool_call_id", "")) for output in outputs]
            if len(actual_ids) != len(call_ids) or set(actual_ids) != expected:
                index += 1
                continue
            outcomes = [self._tool_outcome(output) for output in outputs]
            failed = any(outcome is None or outcome.get("ok") is not True
                         or outcome.get("error") or outcome.get("execution_error")
                         for outcome in outcomes)
            groups.append({"start": index, "end": end, "call_ids": call_ids,
                           "failed": failed, "success": not failed,
                           "side_effect": any(self._has_successful_write(call_id)
                                              for call_id in call_ids)})
            index = end
        return groups

    def _mark_resolved_errors(self, groups: list[dict[str, Any]]) -> list[tuple[str, ...]]:
        newly_pruned = []
        if not groups or not groups[-1]["success"]:
            return newly_pruned
        current = groups[-1]
        for previous in reversed(groups[:-1]):
            if previous["end"] != current["start"]:
                break
            if not previous["failed"] or previous["side_effect"]:
                break
            key = previous["call_ids"]
            if key not in self._pruned_groups:
                self._pruned_groups.add(key)
                newly_pruned.append(key)
            current = previous
        return newly_pruned

    @staticmethod
    def _runtime_tail(*, step: int | None, max_steps: int | None,
                      usage: dict[str, Any] | None, max_cost_usd: float | None,
                      max_input_tokens: int | None, max_output_tokens: int | None,
                      volatile_context: str | None) -> dict[str, str]:
        parts = []
        if step is not None or max_steps is not None:
            parts.append(f"step={step if step is not None else '?'} / max_steps={max_steps if max_steps is not None else '?'}")
        if usage:
            parts.append("usage_before_request=" + json.dumps({
                key: usage.get(key, 0) for key in
                ("input_tokens", "input_cache_hit_tokens", "output_tokens", "cost_usd")},
                ensure_ascii=False, separators=(",", ":")))
        if max_input_tokens is not None:
            parts.append(f"max_input_tokens={max_input_tokens}")
        if max_output_tokens is not None:
            parts.append(f"max_output_tokens_total={max_output_tokens}")
        if max_cost_usd is not None:
            parts.append(f"max_cost_usd={max_cost_usd}")
        if volatile_context:
            parts.append(volatile_context)
        return {"role": "user", "content": "VOLATILE RUNTIME TAIL (current status only):\n" + "\n".join(parts)}

    def build(self, messages: list[dict[str, Any]], *, step: int | None = None,
              max_steps: int | None = None, usage: dict[str, Any] | None = None,
              max_cost_usd: float | None = None, max_input_tokens: int | None = None,
              max_output_tokens: int | None = None,
              volatile_context: str | None = None, tool_definitions: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if len(messages) < 2:
            raise ValueError("Cache-first context requires a system message and an original task message")
        static = deepcopy(messages[:2])
        if self._l0 is None:
            self._l0 = static
        elif static != self._l0:
            raise ValueError("Cache-first L0 messages changed after initialization")

        groups = self._interaction_groups(messages)
        semantic_memory = None
        if self.extractor:
            records = self.store.records()
            valid_refs = {record.id for record in records}
            valid_refs.update(f"step_{record.step}" for record in records)
            semantic_memory = self.extractor.poll(valid_refs)
            if semantic_memory:
                end = semantic_memory.get("covered_step_range")
                if isinstance(end, list) and len(end) == 2:
                    covered_step = end[1]
                    self._extracted_cursor = sum(1 for group in groups
                        if any((self.store.for_call(cid) and self.store.for_call(cid).step <= covered_step)
                               for cid in group["call_ids"]))
        newly_pruned = self._mark_resolved_errors(groups)
        removed = {index for group in groups if group["call_ids"] in self._pruned_groups
                   for index in range(group["start"], group["end"])}
        removed.update(index for group in groups if tuple(group["call_ids"]) in self.compactor.archived_groups
                       for index in range(group["start"], group["end"]))
        compact_level = 0
        provisional = [*deepcopy(self._l0), *[m for i,m in enumerate(messages[2:],2) if i not in removed],
                       self._runtime_tail(step=step,max_steps=max_steps,usage=usage,max_cost_usd=max_cost_usd,
                                          max_input_tokens=max_input_tokens,max_output_tokens=max_output_tokens,
                                          volatile_context=volatile_context)]
        from agentlab.models.siwc_responses import ChatGPTResponsesAdapter
        provisional_body = {"model": "gpt-6-luna",
                            "input": [ChatGPTResponsesAdapter._tools(tool_definitions or []),
                                      *ChatGPTResponsesAdapter._input(provisional)],
                            "store": False, "stream": True}
        serialized = json.dumps(provisional_body, ensure_ascii=False, default=str)
        precompact_estimated = len(serialized) // 4
        extraction_triggered = bool(self.extractor and
            precompact_estimated >= self.config.soft_extract_tokens and
            self.extractor.submit(provisional, tool_definitions, total_groups=len(groups),
                estimated_tokens=precompact_estimated, min_groups=self.config.min_extract_groups,
                min_tokens=self.config.min_extract_tokens, soft_tokens=self.config.soft_extract_tokens))
        if len(serialized)//4 >= self.config.compact_trigger_tokens:
            safe_errors={tuple(g["call_ids"]) for g in groups if g["failed"] and not g["side_effect"]}
            removed, compact_level, _ = self.compactor.compact(messages, removed, level1_candidates=safe_errors,
                estimated_tokens=len(serialized)//4, extracted_cursor=self._extracted_cursor,
                semantic_memory=semantic_memory)
            if compact_level == 1 and len(serialized)//4 >= self.config.compact_trigger_tokens:
                removed, compact_level, _ = self.compactor.compact(messages, removed, level1_candidates=set(),
                                                                     estimated_tokens=len(serialized)//4)
            if compact_level == 2 and not semantic_memory and self.extractor:
                self.extractor.record_fallback()
        visible = []
        offloaded = []
        for index, message in enumerate(messages[2:], 2):
            if index in removed:
                continue
            row = message.copy()
            if row.get("role") == "tool":
                record = self.store.for_call(str(row.get("tool_call_id", "")))
                frozen = self._freeze_output(row)
                if record is not None and frozen != str(row.get("content", "")):
                    offloaded.append(record.id)
                row["content"] = frozen
            visible.append(row)

        tail = self._runtime_tail(step=step, max_steps=max_steps, usage=usage,
                                  max_cost_usd=max_cost_usd,
                                  max_input_tokens=max_input_tokens,
                                  max_output_tokens=max_output_tokens,
                                  volatile_context=volatile_context)
        shards = [{"role":"user","content":s.rendered_text} for s in self.compactor.shards]
        anchor = ([{"role":"user","content":self.compactor.boundary_anchor}]
                  if self.compactor.boundary_anchor else [])
        view = [*deepcopy(self._l0), *shards, *anchor, *visible, tail]
        l0_bytes = len(json.dumps(self._l0, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        l2_bytes = len(json.dumps(visible, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        tail_bytes = len(json.dumps(tail, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        final_body = {"model": "gpt-6-luna",
                      "input": [ChatGPTResponsesAdapter._tools(tool_definitions or []),
                                *ChatGPTResponsesAdapter._input(view)],
                      "store": False, "stream": True}
        serialized_view = json.dumps(final_body, ensure_ascii=False, default=str)
        meter_stats=self.meter.measure(serialized_view, usage, len(self.compactor.shards), compact_level)
        level3 = False
        if meter_stats["estimated_tokens"] >= self.config.max_context_tokens * self.config.level3_trigger_ratio:
            self.compactor.consolidate_checkpoints(
                unresolved=(semantic_memory or {}).get("unresolved", ()),
                entities=(semantic_memory or {}).get("entities", ()))
            level3 = bool(self.compactor.shards)
            if level3:
                shards = [{"role":"user","content":s.rendered_text} for s in self.compactor.shards]
                view = [*deepcopy(self._l0), *shards,
                        {"role":"user","content":self.compactor.boundary_anchor}, *visible, tail]
                compact_level = 3
                serialized_view = json.dumps({"model":"gpt-6-luna","input":[ChatGPTResponsesAdapter._tools(tool_definitions or []), *ChatGPTResponsesAdapter._input(view)],"store":False,"stream":True},ensure_ascii=False,default=str)
                meter_stats=self.meter.measure(serialized_view, usage, len(self.compactor.shards), compact_level)
        return view, {
            "offloaded_ids": list(dict.fromkeys(offloaded)),
            "frozen_observation_count": len(self._frozen_outputs),
            "newly_pruned_error_groups": [list(group) for group in newly_pruned],
            "l0_message_bytes": l0_bytes,
            "l2_message_bytes": l2_bytes,
            "volatile_tail_bytes": tail_bytes,
            "extraction": ({"trigger_count": self.extractor.triggers,
                            "hit_count": self.extractor.hits,
                            "fallback_count": self.extractor.fallbacks,
                            "usage": dict(self.extractor.usage),
                            "triggered_this_step": extraction_triggered} if self.extractor else {}),
            "visible_bytes": len(json.dumps(view, ensure_ascii=False).encode("utf-8")),
            "prefix_stable": view[:2] == messages[:2],
            "over_trigger_warning": meter_stats["estimated_tokens"] > self.config.compact_trigger_tokens,
            "level3_compacted": level3,
            "replay_layers": [{"name": name, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                               "chars": len(text)} for name, text in (
                ("L0", json.dumps(self._l0, ensure_ascii=False, separators=(",", ":"))),
                ("L1", json.dumps(shards, ensure_ascii=False, separators=(",", ":"))),
                ("L2", json.dumps(visible, ensure_ascii=False, separators=(",", ":"))),
                ("Volatile Tail", json.dumps(tail, ensure_ascii=False, separators=(",", ":"))))],
            **meter_stats,
        }

    def trigger_extraction(self, messages: list[dict[str, Any]], estimated_tokens: int) -> bool:
        if not self.extractor or estimated_tokens < self.config.soft_extract_tokens:
            return False
        return self.extractor.submit(messages)

    def close_extraction(self) -> None:
        if self.extractor:
            self.extractor.cancel()

    def extraction_summary(self) -> dict[str, Any]:
        return self.extractor.summary() if self.extractor else {"triggers": 0, "hits": 0, "fallbacks": 0,
            "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}

    def force_overflow_compaction(self, messages: list[dict[str, Any]]) -> None:
        groups = self._interaction_groups(messages)
        removed = {i for g in groups if tuple(g["call_ids"]) in self.compactor.archived_groups
                   for i in range(g["start"], g["end"])}
        self.compactor.compact(messages, removed, level1_candidates=set(),
                               estimated_tokens=self.config.max_context_tokens * 2,
                               extracted_cursor=self._extracted_cursor)
        if self.compactor.shards:
            self.compactor.consolidate_checkpoints()
