"""Deterministic old interaction checkpoints; original messages remain untouched."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agentlab.memory.store import ObservationStore


@dataclass(frozen=True)
class CheckpointShard:
    shard_id: str
    event_range: tuple[int, int]
    mutations: tuple[dict[str, Any], ...]
    key_observations: tuple[dict[str, Any], ...]
    rendered_text: str
    facts: tuple[dict[str, Any], ...] = ()
    entities: tuple[dict[str, Any], ...] = ()
    unresolved: tuple[dict[str, Any], ...] = ()


class EpochCompactor:
    """Deterministic, task-local L1/L2 compaction for cache_first only."""
    def __init__(self, store: ObservationStore, *, keep_recent_groups: int = 4,
                 min_compact_groups: int = 4, min_growth_tokens: int = 3000):
        if keep_recent_groups < 1 or min_compact_groups < 1:
            raise ValueError("compaction group limits must be positive")
        self.store = store
        self.keep_recent_groups = keep_recent_groups
        self.min_compact_groups = min_compact_groups
        self.min_growth_tokens = min_growth_tokens
        self._last_compact_tokens = 0
        self.shards: list[CheckpointShard] = []
        self._covered_until = 0
        self.archived_groups: set[tuple[str, ...]] = set()
        self.boundary_anchor: str | None = None

    def _groups(self, messages: list[dict[str, Any]]) -> list[tuple[int, int, list[str]]]:
        groups, i = [], 2
        while i < len(messages):
            msg = messages[i]
            calls = msg.get("tool_calls", []) if msg.get("role") == "assistant" else []
            ids = [str(c.get("id", "")) for c in calls]
            if not ids or not all(ids):
                i += 1; continue
            j, outputs = i + 1, []
            while j < len(messages) and messages[j].get("role") == "tool" and messages[j].get("tool_call_id") in ids:
                outputs.append(messages[j].get("tool_call_id")); j += 1
            if len(outputs) == len(ids) and set(outputs) == set(ids):
                groups.append((i, j, ids)); i = j
            else:
                i += 1
        return groups

    def compact(self, messages: list[dict[str, Any]], removed: set[int], *,
                level1_candidates: set[tuple[str, ...]], estimated_tokens: int = 0,
                extracted_cursor: int | None = None, semantic_memory: dict[str, Any] | None = None):
        groups = self._groups(messages)
        for start, end, ids in groups:
            if tuple(ids) in self.archived_groups:
                removed.update(range(start, end))
        eligible = groups[:-self.keep_recent_groups]
        # Level 1 candidates are already proven side-effect-free by the builder.
        newly_removed = set()
        level1_count = 0
        for position, (start, end, ids) in enumerate(eligible):
            if tuple(ids) in level1_candidates:
                newly_removed.update(range(start, end))
                if position == level1_count:
                    level1_count += 1
                else:
                    break
        if newly_removed:
            removed.update(newly_removed)
            self._covered_until += level1_count
            return removed, 1, 0
        remaining = [(s, e, ids) for s, e, ids in eligible
                     if tuple(ids) not in self.archived_groups and not any(k in removed for k in range(s, e))]
        if extracted_cursor is not None:
            remaining = remaining[:max(0, extracted_cursor)]
        if (len(remaining) < self.min_compact_groups or
                (self.shards and estimated_tokens - self._last_compact_tokens < self.min_growth_tokens)):
            return removed, 0, 0
        # Archive a contiguous head prefix only; stop at any unknown/unsafe event.
        chosen = []
        for group in remaining:
            start, end, ids = group
            records = [self.store.for_call(cid) for cid in ids]
            if any(record is None for record in records):
                break
            chosen.append((start, end, ids, records))
        if not chosen:
            return removed, 0, 0
        event_start = self._covered_until + 1
        event_end = event_start + len(chosen) - 1
        mutations, observations = [], []
        for _, _, ids, records in chosen:
            for cid, record in zip(ids, records):
                observations.append({"obs_id": record.id, "tool_call_id": cid,
                                     "step": record.step, "tool": record.tool_name})
                requests = record.executed_requests or ()
                for req in requests:
                    if str(req.get("method", "")).upper() not in {"GET", "HEAD"}:
                        mutations.append({"api": req.get("api_name"), "status": req.get("status"),
                                          "evidence_ref": record.id, "method": req.get("method")})
                if record.error_type:
                    mutations.append({"api": ",".join(record.api_names), "status": "repaired_or_recorded_error",
                                      "evidence_ref": record.id, "error_type": record.error_type})
        shard_id = f"CP-{len(self.shards)+1:02d}"
        payload = {"shard_id": shard_id, "event_range": [event_start, event_end],
                   "mutations": mutations, "key_observations": observations}
        if semantic_memory:
            payload.update({key: semantic_memory.get(key, [])
                            for key in ("facts", "entities", "progress")})
        rendered = "Immutable checkpoint (execution evidence only):\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        shard = CheckpointShard(shard_id, (event_start, event_end), tuple(mutations), tuple(observations), rendered,
            tuple((semantic_memory or {}).get("facts", ())),
            tuple((semantic_memory or {}).get("entities", ())),
            tuple((semantic_memory or {}).get("unresolved", ())))
        self.shards.append(shard); self._covered_until = event_end
        self._last_compact_tokens = estimated_tokens
        self.archived_groups.update(tuple(ids) for _, _, ids, _ in chosen)
        self.boundary_anchor = f"[Context Boundary: Events 1-{event_end} archived into {shard_id}]"
        for start, end, _, _ in chosen:
            removed.update(range(start, end))
        return removed, 2, len(chosen)

    def consolidate_checkpoints(self, *, unresolved=(), entities=()) -> CheckpointShard | None:
        """Replace L1 with one compact immutable master checkpoint."""
        if not self.shards:
            return None
        mutations = tuple({json.dumps(row, sort_keys=True, ensure_ascii=False): row
                           for shard in self.shards for row in shard.mutations}.values())
        observations = tuple({row["obs_id"]: row for shard in self.shards
                              for row in shard.key_observations}.values())
        facts = tuple({json.dumps(row, sort_keys=True, ensure_ascii=False): row
                       for shard in self.shards for key in ("facts", "progress")
                       for row in getattr(shard, key, ())}.values())
        entity_source = list(entities) or [row for shard in self.shards for row in shard.entities]
        entity_rows = tuple({str(row.get("id")): row for row in entity_source if isinstance(row, dict)}.values())
        start, end = self.shards[0].event_range[0], self.shards[-1].event_range[1]
        payload = {"shard_id": "CP-MASTER-01", "event_range": [start, end],
                   "mutations": list(mutations), "key_observations": list(observations),
                   "entities": list(entity_rows), "facts": list(facts)}
        rendered = "Consolidated immutable checkpoint (execution evidence):\n" + json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"))
        unresolved_rows = list(unresolved) or [row for shard in self.shards for row in shard.unresolved]
        payload["unresolved"] = unresolved_rows
        # Re-render with the dynamic unresolved set included in this master snapshot.
        rendered = "Consolidated immutable checkpoint (execution evidence):\n" + json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"))
        master = CheckpointShard("CP-MASTER-01", (start, end), mutations, observations, rendered,
                                 facts, entity_rows, tuple(unresolved_rows))
        self.shards = [master]
        self.boundary_anchor = f"[Context Boundary: Events {start}-{end} consolidated into CP-MASTER-01]"
        self.level3_compactions = getattr(self, "level3_compactions", 0) + 1
        self.latest_unresolved = tuple(unresolved)
        return master


class TrajectoryCompactor:
    def __init__(self, store: ObservationStore, *, recent_groups: int = 4):
        if recent_groups < 1:
            raise ValueError("recent_groups must be positive")
        self.store = store
        self.recent_groups = recent_groups

    def compact(self, view: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        groups: list[tuple[int, int, list[dict[str, Any]]]] = []
        index = 0
        while index < len(view):
            message = view[index]
            calls = message.get("tool_calls", []) if message.get("role") == "assistant" else []
            if not calls:
                index += 1
                continue
            ids = {call.get("id") for call in calls}
            end = index + 1
            events = []
            while end < len(view) and view[end].get("role") == "tool" and view[end].get("tool_call_id") in ids:
                tool_message = view[end]
                record = self.store.for_call(tool_message["tool_call_id"])
                if record is None:
                    break
                try:
                    result = json.loads(record.content)
                except (ValueError, TypeError):
                    result = {}
                events.append({"observation_id": record.id, "tool_call_id": record.tool_call_id,
                               "tool": record.tool_name, "step": record.step,
                               "tool_ok": result.get("ok") if isinstance(result, dict) else None,
                               "api_names": list(record.api_names[:3]),
                               "error_type": record.error_type})
                end += 1
            if len(events) == len(ids) and {e["tool_call_id"] for e in events} == ids:
                groups.append((index, end, events))
                index = end
            else:
                index += 1
        older = groups[:-self.recent_groups]
        if not older:
            return view, {"compacted_groups": 0, "checkpoint_count": 0,
                          "compacted_observation_ids": []}
        start = older[0][0]
        end = older[-1][1]
        all_events = [event for _, _, group_events in older for event in group_events]
        checkpoint = {"kind": "compact_checkpoint", "step_from": min(e["step"] for e in all_events),
                      "step_to": max(e["step"] for e in all_events),
                      "events": all_events,
                      "recovery": "read_observation(observation_id, field='content' or 'arguments') pages the result or original call arguments",
                      "caution": "tool_ok records execution only, not business-goal success"}
        replacement = {"role": "user", "content": "Compacted prior tool interactions (event index, not a new observation):\n"
                       + json.dumps(checkpoint, ensure_ascii=False)}
        # Preserve non-group messages between complete groups. Typically there
        # are none; if present, keep them rather than silently dropping text.
        covered = {pos for group_start, group_end, _ in older for pos in range(group_start, group_end)}
        middle = [view[pos] for pos in range(start, end) if pos not in covered]
        compacted = [*view[:start], replacement, *middle, *view[end:]]
        return compacted, {"compacted_groups": len(older), "checkpoint_count": 1,
                           "compacted_observation_ids": [e["observation_id"] for e in all_events]}
