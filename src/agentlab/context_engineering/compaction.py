"""Deterministic old interaction checkpoints; original messages remain untouched."""
from __future__ import annotations

import json
from typing import Any

from agentlab.memory.store import ObservationStore


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
