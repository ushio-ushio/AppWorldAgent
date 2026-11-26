"""Build full or compact message views while preserving tool-call pairs."""
from __future__ import annotations

import json
from typing import Any

from memory.store import ObservationStore


class DynamicContextBuilder:
    """Create a smaller copy of messages, retaining valid tool-call/result pairs."""

    def __init__(self, store: ObservationStore, *, strategy: str = "dynamic",
                 recent_tool_results: int = 2, large_python_bytes: int = 2048):
        if strategy not in {"full", "dynamic"}:
            raise ValueError("strategy must be full or dynamic")
        self.store = store
        self.strategy = strategy
        self.recent_tool_results = recent_tool_results
        self.large_python_bytes = large_python_bytes

    def build(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if self.strategy == "full":
            return messages, {"offloaded_ids": [], "visible_bytes": len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))}
        tool_positions = [i for i, message in enumerate(messages) if message.get("role") == "tool"]
        recent = set(tool_positions[-self.recent_tool_results:])
        view = []
        offloaded = []
        seen_errors: set[tuple[str, str]] = set()
        for index, message in enumerate(messages):
            if message.get("role") != "tool":
                view.append(message.copy())
                continue
            record = self.store.for_call(message.get("tool_call_id", ""))
            if record is None:
                view.append(message.copy())
                continue
            repeated_error = bool(record.error_type and
                                  (record.tool_name, record.error_type) in seen_errors)
            if record.error_type:
                seen_errors.add((record.tool_name, record.error_type))
            compact = index not in recent and (
                record.tool_name in {"search_api", "get_api_schema", "search_history"}
                or (self.strategy == "dynamic" and record.tool_name in {"execute_python", "read_observation"}
                    and record.size_bytes > self.large_python_bytes)
                or (self.strategy == "dynamic" and repeated_error)
            )
            if compact:
                pointer = {"observation_id": record.id, "source_tool": record.tool_name,
                           "step": record.step, "size_bytes": record.size_bytes,
                           "api_names": list(record.api_names), "error_type": record.error_type,
                           "recovery": "Use search_history or read_observation to recover the full result."}
                view.append({**message, "content": json.dumps(pointer, ensure_ascii=False)})
                offloaded.append(record.id)
            else:
                view.append(message.copy())
        stats = {"offloaded_ids": offloaded}
        stats["visible_bytes"] = len(json.dumps(view, ensure_ascii=False).encode("utf-8"))
        return view, stats
