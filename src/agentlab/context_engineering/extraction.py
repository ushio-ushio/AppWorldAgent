"""Best-effort asynchronous semantic extraction for cache_first contexts."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
from typing import Any


class AsyncContextExtractor:
    def __init__(self, model: Any):
        self.model = model
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agentlab-extract")
        self.future: Future | None = None
        self.pending: dict[str, Any] | None = None
        self.usage = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}
        self.triggers = self.hits = self.fallbacks = 0
        self.closed = False
        self.last_submit_groups = 0
        self.last_submit_tokens = 0

    @staticmethod
    def _valid(value: Any, valid_refs: set[str]) -> list[dict[str, Any]]:
        if not isinstance(value, list): return []
        return [item for item in value if isinstance(item, dict)
                and isinstance(item.get("evidence_ref"), str)
                and item["evidence_ref"] in valid_refs]

    def poll(self, valid_refs: set[str]) -> dict[str, Any] | None:
        if not self.future or not self.future.done(): return self.pending
        future, self.future = self.future, None
        try:
            result, usage = future.result()
            if not isinstance(result, dict): raise ValueError("extraction must be an object")
            result = {key: self._valid(result.get(key), valid_refs)
                      for key in ("facts", "entities", "progress", "unresolved")}
            result["covered_step_range"] = result.get("covered_step_range", [])
            self.pending = result
            self.usage = {k: self.usage.get(k, 0) + int(usage.get(k, 0)) for k in self.usage}
            self.hits += 1
        except Exception:
            self.fallbacks += 1
        return self.pending

    def submit(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
               *, total_groups: int = 0, estimated_tokens: int = 0,
               min_groups: int = 4, min_tokens: int = 3500, soft_tokens: int = 0) -> bool:
        if self.closed or self.future is not None: return False
        if estimated_tokens < soft_tokens: return False
        if self.triggers and (total_groups - self.last_submit_groups < min_groups or
                              estimated_tokens - self.last_submit_tokens < min_tokens): return False
        prefix = messages[:-1] if messages and "VOLATILE RUNTIME TAIL" in str(messages[-1].get("content", "")) else messages
        frozen = json.loads(json.dumps(prefix, ensure_ascii=False, default=str))
        instruction = {"role": "user", "content": (
            "Extract JSON only: covered_step_range, facts, entities, progress, unresolved. "
            "Every item must cite evidence_ref exactly as an observation_id or step_k. "
            "Do not infer verification or perform actions.")}
        self.future = self._pool.submit(self.model.extract_context, [*frozen, instruction], tools or [])
        self.triggers += 1
        self.last_submit_groups, self.last_submit_tokens = total_groups, estimated_tokens
        return True

    def cancel(self) -> None:
        self.closed = True
        if self.future:
            self.future.cancel()
            self.future = None
        self._pool.shutdown(wait=False, cancel_futures=True)

    def record_fallback(self) -> None:
        self.fallbacks += 1

    def summary(self) -> dict[str, Any]:
        return {"triggers": self.triggers, "hits": self.hits, "fallbacks": self.fallbacks,
                **self.usage}
