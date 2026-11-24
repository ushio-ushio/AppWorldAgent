from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _zero_usage() -> dict[str, float | int]:
    return {"input_tokens": 0, "input_cache_hit_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

def _add_usage(left: dict[str, float | int], right: dict[str, Any]) -> dict[str, float | int]:
    return {key: left.get(key, 0) + right.get(key, 0) for key in _zero_usage()}

def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "[REDACTED]" if str(key).lower() in {"api_key", "authorization", "access_token"} else _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "[TOKEN_REDACTED]", value)
        value = re.sub(
            r"(?i)((?:[\"']?\b(?:api[_-]?key|password|access[_-]?token|authorization|refresh[_-]?token)\b[\"']?)\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;})]+)",
            r'\1"[REDACTED]"', value,
        )
        value = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL_REDACTED]", value)
        value = re.sub(r"(?<!\d)(?:\+?\d[\d() .-]{7,}\d)(?!\d)", "[PHONE_REDACTED]", value)
    return value

class TraceLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_redact(event), ensure_ascii=False, default=str) + "\n")

    def write_summary(self, summary_path: Path, summary: dict[str, Any]) -> None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(_redact(summary), ensure_ascii=False, indent=2), encoding="utf-8")
