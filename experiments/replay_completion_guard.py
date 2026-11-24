"""Replay CompletionGuard at frozen completion attempts; never run AppWorld.

Uses task instructions, public API method metadata, and saved trace events only.
It cannot prove individual business outcomes inside a Python batch.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

from agentlab.execution_assurance.completion_guard import CompletionGuard


ROOT = Path(__file__).resolve().parents[2]
TRACE_ROOT = ROOT / "runs" / "agentlab_v2_lite_fixed10_siwc_20261123_150216"
API_ROOT = ROOT / "vendor" / "appworld" / "data" / "api_docs" / "standard"
TASK_ROOT = ROOT / "vendor" / "appworld" / "data" / "tasks"
TASK_IDS = (
    "229360a_1", "3c13f5a_1", "771d8fc_1", "692c77d_1",
    "29caf6f_1", "302c169_1", "e3d6c94_2",
)
API_PATTERN = re.compile(r"\bapis\.([a-z][a-z_0-9]*)\.([a-z][a-z_0-9]*)\s*\(")


def api_methods() -> dict[str, str]:
    methods = {}
    for path in API_ROOT.glob("*.json"):
        docs = json.loads(path.read_text(encoding="utf-8"))
        for api_name, doc in docs.items():
            methods[f"{path.stem}.{api_name}"] = str(doc.get("method", "")).upper()
    return methods


def replay(task_id: str, methods: dict[str, str]) -> dict:
    instruction = json.loads((TASK_ROOT / task_id / "specs.json").read_text(encoding="utf-8"))["instruction"]
    rows = [json.loads(line) for line in (TRACE_ROOT / task_id / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    references = {(row["step"], row["source_tool"]): row.get("api_names", [])
                  for row in rows if row.get("event") == "observation_ref"}
    evidence = []
    guard = CompletionGuard(instruction)
    for row in rows:
        if row.get("event") != "tool_call" or row.get("tool_name") != "execute_python":
            continue
        try:
            code = json.loads(row["arguments"])["code"]
        except (KeyError, TypeError, ValueError):
            # Older sanitized traces may contain invalid JSON arguments. The
            # raw field can still establish API names, but not parameter values.
            code = str(row.get("arguments") or "")
        if "complete_task" in code:
            decision = guard.review(evidence, unresolved_error=False)
            return {
                "task_id": task_id, "completion_step": row["step"],
                "status": decision.status, "allow": decision.allow,
                "observed_write_apis": list(decision.observed),
                "known_api_evidence": len(evidence),
                "limitation": "Trace API names and outer tool success do not prove each operation's business outcome.",
            }
        names = set(references.get((row["step"], "execute_python"), []))
        seen = set()
        ordered = []
        for match in API_PATTERN.finditer(code):
            name = f"{match.group(1)}.{match.group(2)}"
            if name in names and name not in seen:
                ordered.append(name)
                seen.add(name)
        ordered.extend(sorted(names - seen))
        result = row.get("tool_result") or {}
        for name in ordered:
            evidence.append({
                "step": row["step"], "order": len(evidence), "api_name": name,
                "method": methods.get(name, "UNKNOWN"), "ok": result.get("ok") is True,
            })
    return {"task_id": task_id, "completion_step": None, "status": "NO_ATTEMPT", "allow": False}


if __name__ == "__main__":
    sys.stdout.write(json.dumps([replay(task_id, api_methods()) for task_id in TASK_IDS], indent=2))
    sys.stdout.write("\n")
