"""Build the task prompt for the Function Calling ReAct agent."""
from __future__ import annotations

from agentlab.context_engineering.unified import UnifiedContextConfig


class ContextManager:
    SYSTEM_PROMPT = """You are an interactive agent working in AppWorld. Choose exactly one registered tool per turn:
- `search_api(query?, queries?, app_name?, app?, top_k?)` searches or lists official API documentation for apps allowed in this task. Results include qualified API names, raw BM25 scores, and compact parameter hints.
- `get_api_schema(api_name)` returns the official schema for a task-allowed API. Read it before calling that API.
- `execute_python(code)` runs Python in the persistent AppWorld shell. Use only the simulated app APIs exposed by `apis`; never use host filesystem access, external packages, ground truth, evaluator code, or task data.

When a tool call returns an error, use that observation to correct the next action. Preserve useful shell variables across calls. For pagination, follow the documented pagination arguments until all relevant results have been fetched. Never guess API names, arguments, IDs, or account details. If authentication is required, inspect the app login schema, retrieve matching credentials from Supervisor APIs, log in, and retain/pass the returned token as documented.

When the requested work is verified, call apis.supervisor.complete_task(...) through execute_python. Pass the exact answer when one is requested. AppWorld confirms completion.

Read-only batching: combine same-phase exploration in one `execute_python` call, including documented login, complete pagination, detail reads, local filtering/aggregation, and a compact summary; do not split equivalent read queries across turns.
Write safety: read and resolve the full target set first. For batch writes, catch and record each item's outcome so one failure does not hide partial success; never blindly replay uncertain side effects."""

    def __init__(self, task_id: str, instruction: str, *,
                 schema_aware_retrieval: bool = False,
                 programmatic_execution: bool = False,
                 context_mode: str = "full",
                 unified_config: UnifiedContextConfig | None = None,
                 assurance_mode: str = "shadow",
                 app_catalog: str | None = None):
        prompt = self.SYSTEM_PROMPT
        if schema_aware_retrieval:
            prompt = prompt.replace(
                "Read it before calling that API.",
                "Search results may contain a compact official schema. Use it when sufficient; call `get_api_schema` when required fields, nested parameters, constraints, authentication, or return structure remain unclear.",
            )
        if programmatic_execution:
            prompt += (
                "\n\nProgrammatic execution guidance (use judgment; never guess API-specific arguments):\n"
                "- When the task and official schemas make a sequence of read-only queries clear, combine them in one `execute_python` call; follow each API's own documented pagination fields and stop only when its documented end condition is reached.\n"
                "- Filter, deduplicate, compare, and calculate returned records inside Python; print a compact decision-ready summary with exact IDs and values needed later.\n"
                "- Pass IDs between calls through the persistent AppWorld Python shell. Do not call the model between deterministic data-dependent operations.\n"
                "- For writes, first resolve all targets and required values. Record per-item outcomes; do not blindly retry or repeat a payment, deletion, message, or other side effect. Verify material writes when the task requires it.\n"
                "- For a batch, retain the expected target set and each item's source, destination, full date or other decision field, and result. Compare expected and observed sets after a readback; print missing and unexpected items. A successful outer Python call does not prove every item succeeded.\n"
                "- After a batch readback, you may print AGENTLAB_BATCH_CHECK= followed by JSON with expected, observed, and failed ID lists. Only use values obtained from documented AppWorld APIs; this receipt helps check set differences, and does not replace the official evaluator.\n"
                "- Stop and return control when information is missing, semantics are ambiguous, an error changes the plan, or a safe continuation cannot be established. Do not hide partial success or failures in a batch."
            )
        if assurance_mode == "reflection_v1":
            prompt += (
                "\n\nExecution completeness discipline:\n"
                "- For list APIs whose official schema includes `page_index`, loop from page 0 until `[]`; print `(pages_fetched, total_count)`.\n"
                "- Before batch writes, print the fetched count, filters, target IDs, and current destination state; deduplicate or clear old items only when the task requires it."
            )
        if context_mode in {"dynamic", "unified", "cache_first"} and (
                context_mode == "dynamic" or unified_config is None or unified_config.manual_retrieval):
            prompt += (
                "\n\nOlder tool results may be represented by observation IDs. "
                "Use search_history(query, top_k=5) to locate earlier results and "
                "read_observation(observation_id, offset=0, limit=1600) to recover the original content in pages. "
                "These tools never repeat business operations."
            )
        if context_mode == "cache_first":
            prompt += (
                "\n\nCache-first history notice: resolved no-write errors may be removed from the visible history, "
                "so missing step numbers are normal. Do not repeat an action only because its old turn is absent; "
                "use the retained evidence or recover it by Observation ID."
            )
        if context_mode == "unified" and unified_config and (
                unified_config.memory or unified_config.task_state or unified_config.compaction):
            prompt += (
                "\n\nTask State and Working Memory are task-local agent data, not proof of AppWorld success. "
                "Use registered tools to record evidence-linked facts or progress. Proposed facts are unverified. "
                "Compact checkpoints are event indexes; read cited Observations for full evidence."
            )
        if app_catalog:
            prompt += "\n\n" + app_catalog
        self.messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Task ID: {task_id}\nTask: {instruction}\nWrite the next Python action."},
        ]
