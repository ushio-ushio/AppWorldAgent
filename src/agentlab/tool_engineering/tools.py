"""Task API discovery and validated Function Calling dispatch."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import re
from typing import Any, Callable, Iterable, Mapping


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def compact_api_schema(api_name: str, official_doc: Mapping[str, Any], *,
                       max_chars: int = 1800) -> dict[str, Any]:
    """Build a bounded, loss-conscious view of fields from an official AppWorld doc.

    If the public parameter description cannot be represented without dropping
    structure or exceeding the budget, callers must fetch the full schema.
    """
    if max_chars < 128:
        raise ValueError("max_chars must be at least 128")
    parameters = official_doc.get("parameters")
    if not isinstance(parameters, list):
        return {"api_name": api_name, "schema_status": "use_get_api_schema",
                "schema_lookup": api_name}

    compact_parameters = []
    for parameter in parameters:
        if not isinstance(parameter, Mapping) or not isinstance(parameter.get("name"), str):
            return {"api_name": api_name, "schema_status": "use_get_api_schema",
                    "schema_lookup": api_name}
        if (not isinstance(parameter.get("type"), str)
                or not parameter.get("type", "").strip()
                or type(parameter.get("required")) is not bool
                or set(parameter) - {"name", "type", "required", "description", "default", "constraints"}
                or any(token in parameter.get("type", "").lower()
                       for token in ("dict", "object", "mapping"))):
            return {"api_name": api_name, "schema_status": "use_get_api_schema",
                    "schema_lookup": api_name}
        row = {
            "name": parameter["name"],
            "type": parameter["type"],
            "required": parameter["required"],
        }
        # Keep every official constraint and the parameter description. These
        # often carry enum, format, pagination, or nested-object requirements.
        for key in ("constraints", "description"):
            if parameter.get(key) not in (None, "", []):
                row[key] = parameter[key]
        if "default" in parameter and parameter["default"] is not None:
            row["default"] = parameter["default"]
        compact_parameters.append(row)

    summary = {
        "api_name": api_name,
        "method": official_doc.get("method"),
        "description": official_doc.get("description", ""),
        "parameters": compact_parameters,
        "schema_status": "compact_official_schema",
        "full_schema_tool": "get_api_schema",
        "schema_lookup": api_name,
    }
    summary = {key: value for key, value in summary.items() if value not in (None, "")}
    if len(json.dumps(summary, ensure_ascii=False, default=str)) > max_chars:
        return {"api_name": api_name, "schema_status": "use_get_api_schema",
                "schema_lookup": api_name}
    return summary


def _tokens(text: str) -> list[str]:
    # Split snake_case and camelCase names in addition to ordinary prose.
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return _TOKEN_RE.findall(text.lower().replace("_", " ").replace(".", " "))


@dataclass(frozen=True)
class ApiDocument:
    qualified_name: str
    app_name: str
    api_name: str
    description: str
    official_doc: Mapping[str, Any]
    terms: tuple[str, ...]


class ApiRetriever:
    """Small in-process BM25 index built only from ``task.api_docs``.

    The task's AppWorld ``ApiDocCollection`` contains the official docs for its
    allowed apps and excludes private APIs. No task answer or ground truth is
    consulted here.
    """

    def __init__(self, api_docs: Mapping[str, Mapping[str, Mapping[str, Any]]],
                 allowed_apps: Iterable[str], *, k1: float = 1.5, b: float = 0.75,
                 schema_aware: bool = False, schema_top_k: int = 2,
                 schema_max_chars: int = 1800):
        if schema_top_k < 0 or schema_top_k > 2:
            raise ValueError("schema_top_k must be between 0 and 2")
        self.k1 = k1
        self.b = b
        self.schema_aware = schema_aware
        self.schema_top_k = schema_top_k
        self.schema_max_chars = schema_max_chars
        allowed = set(allowed_apps)
        documents: list[ApiDocument] = []
        for app_name, api_map in api_docs.items():
            if app_name not in allowed:
                continue
            for api_name, doc in api_map.items():
                description = str(doc.get("description", ""))
                name = f"{app_name}.{api_name}"
                text = " ".join((name, app_name, api_name, description))
                documents.append(ApiDocument(
                    qualified_name=name,
                    app_name=app_name,
                    api_name=api_name,
                    description=description,
                    official_doc=doc,
                    terms=tuple(_tokens(text)),
                ))
        self.documents = tuple(documents)
        self._lengths = tuple(len(doc.terms) for doc in self.documents)
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        self._document_frequency: Counter[str] = Counter()
        for doc in self.documents:
            self._document_frequency.update(set(doc.terms))

    def search_api(self, query: str, app: str | None = None, top_k: int = 5) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            return []
        if app is not None and (not isinstance(app, str) or not app.strip()):
            return []
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            return []
        if self.schema_aware:
            top_k = min(top_k, 5)
        query_terms = _tokens(query)
        if not query_terms:
            return []
        app_filter = app.strip().lower() if app is not None else None
        candidates: list[tuple[float, ApiDocument]] = []
        n_docs = len(self.documents)
        for index, document in enumerate(self.documents):
            if app_filter is not None and document.app_name.lower() != app_filter:
                continue
            term_counts = Counter(document.terms)
            length = self._lengths[index]
            score = 0.0
            for term in query_terms:
                frequency = term_counts.get(term, 0)
                if not frequency:
                    continue
                df = self._document_frequency[term]
                # Positive-IDF Okapi BM25; the returned value is the raw
                # relevance score and is not normalized into a probability.
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                norm = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / max(self._average_length, 1e-12)
                )
                score += idf * frequency * (self.k1 + 1.0) / norm
            if score > 0.0:
                candidates.append((score, document))
        candidates.sort(key=lambda item: (-item[0], item[1].qualified_name))
        results = []
        for rank, (score, document) in enumerate(candidates[:top_k], start=1):
            result = {
                "api_name": document.qualified_name,
                "description": document.description,
                "score": score,
            }
            if self.schema_aware:
                result["rank"] = rank
                if rank <= self.schema_top_k:
                    result["schema"] = compact_api_schema(
                        document.qualified_name, document.official_doc,
                        max_chars=self.schema_max_chars)
            results.append(result)
        return results

    def get_api_schema(self, api_name: str) -> dict[str, Any] | None:
        if not isinstance(api_name, str) or not api_name.strip():
            return None
        normalized = api_name.strip().lower()
        matches = [
            doc for doc in self.documents
            if doc.qualified_name.lower() == normalized
        ]
        if not matches and "." not in normalized:
            matches = [doc for doc in self.documents if doc.api_name.lower() == normalized]
        # An unqualified duplicate across apps is ambiguous; require the
        # qualified app.api_name form instead of guessing.
        if len(matches) != 1:
            return None
        return dict(matches[0].official_doc)


class ToolRegistry:
    def __init__(self, *, search_api: Callable[..., Any], get_api_schema: Callable[..., Any],
                 execute_python: Callable[..., Any], api_retrieval_enabled: bool = True,
                 schema_aware_retrieval: bool = False,
                 search_history: Callable[..., Any] | None = None,
                 read_observation: Callable[..., Any] | None = None,
                 remember_fact: Callable[..., Any] | None = None,
                 list_memory: Callable[..., Any] | None = None,
                 invalidate_memory: Callable[..., Any] | None = None,
                 update_task_state: Callable[..., Any] | None = None):
        self._handlers = {
            "search_api": search_api,
            "get_api_schema": get_api_schema,
            "execute_python": execute_python,
        }
        self.schema_aware_retrieval = schema_aware_retrieval
        self.tool_names = frozenset(self._handlers)

        self.definitions = [
            self._definition("search_api", (
                "BM25 search over official API docs allowed for this task. "
                + ("Top candidates may include a compact official parameter schema; use get_api_schema(api_name) for the complete schema when fields are omitted or constraints are complex."
                   if schema_aware_retrieval else "Results include API name, description, and raw relevance score.")
            ), {
                "query": {"type": "string", "minLength": 1},
                "app": {"type": ["string", "null"]},
                "top_k": ({"type": "integer", "minimum": 1, "maximum": 5}
                          if schema_aware_retrieval else {"type": "integer", "minimum": 1}),
            }, ["query"]),
            self._definition("get_api_schema", "Return the official full schema for one task-allowed AppWorld API.", {
                "api_name": {"type": "string", "minLength": 1},
            }, ["api_name"]),
            self._definition("execute_python", "Execute Python code in the persistent AppWorld task shell.", {
                "code": {"type": "string", "minLength": 1},
            }, ["code"]),
        ]
        if not api_retrieval_enabled:
            self.definitions = [self.definitions[-1]]
        if (search_history is None) != (read_observation is None):
            raise ValueError("History retrieval callbacks must be supplied together")
        if search_history is not None and read_observation is not None:
            self._handlers.update(search_history=search_history,
                                  read_observation=read_observation)
            self.definitions.extend([
                self._definition("search_history", "Search earlier task tool results by API name, observation ID, or keywords. Returns sources, not full content.", {
                    "query": {"type": "string", "minLength": 1},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                }, ["query"]),
                self._definition("read_observation", "Read a saved tool result or original call arguments by observation ID in pages; never re-executes the action.", {
                    "observation_id": {"type": "string", "minLength": 1},
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 4096},
                    "field": {"type": "string", "enum": ["content", "arguments"]},
                }, ["observation_id"]),
            ])
        if any(callback is not None for callback in (remember_fact, list_memory, invalidate_memory)):
            if not all(callback is not None for callback in (remember_fact, list_memory, invalidate_memory)):
                raise ValueError("Working Memory callbacks must be supplied together")
            self._handlers.update(remember_fact=remember_fact, list_memory=list_memory,
                                  invalidate_memory=invalidate_memory)
            self.definitions.extend([
                self._definition("remember_fact", "Propose a task-local fact with an existing Observation ID; this does not verify it.", {
                    "namespace": {"type": "string"}, "entity": {"type": "string"},
                    "key": {"type": "string"}, "value": {"type": ["string", "number", "boolean", "null"]},
                    "observation_id": {"type": "string"}},
                    ["namespace", "entity", "key", "value", "observation_id"]),
                self._definition("list_memory", "List current task-local facts with status and evidence references.", {
                    "namespace": {"type": ["string", "null"]}, "entity": {"type": ["string", "null"]},
                    "include_invalidated": {"type": "boolean"}}, []),
                self._definition("invalidate_memory", "Invalidate a memory only with a task-local Observation ID and a reason; observed facts require contrary structured evidence.", {
                    "memory_id": {"type": "string"}, "observation_id": {"type": "string"},
                    "reason": {"type": "string"}}, ["memory_id", "observation_id", "reason"]),
            ])
        if update_task_state is not None:
            self._handlers["update_task_state"] = update_task_state
            self.definitions.append(self._definition("update_task_state", "Record focus, a subtask claim, or an unresolved issue; this does not complete AppWorld.", {
                "action": {"type": "string", "enum": ["focus", "subtask", "issue"]},
                "subtask_id": {"type": "string"}, "description": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "done", "failed"]},
                "observation_id": {"type": "string"}, "focus": {"type": "string"},
                "note": {"type": "string"}}, ["action"]))
        self.tool_names = frozenset(self._handlers)

    @staticmethod
    def _definition(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object", "properties": properties,
                "required": required, "additionalProperties": False,
            },
        }}

    def _validate(self, name: str, arguments: Any) -> tuple[dict[str, Any] | None, str | None]:
        if not isinstance(arguments, dict):
            return None, "Arguments must be a JSON object."
        schemas = {
            "search_api": {"required": {"query"}, "allowed": {"query", "app", "top_k"}},
            "get_api_schema": {"required": {"api_name"}, "allowed": {"api_name"}},
            "execute_python": {"required": {"code"}, "allowed": {"code"}},
            "search_history": {"required": {"query"}, "allowed": {"query", "top_k"}},
            "read_observation": {"required": {"observation_id"}, "allowed": {"observation_id", "offset", "limit", "field"}},
            "remember_fact": {"required": {"namespace", "entity", "key", "value", "observation_id"},
                              "allowed": {"namespace", "entity", "key", "value", "observation_id"}},
            "list_memory": {"required": set(), "allowed": {"namespace", "entity", "include_invalidated"}},
            "invalidate_memory": {"required": {"memory_id", "observation_id", "reason"},
                                  "allowed": {"memory_id", "observation_id", "reason"}},
            "update_task_state": {"required": {"action"},
                                  "allowed": {"action", "subtask_id", "description", "status", "observation_id", "focus", "note"}},
        }
        schema = schemas[name]
        missing = schema["required"] - arguments.keys()
        unknown = arguments.keys() - schema["allowed"]
        if missing:
            return None, f"Missing required argument(s): {', '.join(sorted(missing))}."
        if unknown:
            return None, f"Unknown argument(s): {', '.join(sorted(unknown))}."
        if name in {"search_api", "search_history"}:
            if not isinstance(arguments["query"], str) or not arguments["query"].strip():
                return None, "query must be a non-empty string."
            if name == "search_api" and "app" in arguments and arguments["app"] is not None and not isinstance(arguments["app"], str):
                return None, "app must be a string or null."
            if "top_k" in arguments and (isinstance(arguments["top_k"], bool) or not isinstance(arguments["top_k"], int) or arguments["top_k"] < 1 or (name == "search_history" and arguments["top_k"] > 10)):
                return None, "top_k is outside the allowed range."
            if name == "search_api" and self.schema_aware_retrieval and arguments.get("top_k", 5) > 5:
                return None, "top_k cannot exceed 5 when compact schemas are included."
        elif name == "read_observation":
            if not isinstance(arguments["observation_id"], str) or not arguments["observation_id"].strip():
                return None, "observation_id must be a non-empty string."
            for key, low, high in (("offset", 0, 2**31 - 1), ("limit", 1, 4096)):
                if key in arguments and (isinstance(arguments[key], bool) or not isinstance(arguments[key], int) or not low <= arguments[key] <= high):
                    return None, f"{key} is outside the allowed range."
            if "field" in arguments and arguments["field"] not in {"content", "arguments"}:
                return None, "field must be content or arguments."
        elif name in {"get_api_schema", "execute_python"}:
            key = "api_name" if name == "get_api_schema" else "code"
            if not isinstance(arguments[key], str) or not arguments[key].strip():
                return None, f"{key} must be a non-empty string."
        elif name == "remember_fact":
            if any(not isinstance(arguments[key], str) or not arguments[key].strip()
                   for key in ("namespace", "entity", "key", "observation_id")):
                return None, "Memory identifiers must be non-empty strings."
            if not (arguments["value"] is None or type(arguments["value"]) in (str, int, float, bool)):
                return None, "value must be a JSON scalar."
        elif name == "invalidate_memory":
            if any(not isinstance(arguments[key], str) or not arguments[key].strip()
                   for key in ("memory_id", "observation_id", "reason")):
                return None, "Invalidation requires three non-empty strings."
        elif name == "list_memory":
            if any(key in arguments and arguments[key] is not None and not isinstance(arguments[key], str)
                   for key in ("namespace", "entity")):
                return None, "Memory filters must be strings or null."
            if "include_invalidated" in arguments and not isinstance(arguments["include_invalidated"], bool):
                return None, "include_invalidated must be boolean."
        elif name == "update_task_state":
            if arguments["action"] not in {"focus", "subtask", "issue"}:
                return None, "Unknown Task State action."
            if any(not isinstance(value, str) for value in arguments.values()):
                return None, "Task State fields must be strings."
        return arguments, None

    def dispatch(self, *, name: Any, raw_arguments: Any, tool_call_id: str | None) -> dict[str, Any]:
        if not isinstance(name, str) or name not in self._handlers:
            return {"tool_call_id": tool_call_id, "name": name, "ok": False,
                    "validation": "rejected", "error": "Unknown tool name; choose one of the registered tools."}
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        except (json.JSONDecodeError, TypeError) as exc:
            return {"tool_call_id": tool_call_id, "name": name, "ok": False,
                    "validation": "rejected", "error": f"Arguments are not valid JSON: {exc.msg if isinstance(exc, json.JSONDecodeError) else type(exc).__name__}."}
        validated, error = self._validate(name, arguments)
        if error:
            return {"tool_call_id": tool_call_id, "name": name, "ok": False,
                    "validation": "rejected", "error": error}
        try:
            value = self._handlers[name](**validated)
            tool_error = None
            if name == "execute_python" and isinstance(value, tuple) and len(value) == 2:
                value, tool_error = value
            if name == "get_api_schema" and value is None:
                tool_error = "No unique matching API schema found. Use a task-allowed qualified name such as app.api_name."
            return {"tool_call_id": tool_call_id, "name": name,
                    "ok": tool_error is None, "validation": "accepted",
                    "result": value, **({"execution_error": tool_error} if tool_error else {})}
        except Exception as exc:
            return {"tool_call_id": tool_call_id, "name": name, "ok": False,
                    "validation": "accepted", "error": f"Tool execution failed ({type(exc).__name__}): {exc}"}
