"""Task API discovery and validated Function Calling dispatch."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import re
from typing import Any, Callable, Iterable, Mapping


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_QUERY_SYNONYM_GROUPS = (
    frozenset("delete remove close erase".split()),
    frozenset("pay payment venmo transfer".split()),
    frozenset("note memo".split()),
    frozenset("song songs track tracks music playlist playlists".split()),
    frozenset("backup export file files".split()),
    frozenset("contact contacts friend friends coworker coworkers".split()),
)
_CATALOG_STOPWORDS = frozenset((
    "api", "get", "list", "search", "return", "returns", "create", "update",
    "delete", "show", "retrieve", "user", "users", "data", "record", "records",
    "information", "allows", "provide", "provides", "with", "from", "that",
    "this", "into", "your", "their", "and", "the", "for", "all", "using",
))


def app_capability_catalog(retriever: "ApiRetriever", *, max_apps: int = 10) -> str:
    """Render a deterministic, task-local app catalog from allowed official docs."""
    by_app: dict[str, list[ApiDocument]] = {}
    for document in retriever.documents:
        by_app.setdefault(document.app_name, []).append(document)
    lines = []
    for app_name in sorted(by_app)[:max_apps]:
        documents = by_app[app_name]
        excluded = _CATALOG_STOPWORDS | set(_tokens(app_name))
        weights: Counter[str] = Counter()
        for document in documents:
            for token in _tokens(document.api_name):
                if len(token) > 2 and token not in excluded:
                    weights[token] += 3
            for token in _tokens(document.description):
                if len(token) > 2 and token not in excluded:
                    weights[token] += 1
        topics = [token for token, _ in sorted(weights.items(), key=lambda item: (-item[1], item[0]))[:4]]
        summary = ("APIs primarily cover " + ", ".join(topics) if topics
                   else f"provides {len(documents)} task-allowed APIs")
        lines.append(f"- {app_name}: {summary}.")
    if len(by_app) > max_apps:
        lines.append(f"- Additional task-allowed apps: {len(by_app) - max_apps}.")
    if not lines:
        return ""
    return "Task-allowed AppWorld app capabilities (static task catalog):\n" + "\n".join(lines)


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


def _paginated(doc: Mapping[str, Any]) -> bool:
    return any(isinstance(p, Mapping) and p.get("name") in {"page_index", "page_limit"}
               for p in doc.get("parameters", ()))


_KEY_OPTIONAL_PARTS = ("page", "query", "search", "filter", "date", "start", "end",
                       "since", "before", "after", "status", "limit", "offset", "sort")


def _parameter_hint(doc: Mapping[str, Any]) -> str:
    params = doc.get("parameters")
    if not isinstance(params, list):
        return ""
    required, optional = [], []
    for param in params:
        if not isinstance(param, Mapping) or not isinstance(param.get("name"), str):
            continue
        name = param["name"]
        value = f"{name}:{param.get('type', '?')}"
        if param.get("required") is True:
            required.append(value)
        elif any(part in name.lower() for part in _KEY_OPTIONAL_PARTS):
            optional.append(value)
    pieces = []
    if required:
        pieces.append("required " + ",".join(required))
    if optional:
        pieces.append("optional " + ",".join(optional))
    return "; ".join(pieces)


def _search_schema(doc: Mapping[str, Any]) -> dict[str, Any]:
    parameters = doc.get("parameters")
    if not isinstance(parameters, list):
        return {"schema_status": "use_get_api_schema"}
    compact = []
    for p in parameters:
        if (not isinstance(p, Mapping) or not isinstance(p.get("name"), str)
                or not isinstance(p.get("type"), str) or type(p.get("required")) is not bool
                or any(k not in {"name", "type", "required", "description", "default", "constraints"} for k in p)):
            return {"schema_status": "use_get_api_schema"}
        label = p["name"] + ("*" if p["required"] else "?")
        row = {"param": label, "type": p["type"]}
        for key in ("constraints", "default"):
            if p.get(key) not in (None, "", []):
                row[key] = p[key]
        compact.append(row)
    return {"schema_status": "compact_official_schema", "method": doc.get("method"),
            "parameters": compact}


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

    def search_api(self, query: str | None = None, app: str | None = None, top_k: int = 5,
                   *, queries: Iterable[str] | None = None,
                   app_name: str | None = None) -> list[dict[str, Any]] | dict[str, Any]:
        query_list = []
        if query is not None:
            if not isinstance(query, str) or not query.strip():
                return []
            query_list.append(query)
        if queries is not None:
            if isinstance(queries, str) or not isinstance(queries, Iterable):
                return []
            batch = list(queries)
            if any(not isinstance(item, str) or not item.strip() for item in batch):
                return []
            query_list.extend(batch)
        if not query_list and app_name is not None:
            return self.list_app_apis(app_name)
        if not query_list:
            return []
        if app is not None and (not isinstance(app, str) or not app.strip()):
            return []
        if app_name is not None and (not isinstance(app_name, str) or not app_name.strip()):
            return []
        if app is not None and app_name is not None and app.strip().lower() != app_name.strip().lower():
            return []
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            return []
        if self.schema_aware:
            top_k = min(top_k, 5)
        query_terms = list(dict.fromkeys(token for item in query_list for token in _tokens(item)))
        if not query_terms:
            return []
        query_set = set(query_terms)
        expanded_terms = []
        for group in _QUERY_SYNONYM_GROUPS:
            if query_set & group:
                expanded_terms.extend(sorted(group - query_set))
                query_set.update(group)
        query_terms.extend(expanded_terms)
        app_filter = (app_name or app).strip().lower() if (app_name or app) is not None else None
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
            parameter_hint = _parameter_hint(document.official_doc)
            if parameter_hint:
                result["parameter_hint"] = parameter_hint
            if self.schema_aware:
                result["rank"] = rank
                if _paginated(document.official_doc):
                    result["paginated"] = True
                if rank <= self.schema_top_k:
                    result["schema"] = _search_schema(document.official_doc)
            results.append(result)
        if self.schema_aware:
            size = lambda: len(json.dumps(results, ensure_ascii=False, separators=(",", ":"), default=str))
            for row in reversed(results):
                if size() < 1800:
                    break
                if "schema" in row:
                    row["schema"] = {"schema_status": "use_get_api_schema"}
            for row in reversed(results):
                if size() < 1800:
                    break
                row["description"] = row["description"][:60]
        return results

    def list_app_apis(self, app_name: str, *, limit: int = 12) -> dict[str, Any]:
        """Return a compact, deterministic index of one task-allowed app's APIs."""
        if not isinstance(app_name, str) or not app_name.strip() or limit < 1:
            return {"app_name": app_name, "apis": [], "total_apis": 0, "truncated": False}
        matches = sorted((doc for doc in self.documents
                          if doc.app_name.lower() == app_name.strip().lower()),
                         key=lambda doc: doc.qualified_name)
        rows = [{"api_name": doc.qualified_name,
                 "method": doc.official_doc.get("method"),
                 "parameter_hint": _parameter_hint(doc.official_doc),
                 "description": doc.description[:100]}
                for doc in matches[:limit]]
        return {"app_name": app_name.strip(), "total_apis": len(matches),
                "truncated": len(matches) > limit, "apis": rows}

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
        result = dict(matches[0].official_doc)
        if _paginated(result):
            result["paginated"] = True
        return result


class ToolRegistry:
    def __init__(self, *, search_api: Callable[..., Any], get_api_schema: Callable[..., Any],
                 execute_python: Callable[..., Any], execution_summary_provider: Callable[[], Any] | None = None,
                 api_retrieval_enabled: bool = True,
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
        self.execution_summary_provider = execution_summary_provider
        self.seen_schemas: set[str] = set()
        self.tool_names = frozenset(self._handlers)

        self.definitions = [
            self._definition("search_api", (
                "Search official task-allowed APIs by query, several queries, or app_name alone for a compact app API index. "
                + ("Top candidates may include a compact official parameter schema; use get_api_schema(api_name) for the complete schema when fields are omitted or constraints are complex."
                   if schema_aware_retrieval else "Results include API name, description, and raw relevance score.")
            ), {
                "query": {"type": "string", "minLength": 1},
                "queries": {"type": "array", "items": {"type": "string", "minLength": 1},
                            "minItems": 1},
                "app_name": {"type": "string", "minLength": 1},
                "app": {"type": ["string", "null"]},
                "top_k": ({"type": "integer", "minimum": 1, "maximum": 5}
                          if schema_aware_retrieval else {"type": "integer", "minimum": 1}),
            }, []),
            self._definition("get_api_schema", "Return the official full schema for one task-allowed AppWorld API.", {
                "api_name": {"type": "string", "minLength": 1},
            }, ["api_name"]),
            self._definition("execute_python", "Execute Python in the persistent AppWorld shell. After writes and readback, submit alone with apis.supervisor.complete_task(answer=...); use a plain value for answers and answer=None for action-only tasks.", {
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
            "search_api": {"required": set(), "allowed": {"query", "queries", "app_name", "app", "top_k"}},
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
        if name == "search_api":
            if not any(key in arguments for key in ("query", "queries", "app_name")):
                return None, "Provide query, queries, or app_name."
            if "query" in arguments and (not isinstance(arguments["query"], str) or not arguments["query"].strip()):
                return None, "query must be a non-empty string."
            if "queries" in arguments and (not isinstance(arguments["queries"], list) or not arguments["queries"]
                    or any(not isinstance(item, str) or not item.strip() for item in arguments["queries"])):
                return None, "queries must be a non-empty list of non-empty strings."
            for key in ("app", "app_name"):
                if key in arguments and arguments[key] is not None and (
                        not isinstance(arguments[key], str) or not arguments[key].strip()):
                    return None, f"{key} must be a non-empty string or null."
            if ("app" in arguments and "app_name" in arguments and arguments["app"] is not None
                    and arguments["app"].lower() != arguments["app_name"].lower()):
                return None, "app and app_name must identify the same app."
            if "top_k" in arguments and (isinstance(arguments["top_k"], bool)
                    or not isinstance(arguments["top_k"], int) or arguments["top_k"] < 1
                    or (self.schema_aware_retrieval and arguments["top_k"] > 5)):
                return None, "top_k is outside the allowed range."
        elif name == "search_history":
            if not isinstance(arguments["query"], str) or not arguments["query"].strip():
                return None, "query must be a non-empty string."
            if "top_k" in arguments and (isinstance(arguments["top_k"], bool) or not isinstance(arguments["top_k"], int) or arguments["top_k"] < 1 or arguments["top_k"] > 10):
                return None, "top_k is outside the allowed range."
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
            elif name == "get_api_schema" and isinstance(value, Mapping):
                canonical = self._schema_identity(arguments["api_name"], value)
                if canonical in self.seen_schemas:
                    required = [
                        f"{parameter['name']}: {parameter.get('type', 'unknown')}*"
                        for parameter in value.get("parameters", ())
                        if isinstance(parameter, Mapping) and parameter.get("required") is True
                        and isinstance(parameter.get("name"), str)
                    ]
                    value = {"api_name": canonical,
                             "required_signature": ", ".join(required) or "()",
                             "already_fetched_in_session": True,
                             **({"paginated": True} if value.get("paginated") else {})}
                else:
                    self.seen_schemas.add(canonical)
            summary = self.execution_summary_provider() if name == "execute_python" and self.execution_summary_provider else None
            envelope = self._error_envelope(str(tool_error or ""), summary) if name == "execute_python" and tool_error else None
            return {"tool_call_id": tool_call_id, "name": name,
                    "ok": tool_error is None, "validation": "accepted",
                    "result": value, **({"execution_error": tool_error} if tool_error else {}),
                    **({"execution_summary": summary} if summary else {}),
                    **({"error_envelope": envelope} if envelope else {})}
        except Exception as exc:
            message = f"Tool execution failed ({type(exc).__name__}): {exc}"
            summary = self.execution_summary_provider() if name == "execute_python" and self.execution_summary_provider else None
            envelope = self._error_envelope(message, summary) if name == "execute_python" else None
            return {"tool_call_id": tool_call_id, "name": name, "ok": False,
                    "validation": "accepted", "error": message,
                    **({"execution_summary": summary} if summary else {}),
                    **({"error_envelope": envelope} if envelope else {})}

    def _error_envelope(self, error: str, summary: Any) -> dict[str, str] | None:
        failure = (summary or {}).get("first_failure") or {}
        endpoint = failure.get("endpoint")
        status = failure.get("status_code")
        if status == 422 or "Response status code is 422" in error:
            required = []
            if endpoint:
                doc = self._handlers["get_api_schema"](api_name=endpoint)
                if isinstance(doc, Mapping):
                    required = [p["name"] for p in doc.get("parameters", [])
                                if isinstance(p, Mapping) and p.get("required") is True and isinstance(p.get("name"), str)]
            suffix = f" Required official parameters: {', '.join(required)}." if required else " Check the full official schema."
            return {"error_type": "http_422_validation", "recovery_hint": f"Correct {endpoint or 'the failed API'} arguments.{suffix}"}
        if status == 401 or "Response status code is 401" in error or "Invalid credentials" in error:
            app_name = str(endpoint or "").split(".", 1)[0].lower()
            consecutive = (summary or {}).get("consecutive_401", {})
            if app_name and isinstance(consecutive, Mapping) and consecutive.get(app_name, 0) >= 2:
                return {"error_type": "http_401_auth", "recovery_hint": (
                    f"Authentication for {app_name} failed twice consecutively. Next, call "
                    "apis.supervisor.show_account_passwords(), verify the exact credentials for this app, "
                    f"call {app_name}.login, and replace the persistent access_token variable with the returned token before retrying. Do not repeat the failed request first.")}
            return {"error_type": "http_401_auth", "recovery_hint": "Check credentials with apis.supervisor.show_account_passwords(), then log in again; do not repeat invalid credentials."}
        if "takes 0 positional arguments" in error:
            return {"error_type": "positional_arg_error", "recovery_hint": "Pass AppWorld API arguments by keyword, including answer= for complete_task."}
        if "Usage of the following function is not allowed" in error or "not allowed" in error and ("csv.writer" in error or "open(" in error):
            return {"error_type": "sandbox_restricted", "recovery_hint": "Use plain Python strings and documented apis.file_system operations; avoid restricted host functions."}
        if "not JSON serializable" in error:
            return {"error_type": "json_not_serializable", "recovery_hint": "Print values or names, not apis function or module objects."}
        return None

    @staticmethod
    def _schema_identity(requested_name: str, schema: Mapping[str, Any]) -> str:
        app_name, api_name = schema.get("app_name"), schema.get("api_name")
        if isinstance(app_name, str) and isinstance(api_name, str):
            return f"{app_name}.{api_name}".lower()
        return requested_name.strip().lower()
