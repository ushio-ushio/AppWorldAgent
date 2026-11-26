from __future__ import annotations

import ast
import hashlib
import json
import re
from typing import Any

from tool_engineering.tools import ApiRetriever, compact_api_schema


def _failed_python_action(raw_arguments: Any) -> tuple[str, str | None]:
    """Return an opaque code digest and a literal API name, never code or secrets."""
    try:
        code = json.loads(raw_arguments).get("code", "")
    except (TypeError, ValueError, AttributeError):
        code = ""
    if not isinstance(code, str):
        code = ""
    names = re.findall(r"\bapis\.([A-Za-z_]+\.[A-Za-z_]+)", code)
    return hashlib.sha256(code.strip().encode("utf-8")).hexdigest(), names[-1].lower() if names else None

def _python_api_names(raw_arguments: Any) -> tuple[str, ...]:
    """Return syntactically mentioned APIs for NOT_EXECUTED diagnostics only."""
    try:
        code = json.loads(raw_arguments).get("code", "")
        tree = ast.parse(code) if isinstance(code, str) else None
    except (TypeError, ValueError, SyntaxError, AttributeError):
        return ()
    if tree is None:
        return ()
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Attribute)
                and isinstance(fn.value.value, ast.Name) and fn.value.value.id == "apis"):
            found.append(f"{fn.value.attr}.{fn.attr}".lower())
    return tuple(found)

def _explicit_business_rejection(error: str, status_code: int | None) -> bool:
    """Require a concrete refusal signal before treating a write response as definite."""
    if status_code is None or not 400 <= status_code < 500:
        return False
    # These responses indicate that the server rejected the request before
    # applying it; a conflict (409) is only definitive with an explicit body.
    if status_code in {400, 401, 403, 404, 422}:
        return True
    # A status alone (including 409) does not prove that a mutation was not
    # applied. Look for a response message that explicitly rejects the request.
    return bool(re.search(
        r"already exists|invalid (?:argument|parameter|credentials?|request)|"
        r"missing required|permission denied|not authorized|forbidden|"
        r"does not exist|not found|validation failed|unprocessable",
        error, re.I))

def _executed_write_fingerprints(
        fingerprints: list[tuple[str, str]],
        executed_requests: tuple[dict[str, Any], ...] | None) -> list[tuple[str, str]]:
    """Keep only planned writes confirmed by actual Requester call events."""
    counts: dict[str, int] = {}
    for row in executed_requests or ():
        name = str(row.get("api_name") or "").lower()
        if name and str(row.get("method") or "").upper() not in {"GET", "HEAD"}:
            counts[name] = counts.get(name, 0) + 1
    result = []
    for fingerprint, api_name in fingerprints:
        key = api_name.lower()
        if re.search(r"\.(login|logout|refresh_token|complete_task)$", key):
            continue
        if counts.get(key, 0) > 0:
            result.append((fingerprint, api_name))
            counts[key] -= 1
    return result

def _persistent_python_names(code: str) -> set[str]:
    """Names a later shell action can use to recover a local computation failure."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, TypeError):
        return set()
    local_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for generator in node.generators:
                local_names.update(child.id for child in ast.walk(generator.target)
                                   if isinstance(child, ast.Name))
        elif isinstance(node, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                local_names.add(arg.arg)
            if args.vararg:
                local_names.add(args.vararg.arg)
            if args.kwarg:
                local_names.add(args.kwarg.arg)
    builtin_names = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
    loaded = {node.id for node in ast.walk(tree)
              if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    return loaded - local_names - builtin_names - {"apis", "print"}

def _guard_blocking_correction(correction: dict[str, Any] | None) -> bool:
    """Only unresolved business writes or uncertain side effects block task submission.

    Read-only discovery failures and local Python computation errors are still
    available to ReAct as correction context, but they are not task-level proof
    that a requested business action remains incomplete.
    """
    if not correction:
        return False
    if correction.get("api_name") == "supervisor.complete_task":
        return bool(correction.get("uncertain_write"))
    return bool(correction.get("uncertain_write") or correction.get("business_write_error"))

def _has_unresolved_business_error(runtime: dict[str, Any]) -> bool:
    return any(not row.get("resolved") and _guard_blocking_correction(row)
               for row in runtime.get("business_error_ledger", ()))

def _correction_recovered(correction: dict[str, Any], *, function_name: str,
                          raw_arguments: Any, outcome: dict[str, Any],
                          executed_requests: tuple[dict[str, Any], ...] | None) -> bool:
    """Match a recovery to the failed operation; unrelated success is not enough."""
    if function_name != "execute_python" or outcome.get("ok") is not True:
        return False
    requests = executed_requests or ()
    if correction.get("uncertain_write"):
        return False
    failed_api = correction.get("api_name")
    failed_methods = set(correction.get("request_methods", ()))
    if correction.get("execution_state") == "DEFINITE_FAILURE" and failed_api:
        failed_app, _, failed_operation = failed_api.partition(".")
        # A rejected sign-up may be safely recovered by authenticating the
        # already-existing account; this does not repeat the rejected write.
        if failed_operation == "signup" and any(
                str(row.get("api_name", "")).lower() == f"{failed_app}.login"
                and str(row.get("method", "")).upper() not in {"GET", "HEAD"}
                for row in requests):
            return True
    if failed_api and requests:
        failed_entities = set(correction.get("entity_keys", ()))
        for row in requests:
            if str(row.get("api_name", "")).lower() != failed_api:
                continue
            if failed_methods and str(row.get("method", "")).upper() not in failed_methods:
                continue
            current_entities = {str(row["entity_key"])} if row.get("entity_key") else set()
            if failed_entities and current_entities and not failed_entities.intersection(current_entities):
                continue
            return True
        return False
    if failed_api and not requests and _failed_python_action(raw_arguments)[1] == failed_api:
        # Fake/test worlds may not expose their HTTP recorder. Keep recovery
        # scoped to a successful retry of the same API call.
        return True
    if correction.get("error_source") == "python_local" and not requests:
        try:
            code = json.loads(raw_arguments).get("code", "")
        except (TypeError, ValueError, AttributeError):
            return False
        if not isinstance(code, str):
            return False
        required = set(correction.get("recovery_names", ()))
        return bool(required and required.intersection(_persistent_python_names(code)))
    return False

def _same_target_readback(correction: dict[str, Any],
                          executed_requests: tuple[dict[str, Any], ...] | None, *,
                          outcome: dict[str, Any] | None = None,
                          request_events: list[Any] | tuple[Any, ...] | None = None) -> bool:
    target_keys = set(correction.get("target_keys", ()))
    entity_keys = set(correction.get("entity_keys", ()))
    if not correction.get("uncertain_write") or not (target_keys or entity_keys):
        return False
    if outcome is not None and outcome.get("ok") is not True:
        return False
    rows = [row for row in (executed_requests or ())
            if str(row.get("method", "")).upper() in {"GET", "HEAD"}
            and (row.get("target_key") in target_keys or
                 (entity_keys and row.get("entity_key") in entity_keys))]
    if not rows:
        return False
    if request_events:
        successful = set()
        for event in request_events:
            try:
                event = (event.as_dict() if hasattr(event, "as_dict") else
                         vars(event) if hasattr(event, "__dict__") else event)
                if event.get("execution_status") == "succeeded" and event.get("api_name"):
                    successful.add(str(event["api_name"]).lower())
            except (AttributeError, TypeError, ValueError):
                continue
        if not successful.intersection(str(row.get("api_name", "")).lower() for row in rows):
            return False
    return True

def _local_correction(error: str, *, raw_arguments: Any,
                      executed_requests: tuple[dict[str, Any], ...] | None,
                      retriever: ApiRetriever | None,
                      request_events: list[Any] | tuple[Any, ...] | None = None) -> dict[str, Any]:
    """Classify a failed action without copying traceback or credentials to context."""
    digest, _literal_api = _failed_python_action(raw_arguments)
    try:
        failed_code = json.loads(raw_arguments).get("code", "")
    except (TypeError, ValueError, AttributeError):
        failed_code = ""
    if not isinstance(failed_code, str):
        failed_code = ""
    requests = executed_requests or ()
    event_rows = []
    for event in request_events or ():
        try:
            row = event.as_dict() if hasattr(event, "as_dict") else dict(event)
        except (TypeError, ValueError):
            continue
        event_rows.append({key: row.get(key) for key in (
            "api_name", "execution_status", "http_status", "reason", "operation_type")})
    failed_events = [row for row in event_rows if row.get("execution_status") in {"failed", "unknown"}]
    failed_event = failed_events[-1] if failed_events else None
    api_status = re.search(r"(?:status(?: code)?\s*(?:is|:|=)?\s*|HTTP\s*)(\d{3})\b", error, re.I)
    status_code = (failed_event.get("http_status") if failed_event and
                   isinstance(failed_event.get("http_status"), int) else
                   int(api_status.group(1)) if api_status else None)
    # A source-code API name is not evidence that the Requester was reached.
    # For a local shell/runtime failure, no business API is assigned as failed.
    network_uncertain = bool(re.search(
        r"timeout|timed out|connection reset|connection aborted|broken pipe|remote end closed|"
        r"incomplete read|read error", error, re.I))
    api_response_error = bool(requests and (status_code is not None or failed_event or network_uncertain))
    api_name = (next((str(row["api_name"]).lower() for row in reversed(requests)
                      if row.get("api_name")), None) if api_response_error else None)
    if failed_event and failed_event.get("api_name"):
        api_name = str(failed_event["api_name"]).lower()
    if api_response_error and status_code is not None:
        category = f"HTTP {status_code}"
    elif re.search(r"Unexpected parameter|takes 0 positional arguments", error, re.I):
        category = "parameter_validation_error"
    elif re.search(r"\bTypeError\b", error):
        category = "Python TypeError"
    elif status_code is not None and api_response_error:
        category = f"HTTP {status_code}"
    else:
        match = re.search(r"\b([A-Za-z_]+Error|[A-Za-z_]+Exception)\b", error)
        category = match.group(1) if match else "tool_execution_error"
    reason = ({"HTTP 401": "Authentication failed. Check the documented account identity and matching app password; do not guess credentials.",
               "HTTP 422": "API rejected the request. Compare required arguments, types, bounds, and constraints with the official schema.",
               "parameter_validation_error": "The API rejected the call signature before completing it. Correct the parameters using the official schema.",
               "Python TypeError": "Python call signature or value type is invalid. Compare the call with the official schema and the observed result."}
              .get(category, "Tool execution failed. Inspect the referenced Observation and correct the action."))
    if category == "HTTP 409" and _explicit_business_rejection(error, status_code):
        reason = ("The server explicitly rejected this write because of a business conflict. Do not repeat the same write; "
                  "use the returned conflict reason to choose a non-duplicating recovery action.")
    schema = retriever.get_api_schema(api_name) if retriever and api_name else None
    compact = compact_api_schema(api_name, schema, max_chars=2500) if schema else None
    failed_names = {str(row.get("api_name") or "").lower() for row in failed_events
                    if row.get("api_name")}
    scoped_requests = [row for row in requests if str(row.get("api_name") or "").lower() in failed_names]
    if not scoped_requests and api_name:
        scoped_requests = [row for row in requests
                           if str(row.get("api_name") or "").lower() == api_name]
    if not scoped_requests and api_name is None:
        scoped_requests = []
    request_methods = tuple(dict.fromkeys(str(row.get("method", "")).upper()
                                          for row in scoped_requests if row.get("method")))
    target_keys = tuple(dict.fromkeys(str(row.get("target_key")) for row in scoped_requests
                                      if row.get("target_key")))
    entity_keys = tuple(dict.fromkeys(str(row.get("entity_key")) for row in scoped_requests
                                      if row.get("entity_key")))
    method = request_methods[-1] if request_methods else ""
    excluded_write = bool(api_name and re.search(
        r"\.(login|logout|complete_task|refresh_token)$", api_name, re.I))
    local_error = not api_response_error or category in {"Python TypeError", "parameter_validation_error"}
    network_uncertain = network_uncertain or bool(
        failed_event and failed_event.get("execution_status") == "unknown")
    definite_rejection = _explicit_business_rejection(error, status_code)
    execution_state = ("LOCAL_ERROR" if local_error else
                       "UNCERTAIN_WRITE" if method not in {"GET", "HEAD", ""} and
                           (network_uncertain or (status_code is not None and status_code >= 500) or
                            (status_code is not None and 400 <= status_code < 500 and
                             not definite_rejection) or
                            (failed_event and (failed_event.get("execution_status") == "unknown" or
                             (failed_event.get("execution_status") == "failed" and
                              status_code is None and not definite_rejection)))) else
                       "DEFINITE_FAILURE")
    business_write_error = bool(method and method not in {"GET", "HEAD"} and not excluded_write
                               and execution_state == "DEFINITE_FAILURE")
    recovery_names = (_persistent_python_names(failed_code) if local_error and category in {
        "KeyError", "IndexError", "NameError"} else set())
    uncertain_write = execution_state == "UNCERTAIN_WRITE"
    request_statuses = []
    event_by_api: dict[str, list[dict[str, Any]]] = {}
    for row in event_rows:
        event_by_api.setdefault(str(row.get("api_name") or "").lower(), []).append(row)
    for row in requests:
        name = str(row.get("api_name") or "").lower()
        matching = event_by_api.get(name, [])
        event = matching.pop(0) if matching else None
        if event:
            state = {"succeeded": "SUCCESS", "failed": "DEFINITE_FAILURE",
                     "unknown": "UNCERTAIN_WRITE"}.get(str(event.get("execution_status")), "NOT_EXECUTED")
            if state == "DEFINITE_FAILURE" and name == api_name and execution_state == "UNCERTAIN_WRITE":
                state = "UNCERTAIN_WRITE"
        elif name == api_name:
            state = execution_state
        else:
            state = "SUCCESS_OR_RESPONSE_UNKNOWN"
        request_statuses.append({"api_name": name or None, "method": row.get("method"),
                                 "execution_state": state,
                                 "http_status": event.get("http_status") if event else None})
    observed_counts: dict[str, int] = {}
    for row in requests:
        name = str(row.get("api_name") or "").lower()
        if name:
            observed_counts[name] = observed_counts.get(name, 0) + 1
    api_attempts = []
    for name in _python_api_names(raw_arguments):
        if observed_counts.get(name, 0):
            observed_counts[name] -= 1
        else:
            api_attempts.append({"api_name": name, "execution_state": "NOT_EXECUTED"})
    return {"category": category, "api_name": api_name, "reason": reason,
            "code_digest": digest, "official_schema": compact,
            "execution_state": execution_state, "request_outcomes": request_statuses,
        "api_attempts": api_attempts,
        "request_events": event_rows,
            "uncertain_write": uncertain_write, "business_write_error": business_write_error,
            "error_source": "python_local" if local_error else "business_api",
            "request_methods": request_methods, "target_keys": target_keys,
            "entity_keys": entity_keys,
            "recovery_names": sorted(recovery_names),
            "attempts": 1}

def _documented_readback_only(code: str, retriever: ApiRetriever | None) -> bool:
    """Conservative gate after an uncertain write; arbitrary Python is not provably read-only."""
    if retriever is None:
        return False
    try:
        tree = ast.parse(code)
    except (SyntaxError, TypeError):
        return False
    api_calls = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id == "print":
            continue
        if not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Attribute)
                and isinstance(fn.value.value, ast.Name) and fn.value.value.id == "apis"):
            return False
        schema = retriever.get_api_schema(f"{fn.value.attr}.{fn.attr}")
        if not schema or str(schema.get("method", "")).upper() not in {"GET", "HEAD"}:
            return False
        api_calls += 1
    return (api_calls > 0 and sum(isinstance(node, ast.Name) and node.id == "apis"
                                  for node in ast.walk(tree)) == api_calls)

def _safe_during_uncertain_write(code: str, correction: dict[str, Any],
                                 retriever: ApiRetriever | None) -> bool:
    """Permit readback/auth recovery and provably cross-app writes, never same-scope replay."""
    if _documented_readback_only(code, retriever):
        return True
    try:
        tree = ast.parse(code)
    except (SyntaxError, TypeError):
        return False
    api_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id in {"print", "str", "len"}:
            continue
        if not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Attribute)
                and isinstance(fn.value.value, ast.Name) and fn.value.value.id == "apis"):
            # Pure local computation cannot repeat a business side effect.
            if any(isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name)
                   and child.value.id == "apis" for child in ast.walk(node)):
                return False
            continue
        app, api = fn.value.attr.lower(), fn.attr.lower()
        name = f"{app}.{api}"
        schema = retriever.get_api_schema(name) if retriever else None
        method = str((schema or {}).get("method", "")).upper()
        if method in {"GET", "HEAD"} or api in {"login", "logout", "refresh_token"}:
            api_calls.append((app, api, "safe"))
        else:
            api_calls.append((app, api, "write"))
    if not api_calls:
        return True
    uncertain_app = str(correction.get("api_name") or "").split(".", 1)[0].lower()
    return all(kind == "safe" or (uncertain_app and app != uncertain_app)
               for app, _api, kind in api_calls)

def write_operation_fingerprints(code: str, retriever: ApiRetriever | None) -> list[tuple[str, str]]:
    """Find literal non-read AppWorld calls; no dynamic code execution or guessing."""
    if retriever is None:
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    parents = {child: parent for parent in ast.walk(tree)
               for child in ast.iter_child_nodes(parent)}
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        parts = []
        current: ast.AST = node.func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name) or current.id != "apis" or len(parts) < 2:
            continue
        api_name = f"{parts[-1]}.{parts[-2]}"
        schema = retriever.get_api_schema(api_name)
        if not schema or str(schema.get("method", "GET")).upper() in {"GET", "HEAD"}:
            continue
        # Include the concrete AST call so the same statement is blocked on replay,
        # while a changed target/arguments remain a separately auditable operation.
        signature: ast.AST = node
        parent = parents.get(node)
        while parent is not None:
            if isinstance(parent, (ast.For, ast.AsyncFor)):
                signature = parent.iter
                break
            parent = parents.get(parent)
        fingerprint_source = ast.dump(node, include_attributes=False)
        if signature is not node:
            fingerprint_source += "|loop=" + ast.dump(signature, include_attributes=False)
        fingerprint = hashlib.sha256(fingerprint_source.encode()).hexdigest()
        found[fingerprint] = api_name
    return list(found.items())
