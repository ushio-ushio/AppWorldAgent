"""Account-scoped SIWC Responses adapter for the existing Function Calling graph."""
from __future__ import annotations

import json
import os
import random
import time
import uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import requests

from models.siwc_auth import RESOURCE, SIWCError, access_token


if os.name == "nt":
    import ctypes

    _get_tick_count = ctypes.windll.kernel32.GetTickCount64
    _get_tick_count.restype = ctypes.c_ulonglong


def _wall_seconds() -> float:
    # AppWorld freezes Python's time.perf_counter during a task. Windows' OS
    # tick count remains monotonic and measures actual network wait time.
    return _get_tick_count() / 1000 if os.name == "nt" else time.monotonic()


class SIWCTransportError(SIWCError):
    """Safe stream diagnostics; never includes response text or credentials."""

    def __init__(self, *, phase: str, error_class: str, request_bytes: int = 0,
                 events_seen: int = 0, partial_output: bool = False,
                 tool_signal_seen: bool = False, elapsed_seconds: float = 0.0,
                 request_may_have_been_received: bool = True, status_code: int | None = None):
        self.diagnostic = {
            "phase": phase, "error_class": error_class, "request_bytes": request_bytes,
            "events_seen": events_seen, "partial_output": partial_output,
            "tool_signal_seen": tool_signal_seen,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "request_may_have_been_received": request_may_have_been_received,
        }
        if status_code is not None:
            self.diagnostic["status_code"] = status_code
        super().__init__("SIWC transport failed: " + phase + " (" + error_class + ")")


class SIWCContextOverflowError(SIWCError):
    """HTTP 400 explicitly reporting that the submitted context is too large."""


class ChatGPTResponsesAdapter:
    def __init__(self, model_name: str | None = None, *, timeout_seconds: int = 90,
                 connect_timeout_seconds: int = 20, session_id: str | None = None):
        self.timeout_seconds = timeout_seconds
        self.connect_timeout_seconds = connect_timeout_seconds
        self.max_transport_retries = 3
        self.session_id = session_id or str(uuid.uuid4())
        if not self.session_id.strip():
            raise ValueError("session_id must be non-empty")
        # Each AppWorld worker owns one adapter. Reuse its proxy/TLS pool across
        # model turns instead of constructing a new Session for every request.
        self._session = requests.Session()
        try:
            self.available_models = self.list_models()
            slugs = [item["slug"] for item in self.available_models]
            if not slugs:
                raise SIWCError("No account-visible SIWC models available")
            if model_name is not None and model_name not in slugs:
                raise SIWCError("Requested model is not in this account's model list")
            self.model_name = model_name or slugs[0]
        except Exception:
            self._session.close()
            raise

    def close(self) -> None:
        self._session.close()

    def _rebuild_session(self) -> None:
        old = self._session
        self._session = requests.Session()
        old.close()

    @staticmethod
    def _contains_permission_error(exc: BaseException) -> bool:
        pending, seen = [exc], set()
        while pending:
            item = pending.pop()
            if id(item) in seen:
                continue
            seen.add(id(item))
            if isinstance(item, PermissionError):
                return True
            pending.extend(x for x in (getattr(item, "__cause__", None),
                                       getattr(item, "__context__", None),
                                       getattr(item, "reason", None)) if isinstance(x, BaseException))
        return False

    @staticmethod
    def _phase(exc: BaseException, *, streaming: bool = False) -> str:
        if streaming:
            return "sse_streaming"
        if isinstance(exc, requests.exceptions.SSLError):
            return "tls_handshake"
        if isinstance(exc, (requests.exceptions.ProxyError, requests.exceptions.ConnectTimeout,
                            requests.exceptions.ConnectionError, ConnectionError)):
            return "proxy_or_connect"
        return "awaiting_headers"

    @staticmethod
    def _retry_after_seconds(headers: dict) -> float | None:
        value = headers.get("Retry-After") if headers else None
        if not value:
            return None
        try:
            return max(0.0, min(15.0, float(value)))
        except (ValueError, TypeError):
            try:
                date = parsedate_to_datetime(value)
                return max(0.0, min(15.0, (date - datetime.now(timezone.utc)).total_seconds()))
            except (TypeError, ValueError, OverflowError):
                return None

    def _attach_retries(self, exc: BaseException, state: dict, *, phase: str,
                        status_code: int | None = None) -> None:
        if state["history"]:
            diagnostic = getattr(exc, "diagnostic", {})
            diagnostic.update(phase=phase, transport_retry_count=len(state["history"]),
                              transport_retries=state["history"])
            if status_code is not None:
                diagnostic["status_code"] = status_code
            exc.diagnostic = diagnostic

    def _retry(self, state: dict, *, stage: str, error_type: str,
               status_code: int | None = None, retry_after: float | None = None,
               rebuild: bool = False) -> None:
        retry_number = len(state["history"])
        delay = retry_after if retry_after is not None else min(15.0, 2 ** retry_number) + random.uniform(0, 0.5)
        state["history"].append({"attempt": state["attempt"], "stage": stage,
                                 "error_type": error_type, "status_code": status_code,
                                 "backoff_ms": round(delay * 1000)})
        if rebuild:
            self._rebuild_session()
        time.sleep(delay)

    @staticmethod
    def _http_error(status: int) -> SIWCError:
        if status == 401:
            return SIWCError("SIWC authorization expired or invalid")
        if status == 403:
            return SIWCError("ChatGPT plan or selected model unavailable")
        if status == 429:
            return SIWCError("ChatGPT plan quota/rate limit reached")
        if status == 400:
            return SIWCError("SIWC Responses tool/request format incompatible")
        return SIWCError(f"SIWC endpoint HTTP {status}")

    def _request(self, method: str, endpoint: str, *, body: dict | None = None,
                 stream: bool = False, _retry_state: dict | None = None):
        if not getattr(self, "session_id", None):
            self.session_id = str(uuid.uuid4())
        state = _retry_state if _retry_state is not None else {"attempt": 0, "history": []}
        max_attempts = getattr(self, "max_transport_retries", 3) + 1
        retryable_statuses = {408, 429, 500, 502, 503, 504}
        while state["attempt"] < max_attempts:
            state["attempt"] += 1
            try:
                result = self._session.request(method, RESOURCE + endpoint,
                    headers={"Authorization": "Bearer " + access_token(), "Content-Type": "application/json",
                             "session-id": self.session_id},
                    json=body, stream=stream,
                    timeout=(self.connect_timeout_seconds, self.timeout_seconds))
            except SIWCError as exc:
                self._attach_retries(exc, state, phase="awaiting_headers")
                raise
            except (requests.RequestException, OSError) as exc:
                permission_denied = self._contains_permission_error(exc)
                retryable = isinstance(exc, (requests.exceptions.SSLError,
                    requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                    requests.exceptions.ChunkedEncodingError, ConnectionError, TimeoutError))
                if retryable and not permission_denied and state["attempt"] < max_attempts:
                    rebuild = isinstance(exc, (requests.exceptions.SSLError,
                        requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError,
                        ConnectionError))
                    self._retry(state, stage=self._phase(exc), error_type=type(exc).__name__, rebuild=rebuild)
                    continue
                phase = self._phase(exc)
                error = SIWCTransportError(phase=phase, error_class=type(exc).__name__,
                    request_may_have_been_received=not isinstance(exc, (requests.exceptions.ProxyError,
                        requests.exceptions.ConnectTimeout, requests.exceptions.SSLError)))
                self._attach_retries(error, state, phase=phase)
                raise error from exc
            if result.status_code == 429:
                # SIWC usage-limit 429s must stop new work; replaying the same
                # request cannot restore quota and can obscure the limit signal.
                try:
                    error_payload = result.json()
                    error_code = (error_payload.get("error") or {}).get("code")
                except (ValueError, AttributeError, TypeError):
                    error_code = None
                if error_code == "subscription_sharing_usage_limit_exceeded":
                    result.close()
                    error = SIWCError("ChatGPT plan or app usage limit reached; stop new requests and review ChatGPT Usage")
                    self._attach_retries(error, state, phase="http_response", status_code=429)
                    raise error
            if result.status_code in retryable_statuses:
                code = result.status_code
                headers = getattr(result, "headers", {}) or {}
                result.close()
                if state["attempt"] < max_attempts:
                    self._retry(state, stage="http_response", error_type="HTTPStatus",
                                status_code=code, retry_after=self._retry_after_seconds(headers))
                    continue
                error = (self._http_error(code) if code == 429 else
                         SIWCError(f"SIWC endpoint HTTP {code} after {state['attempt']} request attempt(s)"))
                self._attach_retries(error, state, phase="http_response", status_code=code)
                raise error
            if result.status_code != 200:
                code = result.status_code
                if code == 400:
                    try:
                        payload_text = json.dumps(result.json(), ensure_ascii=False).lower()
                    except Exception:
                        payload_text = ""
                    if any(term in payload_text for term in ("context_length_exceeded", "maximum context length",
                            "context window", "reduce the length")):
                        result.close()
                        raise SIWCContextOverflowError("SIWC request context exceeded the model window")
                result.close()
                error = self._http_error(code)
                self._attach_retries(error, state, phase="http_response", status_code=code)
                raise error
            return result
        raise SIWCError("SIWC request failed")

    def list_models(self) -> list[dict[str, str]]:
        result = self._request("GET", "/models")
        try:
            items = result.json().get("models", [])
        finally:
            result.close()
        return [{"slug": item["slug"], "display_name": item.get("display_name", item["slug"])}
                for item in items if item.get("visibility") == "list" and item.get("slug")]

    @staticmethod
    def _input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items = []
        for message in messages:
            role = message.get("role")
            if role == "tool":
                items.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                              "output": str(message.get("content", ""))})
            elif role == "assistant":
                if message.get("content"):
                    items.append({"role": "assistant", "content": str(message["content"])})
                for call in message.get("tool_calls") or []:
                    function = call["function"]
                    items.append({"type": "function_call", "call_id": call["id"],
                                  "name": function["name"], "arguments": function["arguments"]})
            elif role in {"system", "developer", "user"}:
                # SIWC rejects explicit system message items; developer is supported.
                items.append({"role": "developer" if role == "system" else role,
                              "content": str(message.get("content", ""))})
        return items

    @staticmethod
    def _tools(tools: list[dict[str, Any]]) -> dict[str, Any]:
        converted = []
        for definition in tools:
            function = definition["function"]
            converted.append({"type": "function", "name": function["name"],
                              "description": function.get("description", ""),
                              "parameters": function["parameters"], "strict": False})
        return {"type": "additional_tools", "role": "developer", "tools": converted}

    def generate_tool_calls(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                            *, tool_choice: str = "required") -> tuple[dict, dict]:
        # The existing graph validates missing/malformed calls; avoid unsupported
        # SIWC fields and never expose hosted execution tools.
        del tool_choice
        self.last_transport = None
        body = {"model": self.model_name,
                "input": [self._tools(tools), *self._input(messages)],
                "store": False, "stream": True}
        if getattr(self, "prompt_cache_key", None):
            body["prompt_cache_key"] = self.prompt_cache_key
        started = _wall_seconds()
        request_bytes = len(json.dumps(body, ensure_ascii=False, default=str).encode("utf-8"))
        retry_state = {"attempt": 0, "history": []}
        while True:
            completed, failure = None, None
            events_seen, partial_output, tool_signal_seen = 0, False, False
            try:
                connection = self._request("POST", "/responses", body=body, stream=True,
                                           _retry_state=retry_state)
            except SIWCError as exc:
                if isinstance(getattr(exc, "diagnostic", None), dict):
                    exc.diagnostic.update(request_bytes=request_bytes,
                        elapsed_seconds=round(_wall_seconds() - started, 3),
                        connect_timeout_seconds=self.connect_timeout_seconds,
                        read_timeout_seconds=self.timeout_seconds)
                raise
            try:
                for line in connection.iter_lines(decode_unicode=True):
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")
                    if not line or not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        event = json.loads(line[6:])
                        if not isinstance(event, dict):
                            raise ValueError("SSE event is not an object")
                    except (ValueError, TypeError) as exc:
                        error = SIWCTransportError(phase="sse_streaming", error_class="InvalidEventJSON",
                            request_bytes=request_bytes, events_seen=events_seen,
                            partial_output=partial_output, tool_signal_seen=tool_signal_seen,
                            elapsed_seconds=_wall_seconds() - started)
                        self._attach_retries(error, retry_state, phase="sse_streaming")
                        raise error from exc
                    events_seen += 1
                    event_type = str(event.get("type", ""))
                    partial_output |= event_type in {"response.output_text.delta", "response.content_part.added"} or (
                        event_type == "response.output_item.added" and
                        (event.get("item") or {}).get("type") == "message")
                    tool_signal_seen |= event_type.startswith("response.function_call_arguments.") or (
                        event_type == "response.output_item.added" and
                        (event.get("item") or {}).get("type") == "function_call")
                    partial_output |= tool_signal_seen
                    if event_type == "response.completed":
                        completed = event.get("response")
                        break
                    if event_type == "response.failed":
                        failure = event.get("response", {}).get("error", {}).get("code", "unknown")
            except (requests.RequestException, OSError, UnicodeError) as exc:
                retryable = isinstance(exc, (requests.exceptions.SSLError,
                    requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                    requests.exceptions.ChunkedEncodingError, ConnectionError, TimeoutError))
                if retryable and not self._contains_permission_error(exc) and retry_state["attempt"] <= getattr(self, "max_transport_retries", 3):
                    self._retry(retry_state, stage="sse_streaming", error_type=type(exc).__name__, rebuild=True)
                    continue
                error = SIWCTransportError(phase="sse_streaming", error_class=type(exc).__name__,
                    request_bytes=request_bytes, events_seen=events_seen,
                    partial_output=partial_output, tool_signal_seen=tool_signal_seen,
                    elapsed_seconds=_wall_seconds() - started)
                self._attach_retries(error, retry_state, phase="sse_streaming")
                raise error from exc
            finally:
                connection.close()
            if failure:
                error = SIWCError(f"SIWC response failed: {failure}")
                self._attach_retries(error, retry_state, phase="sse_streaming")
                raise error
            if completed:
                break
            error = SIWCTransportError(phase="sse_streaming", error_class="IncompleteStream",
                request_bytes=request_bytes, events_seen=events_seen,
                partial_output=partial_output, tool_signal_seen=tool_signal_seen,
                elapsed_seconds=_wall_seconds() - started)
            if retry_state["attempt"] <= getattr(self, "max_transport_retries", 3):
                self._retry(retry_state, stage="sse_streaming", error_type="IncompleteStream", rebuild=True)
                continue
            self._attach_retries(error, retry_state, phase="sse_streaming")
            raise error
        calls = []
        texts = []
        for item in completed.get("output", []):
            if item.get("type") == "function_call":
                calls.append({"id": item["call_id"], "type": "function",
                              "function": {"name": item["name"], "arguments": item["arguments"]}})
            elif item.get("type") == "message":
                texts.extend(part.get("text", "") for part in item.get("content", [])
                             if part.get("type") == "output_text")
        raw_usage = completed.get("usage") or {}
        usage = {"input_tokens": int(raw_usage.get("input_tokens", 0)),
                 "input_cache_hit_tokens": int(raw_usage.get("input_tokens_details", {}).get("cached_tokens", 0)),
                 "output_tokens": int(raw_usage.get("output_tokens", 0)),
                 "cost_usd": 0.0}  # SIWC plan usage has no per-request dollar quote.
        self.last_transport = {"phase": "completed", "request_bytes": request_bytes,
                               "events_seen": events_seen, "partial_output": partial_output,
                               "tool_signal_seen": tool_signal_seen,
                               "elapsed_seconds": round(_wall_seconds() - started, 3)}
        if retry_state["history"]:
            self.last_transport.update(transport_retry_count=len(retry_state["history"]),
                                       transport_retries=retry_state["history"])
        return {"role": "assistant", "content": "".join(texts) or None, "tool_calls": calls}, usage

    def extract_context(self, messages: list[dict[str, Any]],
                        tools: list[dict[str, Any]] | None = None) -> tuple[dict[str, Any], dict[str, int]]:
        """Run a no-tools semantic extraction using an isolated HTTP session."""
        body = {"model": self.model_name,
                "input": [self._tools(tools or []), *self._input(messages)],
                "store": False, "stream": True}
        session = requests.Session()
        try:
            response = session.request("POST", RESOURCE + "/responses",
                headers={"Authorization": "Bearer " + access_token(),
                         "Content-Type": "application/json", "session-id": self.session_id},
                json=body, stream=True, timeout=(self.connect_timeout_seconds, self.timeout_seconds))
            if response.status_code != 200:
                raise SIWCError(f"SIWC extraction HTTP {response.status_code}")
            completed = None
            for line in response.iter_lines(decode_unicode=True):
                if isinstance(line, bytes): line = line.decode("utf-8", errors="replace")
                if line and line.startswith("data: ") and line != "data: [DONE]":
                    event = json.loads(line[6:])
                    if event.get("type") == "response.completed": completed = event.get("response")
            if not completed: raise SIWCError("SIWC extraction stream incomplete")
            text = "".join(part.get("text", "") for item in completed.get("output", [])
                           if item.get("type") == "message" for part in item.get("content", [])
                           if part.get("type") == "output_text")
            raw_usage = completed.get("usage") or {}
            usage = {"input_tokens": int(raw_usage.get("input_tokens", 0)),
                     "cached_tokens": int(raw_usage.get("input_tokens_details", {}).get("cached_tokens", 0)),
                     "output_tokens": int(raw_usage.get("output_tokens", 0))}
            return json.loads(text), usage
        finally:
            try: response.close()
            except UnboundLocalError: pass
            session.close()
