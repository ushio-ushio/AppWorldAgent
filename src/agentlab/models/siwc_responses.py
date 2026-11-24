"""Account-scoped SIWC Responses adapter for the existing Function Calling graph."""
from __future__ import annotations

import json
import os
import time
from typing import Any

import requests

from agentlab.models.siwc_auth import RESOURCE, SIWCError, access_token


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
                 request_may_have_been_received: bool = True):
        self.diagnostic = {
            "phase": phase, "error_class": error_class, "request_bytes": request_bytes,
            "events_seen": events_seen, "partial_output": partial_output,
            "tool_signal_seen": tool_signal_seen,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "request_may_have_been_received": request_may_have_been_received,
        }
        super().__init__("SIWC transport failed: " + phase + " (" + error_class + ")")


class ChatGPTResponsesAdapter:
    def __init__(self, model_name: str | None = None, *, timeout_seconds: int = 90,
                 connect_timeout_seconds: int = 20):
        self.timeout_seconds = timeout_seconds
        self.connect_timeout_seconds = connect_timeout_seconds
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

    def _request(self, method: str, endpoint: str, *, body: dict | None = None, stream: bool = False):
        # A POST can reach the service even if no response headers arrive.
        # Replaying it could duplicate model usage. No Tool Node action executes
        # until a complete response has been parsed, so fail closed instead.
        last_exc = None
        max_attempts = 1 if method.upper() == "POST" else 3
        for attempt in range(max_attempts):
            try:
                result = self._session.request(method, RESOURCE + endpoint,
                    headers={"Authorization": "Bearer " + access_token(), "Content-Type": "application/json"},
                    json=body, stream=stream,
                    timeout=(self.connect_timeout_seconds, self.timeout_seconds))
            except (requests.RequestException, OSError) as exc:
                last_exc = exc
                if attempt + 1 < max_attempts:
                    time.sleep(0.75 * (2 ** attempt))
                    continue
                phase = "connect" if isinstance(exc, requests.ConnectTimeout) else "awaiting_headers"
                raise SIWCTransportError(
                    phase=phase, error_class=type(exc).__name__,
                    request_may_have_been_received=phase != "connect") from exc
            if result.status_code == 429:
                # SIWC usage-limit 429s must stop new work; replaying the same
                # request cannot restore quota and can obscure the limit signal.
                try:
                    error_payload = result.json()
                    error_code = (error_payload.get("error") or {}).get("code")
                except (ValueError, AttributeError, TypeError):
                    error_code = None
                result.close()
                if error_code == "subscription_sharing_usage_limit_exceeded":
                    raise SIWCError("ChatGPT plan or app usage limit reached; stop new requests and review ChatGPT Usage")
                raise self._http_error(429)
            if 500 <= result.status_code <= 599:
                code = result.status_code
                result.close()
                if attempt + 1 < max_attempts:
                    time.sleep(0.75 * (2 ** attempt))
                    continue
                raise SIWCError(f"SIWC endpoint HTTP {code} after {max_attempts} request attempt(s)")
            if result.status_code != 200:
                code = result.status_code
                result.close()
                raise self._http_error(code)
            return result
        raise SIWCError("SIWC request failed") from last_exc

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
        started = _wall_seconds()
        request_bytes = len(json.dumps(body, ensure_ascii=False, default=str).encode("utf-8"))
        try:
            connection = self._request("POST", "/responses", body=body, stream=True)
        except SIWCTransportError as exc:
            exc.diagnostic.update(request_bytes=request_bytes,
                                  elapsed_seconds=round(_wall_seconds() - started, 3),
                                  connect_timeout_seconds=self.connect_timeout_seconds,
                                  read_timeout_seconds=self.timeout_seconds)
            raise
        completed = None
        failure = None
        events_seen = 0
        partial_output = False
        tool_signal_seen = False
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
                    raise SIWCTransportError(
                        phase="streaming", error_class="InvalidEventJSON",
                        request_bytes=request_bytes, events_seen=events_seen,
                        partial_output=partial_output, tool_signal_seen=tool_signal_seen,
                        elapsed_seconds=_wall_seconds() - started) from exc
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
                elif event_type == "response.failed":
                    failure = event.get("response", {}).get("error", {}).get("code", "unknown")
        except (requests.RequestException, OSError, UnicodeError) as exc:
            raise SIWCTransportError(
                phase="streaming", error_class=type(exc).__name__,
                request_bytes=request_bytes, events_seen=events_seen,
                partial_output=partial_output, tool_signal_seen=tool_signal_seen,
                elapsed_seconds=_wall_seconds() - started) from exc
        finally:
            connection.close()
        if failure:
            raise SIWCError(f"SIWC response failed: {failure}")
        if not completed:
            raise SIWCTransportError(
                phase="stream_ended", error_class="IncompleteStream",
                request_bytes=request_bytes, events_seen=events_seen,
                partial_output=partial_output, tool_signal_seen=tool_signal_seen,
                elapsed_seconds=_wall_seconds() - started)
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
        return {"role": "assistant", "content": "".join(texts) or None, "tool_calls": calls}, usage
