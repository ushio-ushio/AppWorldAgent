"""Minimal SIWC login and account-scoped Responses smoke checks."""
from __future__ import annotations

import argparse
import json
import sys

from agentlab.models.siwc_auth import RESOURCE, SIWCError, access_token, login
import requests


def _request(method: str, endpoint: str, *, body: dict | None = None, stream: bool = False):
    try:
        response = requests.request(method, RESOURCE + endpoint,
            headers={"Authorization": "Bearer " + access_token(), "Content-Type": "application/json"},
            json=body, stream=stream, timeout=(5, 90))
    except requests.RequestException as exc:
        raise SIWCError("Official inference endpoint connection failed") from exc
    if response.status_code != 200:
        code = response.status_code
        response.close()
        if code == 401:
            raise SIWCError("OAuth authorization failed or expired")
        if code == 403:
            raise SIWCError("ChatGPT plan or model access unavailable")
        if code == 429:
            raise SIWCError("ChatGPT plan quota/rate limit reached")
        if code == 400:
            raise SIWCError("Responses request or tool format incompatible (HTTP 400)")
        raise SIWCError(f"Official endpoint returned HTTP {code}")
    return response


def models() -> list[dict]:
    response = _request("GET", "/models")
    try:
        data = response.json()
    finally:
        response.close()
    return [item for item in data.get("models", []) if item.get("visibility") == "list" and item.get("slug")]


def response(input_items: list[dict], model: str) -> dict:
    body = {"model": model, "input": input_items, "store": False, "stream": True}
    connection = _request("POST", "/responses", body=body, stream=True)
    completed = None
    failure = None
    try:
        for line in connection.iter_lines(decode_unicode=True):
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            if not line or not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("type") == "response.completed":
                completed = event.get("response")
            elif event.get("type") == "response.failed":
                failure = event.get("response", {}).get("error", {}).get("code", "unknown")
    finally:
        connection.close()
    if failure:
        raise SIWCError(f"Response failed: {failure}")
    if not completed:
        raise SIWCError("Stream ended without response.completed")
    return completed


def _text(result: dict) -> str:
    return "".join(part.get("text", "") for item in result.get("output", [])
                   if item.get("type") == "message"
                   for part in item.get("content", []) if part.get("type") == "output_text")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("login", "models", "text", "function"))
    parser.add_argument("--model")
    args = parser.parse_args()
    try:
        if args.command == "login":
            print(json.dumps(login(), ensure_ascii=False))
            return
        choices = models()
        if args.command == "models":
            print(json.dumps([{"slug": x["slug"], "display_name": x.get("display_name")} for x in choices], ensure_ascii=False))
            return
        model = args.model or (choices[0]["slug"] if choices else None)
        if not model or model not in {x["slug"] for x in choices}:
            raise SIWCError("No account-visible model selected")
        if args.command == "text":
            result = response([{"role": "user", "content": "Reply with exactly: SIWC_OK"}], model)
            print(json.dumps({"model": model, "text": _text(result), "usage": result.get("usage")}, ensure_ascii=False))
            return
        tool = {"type": "additional_tools", "role": "developer", "tools": [{
            "type": "function", "name": "fake_lookup", "description": "Read a fake test value",
            "parameters": {"type": "object", "properties": {"key": {"type": "string"}},
                           "required": ["key"], "additionalProperties": False}, "strict": True}]}
        history = [tool, {"role": "user", "content": "Call fake_lookup with key test, then report its value."}]
        first = response(history, model)
        calls = [x for x in first.get("output", []) if x.get("type") == "function_call" and x.get("name") == "fake_lookup"]
        if len(calls) != 1 or not calls[0].get("call_id"):
            raise SIWCError("Fake function calling format incompatible: expected one fake_lookup call")
        args_obj = json.loads(calls[0]["arguments"])
        if args_obj != {"key": "test"}:
            raise SIWCError("Fake function arguments incompatible")
        history.extend(first.get("output", []))
        history.append({"type": "function_call_output", "call_id": calls[0]["call_id"], "output": "{\"value\":\"FAKE_OK\"}"})
        second = response(history, model)
        print(json.dumps({"model": model, "tool_call": "fake_lookup", "tool_result_fed_back": True,
                          "final_text": _text(second), "first_usage": first.get("usage"),
                          "second_usage": second.get("usage")}, ensure_ascii=False))
    except SIWCError as exc:
        print(f"SIWC_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
