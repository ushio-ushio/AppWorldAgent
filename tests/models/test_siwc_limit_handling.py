import unittest
from unittest.mock import Mock, patch
import requests
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone

from agentlab.models import siwc_auth
from agentlab.models.siwc_responses import ChatGPTResponsesAdapter, SIWCError, SIWCTransportError
from agentlab.agent.react import LangGraphAgent


class FakeResponse:
    def __init__(self, payload, status_code=429, headers=None):
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.closed = False

    def json(self):
        return self.payload

    def close(self):
        self.closed = True


class FakeStreamResponse:
    status_code = 200
    headers = {}

    def __init__(self, events):
        self.events = events
        self.closed = False

    def iter_lines(self, **_kwargs):
        yield from self.events

    def close(self):
        self.closed = True


def completed_event():
    return b'data: ' + json.dumps({"type": "response.completed", "response": {
        "output": [{"type": "function_call", "call_id": "call-1", "name": "search_api",
                    "arguments": "{}"}],
        "usage": {"input_tokens": 3, "output_tokens": 2}}}).encode()


class TestSIWCLimitHandling(unittest.TestCase):
    def test_connection_diagnostic_keeps_only_exception_class(self):
        adapter = ChatGPTResponsesAdapter.__new__(ChatGPTResponsesAdapter)
        adapter.timeout_seconds = 1
        adapter.connect_timeout_seconds = 2
        adapter.max_transport_retries = 3
        adapter._session = Mock()
        with patch("agentlab.models.siwc_responses.access_token", return_value="fake-token"), \
                patch("agentlab.models.siwc_responses.time.sleep"), \
                patch.object(adapter, "_rebuild_session"):
            adapter._session.request.side_effect = requests.exceptions.ProxyError("secret-detail")
            with self.assertRaisesRegex(SIWCError, "ProxyError") as caught:
                adapter._request("POST", "/responses", body={"model": "fake"})
        self.assertEqual(adapter._session.request.call_count, 4)
        self.assertEqual(adapter._session.request.call_args.kwargs["timeout"], (2, 1))
        self.assertNotIn("secret-detail", str(caught.exception))
        self.assertEqual(caught.exception.diagnostic["phase"], "proxy_or_connect")
        self.assertEqual(caught.exception.diagnostic["transport_retry_count"], 3)
        self.assertFalse(caught.exception.diagnostic["request_may_have_been_received"])

    def test_get_model_list_can_retry_transient_connection(self):
        adapter = ChatGPTResponsesAdapter.__new__(ChatGPTResponsesAdapter)
        adapter.timeout_seconds = 1
        adapter.connect_timeout_seconds = 2
        adapter.max_transport_retries = 3
        adapter._session = Mock()
        with patch("agentlab.models.siwc_responses.access_token", return_value="fake-token"), \
                patch("agentlab.models.siwc_responses.time.sleep"), \
                patch.object(adapter, "_rebuild_session"):
            adapter._session.request.side_effect = requests.exceptions.ConnectTimeout("secret-detail")
            with self.assertRaises(SIWCTransportError) as caught:
                adapter._request("GET", "/models")
        self.assertEqual(adapter._session.request.call_count, 4)
        self.assertEqual(caught.exception.diagnostic["phase"], "proxy_or_connect")
        self.assertFalse(caught.exception.diagnostic["request_may_have_been_received"])

    def test_partial_stream_never_dispatches_tool(self):
        class PartialResponse:
            closed = False

            def iter_lines(self, **_kwargs):
                yield b'data: {"type":"response.output_item.added","item":{"type":"function_call"}}'
                raise requests.exceptions.ConnectionError("secret-detail")

            def close(self):
                self.closed = True

        class FakeWorld:
            task_id = "siwc-fake"
            task = SimpleNamespace(instruction="Read a record.", allowed_apps=[], api_docs={})

            def __init__(self):
                self.executed = []

            def execute(self, code):
                self.executed.append(code)
                return "unexpected"

            def task_completed(self):
                return False

        adapter = ChatGPTResponsesAdapter.__new__(ChatGPTResponsesAdapter)
        adapter.model_name = "fake"
        adapter.max_transport_retries = 3
        adapter._session = Mock()
        response = PartialResponse()
        def partial_request(*_args, **kwargs):
            kwargs["_retry_state"]["attempt"] += 1
            return response
        with patch.object(adapter, "_request", side_effect=partial_request), \
                patch.object(adapter, "_rebuild_session"), \
                patch("agentlab.models.siwc_responses.time.sleep"):
            with tempfile.TemporaryDirectory() as directory:
                trace_path = Path(directory) / "trace.jsonl"
                world = FakeWorld()
                result = LangGraphAgent(adapter, max_steps=1).solve(
                    world, trace_path=trace_path)
                rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(response.closed)
        self.assertEqual(world.executed, [])
        self.assertEqual(result["stop_reason"], "model_adapter_error")
        diagnostic = next(row["adapter_diagnostic"] for row in rows if row["event"] == "agent_error")
        self.assertEqual(diagnostic["phase"], "sse_streaming")
        self.assertEqual(diagnostic["events_seen"], 1)
        self.assertTrue(diagnostic["tool_signal_seen"])
        self.assertTrue(diagnostic["partial_output"])
        self.assertNotIn("secret-detail", json.dumps(diagnostic))
        self.assertEqual(diagnostic["transport_retry_count"], 3)

    def test_usage_limit_is_reported_without_retrying_request(self):
        response = FakeResponse({"error": {"code": "subscription_sharing_usage_limit_exceeded"}})
        adapter = ChatGPTResponsesAdapter.__new__(ChatGPTResponsesAdapter)
        adapter.timeout_seconds = 1
        adapter.connect_timeout_seconds = 2
        adapter._session = Mock()
        with patch("agentlab.models.siwc_responses.access_token", return_value="fake-token"), \
                patch.object(adapter._session, "request", return_value=response) as request:
            with self.assertRaisesRegex(SIWCError, "usage limit reached"):
                adapter._request("POST", "/responses", body={"model": "fake"})
        self.assertEqual(request.call_count, 1)
        self.assertTrue(response.closed)

    def test_other_429_is_also_not_blindly_replayed(self):
        response = FakeResponse({"error": {"code": "slow_down"}})
        adapter = ChatGPTResponsesAdapter.__new__(ChatGPTResponsesAdapter)
        adapter.timeout_seconds = 1
        adapter.connect_timeout_seconds = 2
        adapter.max_transport_retries = 3
        adapter._session = Mock()
        with patch("agentlab.models.siwc_responses.access_token", return_value="fake-token"), \
                patch("agentlab.models.siwc_responses.time.sleep"), \
                patch.object(adapter._session, "request", return_value=response) as request:
            with self.assertRaisesRegex(SIWCError, "quota/rate limit reached"):
                adapter._request("POST", "/responses", body={"model": "fake"})
        self.assertEqual(request.call_count, 4)
        self.assertTrue(response.closed)

    def test_adapter_reuses_one_session_and_closes_it(self):
        response = FakeResponse({})
        response.status_code = 200
        with patch("agentlab.models.siwc_responses.requests.Session") as factory, \
                patch.object(ChatGPTResponsesAdapter, "list_models",
                             return_value=[{"slug": "fake"}]), \
                patch("agentlab.models.siwc_responses.access_token", return_value="fake-token"):
            adapter = ChatGPTResponsesAdapter("fake", connect_timeout_seconds=20)
            factory.return_value.request.return_value = response
            assert adapter._request("POST", "/responses") is response
            assert adapter._request("POST", "/responses") is response
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(factory.return_value.request.call_count, 2)
            self.assertEqual(factory.return_value.request.call_args.kwargs["timeout"], (20, 90))
            adapter.close()
            factory.return_value.close.assert_called_once()

    def test_post_504_retries_and_trace_metadata_reports_status(self):
        adapter = ChatGPTResponsesAdapter.__new__(ChatGPTResponsesAdapter)
        adapter.model_name = "fake"
        adapter.max_transport_retries = 3
        adapter.timeout_seconds = 1
        adapter.connect_timeout_seconds = 2
        adapter._session = Mock()
        adapter._session.request.side_effect = [
            FakeResponse({}, status_code=504), FakeStreamResponse([completed_event()])]
        with patch("agentlab.models.siwc_responses.access_token", return_value="fake-token"), \
                patch("agentlab.models.siwc_responses.time.sleep"):
            message, usage = adapter.generate_tool_calls([], [], tool_choice="required")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "search_api")
        self.assertEqual(usage["input_tokens"], 3)
        self.assertEqual(adapter.last_transport["transport_retry_count"], 1)
        retry = adapter.last_transport["transport_retries"][0]
        self.assertEqual((retry["attempt"], retry["status_code"]), (1, 504))
        self.assertGreaterEqual(retry["backoff_ms"], 1000)

    def test_tls_error_rebuilds_session_and_retries_full_response(self):
        first, second = Mock(), Mock()
        second.request.return_value = FakeStreamResponse([completed_event()])
        adapter = ChatGPTResponsesAdapter.__new__(ChatGPTResponsesAdapter)
        adapter.model_name = "fake"
        adapter.max_transport_retries = 3
        adapter.timeout_seconds = 1
        adapter.connect_timeout_seconds = 2
        adapter._session = first
        first.request.side_effect = requests.exceptions.SSLError("tls detail")
        with patch("agentlab.models.siwc_responses.requests.Session", return_value=second) as factory, \
                patch("agentlab.models.siwc_responses.access_token", return_value="fake-token"), \
                patch("agentlab.models.siwc_responses.time.sleep"):
            message, _usage = adapter.generate_tool_calls([], [])
        factory.assert_called_once()
        first.close.assert_called_once()
        self.assertEqual(first.request.call_count, 1)
        self.assertEqual(second.request.call_count, 1)
        self.assertEqual(message["tool_calls"][0]["id"], "call-1")
        retry = adapter.last_transport["transport_retries"][0]
        self.assertEqual((retry["stage"], retry["error_type"]), ("tls_handshake", "SSLError"))

    def test_sse_disconnect_discards_partial_events_rebuilds_and_recovers(self):
        class InterruptedResponse(FakeStreamResponse):
            def iter_lines(self, **_kwargs):
                yield b'data: {"type":"response.output_item.added","item":{"type":"function_call"}}'
                raise requests.exceptions.ChunkedEncodingError("partial secret")

        first, second = Mock(), Mock()
        interrupted = InterruptedResponse([])
        final = FakeStreamResponse([completed_event()])
        first.request.return_value = interrupted
        second.request.return_value = final
        adapter = ChatGPTResponsesAdapter.__new__(ChatGPTResponsesAdapter)
        adapter.model_name = "fake"
        adapter.max_transport_retries = 3
        adapter.timeout_seconds = 1
        adapter.connect_timeout_seconds = 2
        adapter._session = first
        with patch("agentlab.models.siwc_responses.requests.Session", return_value=second), \
                patch("agentlab.models.siwc_responses.access_token", return_value="fake-token"), \
                patch("agentlab.models.siwc_responses.time.sleep"):
            message, _usage = adapter.generate_tool_calls([], [])
        self.assertTrue(interrupted.closed)
        self.assertTrue(final.closed)
        self.assertEqual(message["tool_calls"][0]["id"], "call-1")
        self.assertEqual(adapter.last_transport["transport_retry_count"], 1)
        retry = adapter.last_transport["transport_retries"][0]
        self.assertEqual((retry["stage"], retry["error_type"]),
                         ("sse_streaming", "ChunkedEncodingError"))

    def test_sse_without_terminal_event_retries_with_fresh_session(self):
        first, second = Mock(), Mock()
        incomplete = FakeStreamResponse([
            b'data: {"type":"response.output_item.added","item":{"type":"function_call"}}'])
        first.request.return_value = incomplete
        second.request.return_value = FakeStreamResponse([completed_event()])
        adapter = ChatGPTResponsesAdapter.__new__(ChatGPTResponsesAdapter)
        adapter.model_name = "fake"
        adapter.max_transport_retries = 3
        adapter.timeout_seconds = 1
        adapter.connect_timeout_seconds = 2
        adapter._session = first
        with patch("agentlab.models.siwc_responses.requests.Session", return_value=second), \
                patch("agentlab.models.siwc_responses.access_token", return_value="fake-token"), \
                patch("agentlab.models.siwc_responses.time.sleep"):
            message, _usage = adapter.generate_tool_calls([], [])
        self.assertTrue(incomplete.closed)
        self.assertEqual(message["tool_calls"][0]["id"], "call-1")
        self.assertEqual(adapter.last_transport["transport_retries"][0]["error_type"], "IncompleteStream")

    def test_retry_after_is_capped_and_recorded(self):
        adapter = ChatGPTResponsesAdapter.__new__(ChatGPTResponsesAdapter)
        adapter.model_name = "fake"
        adapter.max_transport_retries = 3
        adapter.timeout_seconds = 1
        adapter.connect_timeout_seconds = 2
        adapter._session = Mock()
        adapter._session.request.side_effect = [
            FakeResponse({"error": {"code": "slow_down"}}, status_code=429,
                         headers={"Retry-After": "60"}),
            FakeStreamResponse([completed_event()])]
        with patch("agentlab.models.siwc_responses.access_token", return_value="fake-token"), \
                patch("agentlab.models.siwc_responses.time.sleep") as sleep:
            adapter.generate_tool_calls([], [])
        sleep.assert_called_once_with(15.0)
        self.assertEqual(adapter.last_transport["transport_retries"][0]["backoff_ms"], 15000)

    def test_nontransient_statuses_do_not_retry(self):
        for status in (400, 401):
            with self.subTest(status=status):
                adapter = ChatGPTResponsesAdapter.__new__(ChatGPTResponsesAdapter)
                adapter.timeout_seconds = 1
                adapter.connect_timeout_seconds = 2
                adapter.max_transport_retries = 3
                adapter._session = Mock()
                adapter._session.request.return_value = FakeResponse({}, status_code=status)
                with patch("agentlab.models.siwc_responses.access_token", return_value="fake-token"), \
                        patch("agentlab.models.siwc_responses.time.sleep"):
                    with self.assertRaises(SIWCError):
                        adapter._request("POST", "/responses", body={})
                self.assertEqual(adapter._session.request.call_count, 1)

    def test_permission_denied_proxy_fails_without_retry_or_session_rebuild(self):
        adapter = ChatGPTResponsesAdapter.__new__(ChatGPTResponsesAdapter)
        adapter.timeout_seconds = 1
        adapter.connect_timeout_seconds = 2
        adapter.max_transport_retries = 3
        adapter._session = Mock()
        exc = requests.exceptions.ProxyError("local proxy denied")
        exc.__cause__ = PermissionError(10013, "socket denied")
        adapter._session.request.side_effect = exc
        with patch("agentlab.models.siwc_responses.requests.Session") as factory, \
                patch("agentlab.models.siwc_responses.access_token", return_value="fake-token"), \
                patch("agentlab.models.siwc_responses.time.sleep"):
            with self.assertRaises(SIWCTransportError) as caught:
                adapter._request("POST", "/responses", body={})
        self.assertEqual(adapter._session.request.call_count, 1)
        factory.assert_not_called()
        self.assertEqual(caught.exception.diagnostic["phase"], "proxy_or_connect")
        self.assertNotIn("socket denied", str(caught.exception))

    def test_credential_store_uses_unique_temporary_name_and_cleans_it(self):
        with tempfile.TemporaryDirectory() as directory:
            store = siwc_auth.CredentialStore(Path(directory) / "siwc.dat")
            names = []
            original_replace = siwc_auth.os.replace

            def replace(source, target):
                names.append(Path(source).name)
                original_replace(source, target)

            with patch("agentlab.models.siwc_auth._protect", side_effect=lambda data, decrypt=False: data), \
                    patch("agentlab.models.siwc_auth.os.replace", side_effect=replace):
                store.save({"test": True})
            self.assertRegex(names[0], r"siwc\.dat\.tmp\.\d+\.[0-9a-f]+")
            self.assertEqual(list(Path(directory).glob("*.tmp.*")), [])

    def test_refresh_lock_double_check_uses_new_disk_token_without_post(self):
        now = datetime.now(timezone.utc)
        expired = {"scopes": ["chatgpt.tokens.use.direct"], "expires_at": (now - timedelta(minutes=2)).isoformat(),
                   "refresh_token": "old-refresh"}
        refreshed = {"scopes": ["chatgpt.tokens.use.direct"], "expires_at": (now + timedelta(hours=1)).isoformat(),
                     "access_token": "fresh-access"}
        with tempfile.TemporaryDirectory() as directory:
            store = siwc_auth.CredentialStore(Path(directory) / "siwc.dat")
            with patch.object(store, "load", side_effect=[expired, refreshed]), \
                    patch("agentlab.models.siwc_auth._token_request") as request:
                result = siwc_auth.access_token(store)
        self.assertEqual(result, "fresh-access")
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
