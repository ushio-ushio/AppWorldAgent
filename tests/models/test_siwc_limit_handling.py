import unittest
from unittest.mock import Mock, patch
import requests
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from agentlab.models.siwc_responses import ChatGPTResponsesAdapter, SIWCError, SIWCTransportError
from agentlab.agent.react import LangGraphAgent


class FakeResponse:
    status_code = 429

    def __init__(self, payload):
        self.payload = payload
        self.closed = False

    def json(self):
        return self.payload

    def close(self):
        self.closed = True


class TestSIWCLimitHandling(unittest.TestCase):
    def test_connection_diagnostic_keeps_only_exception_class(self):
        adapter = ChatGPTResponsesAdapter.__new__(ChatGPTResponsesAdapter)
        adapter.timeout_seconds = 1
        adapter.connect_timeout_seconds = 2
        adapter._session = Mock()
        with patch("agentlab.models.siwc_responses.access_token", return_value="fake-token"), \
                patch("agentlab.models.siwc_responses.time.sleep"):
            adapter._session.request.side_effect = requests.exceptions.ProxyError("secret-detail")
            with self.assertRaisesRegex(SIWCError, "ProxyError") as caught:
                adapter._request("POST", "/responses", body={"model": "fake"})
        self.assertEqual(adapter._session.request.call_count, 1)
        self.assertEqual(adapter._session.request.call_args.kwargs["timeout"], (2, 1))
        self.assertNotIn("secret-detail", str(caught.exception))
        self.assertEqual(caught.exception.diagnostic["phase"], "awaiting_headers")
        self.assertTrue(caught.exception.diagnostic["request_may_have_been_received"])

    def test_get_model_list_can_retry_transient_connection(self):
        adapter = ChatGPTResponsesAdapter.__new__(ChatGPTResponsesAdapter)
        adapter.timeout_seconds = 1
        adapter.connect_timeout_seconds = 2
        adapter._session = Mock()
        with patch("agentlab.models.siwc_responses.access_token", return_value="fake-token"), \
                patch("agentlab.models.siwc_responses.time.sleep"):
            adapter._session.request.side_effect = requests.exceptions.ConnectTimeout("secret-detail")
            with self.assertRaises(SIWCTransportError) as caught:
                adapter._request("GET", "/models")
        self.assertEqual(adapter._session.request.call_count, 3)
        self.assertEqual(caught.exception.diagnostic["phase"], "connect")
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
        response = PartialResponse()
        with patch.object(adapter, "_request", return_value=response):
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
        self.assertEqual(diagnostic["phase"], "streaming")
        self.assertEqual(diagnostic["events_seen"], 1)
        self.assertTrue(diagnostic["tool_signal_seen"])
        self.assertTrue(diagnostic["partial_output"])
        self.assertNotIn("secret-detail", json.dumps(diagnostic))

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
        adapter._session = Mock()
        with patch("agentlab.models.siwc_responses.access_token", return_value="fake-token"), \
                patch.object(adapter._session, "request", return_value=response) as request:
            with self.assertRaisesRegex(SIWCError, "quota/rate limit reached"):
                adapter._request("POST", "/responses", body={"model": "fake"})
        self.assertEqual(request.call_count, 1)
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


if __name__ == "__main__":
    unittest.main()
