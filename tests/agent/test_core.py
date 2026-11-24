import unittest

from agentlab.agent.prompt import ContextManager
from agentlab.runtime.trace import _redact


class TestAgentContext(unittest.TestCase):
    def test_context_starts_with_system_and_original_task(self):
        messages = ContextManager("task-1", "example request").messages
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("example request", messages[1]["content"])
        self.assertIn("complete_task", messages[0]["content"])
        self.assertIn("execute_python", messages[0]["content"])

    def test_trace_redacts_credentials_and_personal_fields(self):
        cleaned = _redact('login(password="secret", email="user@example.com", access_token="token")')
        self.assertNotIn("secret", cleaned)
        self.assertNotIn("user@example.com", cleaned)
        self.assertNotIn('access_token="token"', cleaned)


if __name__ == "__main__":
    unittest.main()
