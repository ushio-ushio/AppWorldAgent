"""Verify the production SIWC adapter's Function Calling message shape."""
from agentlab.models.siwc_responses import ChatGPTResponsesAdapter

model = ChatGPTResponsesAdapter("gpt-6-luna")
tools = [{"type": "function", "function": {"name": "fake_lookup",
          "description": "Read a fake test value",
          "parameters": {"type": "object", "properties": {"key": {"type": "string"}},
                         "required": ["key"], "additionalProperties": False}}}]
messages = [{"role": "user", "content": "Call fake_lookup with key test, then report its value."}]
assistant, first_usage = model.generate_tool_calls(messages, tools)
assert len(assistant["tool_calls"]) == 1
call = assistant["tool_calls"][0]
assert call["function"]["name"] == "fake_lookup"
messages.extend([assistant, {"role": "tool", "tool_call_id": call["id"],
                             "content": '{"value":"FAKE_OK"}'}])
answer, second_usage = model.generate_tool_calls(messages, tools)
assert "FAKE_OK" in (answer.get("content") or "")
print({"adapter": "ok", "tool": call["function"]["name"],
       "feedback": "ok", "tokens": first_usage["input_tokens"] + first_usage["output_tokens"] +
       second_usage["input_tokens"] + second_usage["output_tokens"]})
