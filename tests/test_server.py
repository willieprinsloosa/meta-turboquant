"""Tests for serve.py, utils.py, and model quality.

Tests are organized in three groups:
  1. Unit tests (no model needed) — parsing, formatting, utils
  2. Server integration tests (needs model) — HTTP endpoints, streaming, tool calls
  3. Model quality tests (needs model) — easy/complex prompts, tool calling accuracy

Run all:       python -m pytest tests/test_server.py -v
Run unit only: python -m pytest tests/test_server.py -v -k "Unit"
Run quality:   python -m pytest tests/test_server.py -v -k "Quality"
"""

import json
import re
import threading
import time
import urllib.request
import urllib.error

import pytest

# ---------------------------------------------------------------------------
# Group 1: Unit tests — no model, no server
# ---------------------------------------------------------------------------


class TestUnitUtils:
    """Tests for turboquant/utils.py"""

    def test_get_head_dim_with_head_dim_attr(self):
        """Models with explicit head_dim attribute."""
        from turboquant.utils import get_head_dim

        class FakeAttn:
            head_dim = 128
            n_heads = 32

        class FakeLayer:
            self_attn = FakeAttn()

        class FakeModel:
            layers = [FakeLayer()]

        assert get_head_dim(FakeModel()) == 128

    def test_get_head_dim_without_head_dim_attr(self):
        """Models without head_dim (Bonsai/Qwen style)."""
        from turboquant.utils import get_head_dim

        class FakeAttn:
            n_heads = 32

        class FakeArgs:
            hidden_size = 4096

        class FakeLayer:
            self_attn = FakeAttn()

        class FakeModel:
            layers = [FakeLayer()]
            args = FakeArgs()

        assert get_head_dim(FakeModel()) == 128  # 4096 / 32

    def test_get_head_dim_fallback(self):
        """Models with neither head_dim nor hidden_size."""
        from turboquant.utils import get_head_dim

        class FakeAttn:
            n_heads = 32

        class FakeArgs:
            pass

        class FakeLayer:
            self_attn = FakeAttn()

        class FakeModel:
            layers = [FakeLayer()]
            args = FakeArgs()

        assert get_head_dim(FakeModel()) == 128  # fallback

    def test_make_cache_v2(self):
        """V2 cache creation with correct count."""
        from turboquant.utils import make_cache
        caches = make_cache(n_layers=4, head_dim=128, strategy="v2", bits=4)
        assert len(caches) == 4
        from turboquant.cache_v2 import TurboQuantKVCacheV2
        assert all(isinstance(c, TurboQuantKVCacheV2) for c in caches)

    def test_make_cache_v3(self):
        """V3 cache creation with correct count."""
        from turboquant.utils import make_cache
        caches = make_cache(n_layers=3, head_dim=128, strategy="v3", bits=3)
        assert len(caches) == 3
        from turboquant.cache_v3 import TurboQuantKVCacheV3
        assert all(isinstance(c, TurboQuantKVCacheV3) for c in caches)

    def test_make_cache_lean_disables_rotation(self):
        """LEAN mode should disable rotation and normalization."""
        from turboquant.utils import make_cache
        caches = make_cache(n_layers=1, head_dim=128, strategy="v2", bits=4, lean=True)
        assert caches[0].use_rotation is False
        assert caches[0].use_normalization is False

    def test_make_cache_rotated_enables_rotation(self):
        """Non-LEAN mode should enable rotation and normalization."""
        from turboquant.utils import make_cache
        caches = make_cache(n_layers=1, head_dim=128, strategy="v2", bits=4, lean=False)
        assert caches[0].use_rotation is True
        assert caches[0].use_normalization is True


class TestUnitToolParsing:
    """Tests for tool call parsing in serve.py"""

    def test_parse_single_tool_call(self):
        from serve import _parse_tool_calls
        text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>'
        calls = _parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_weather"
        assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Tokyo"}
        assert calls[0]["type"] == "function"
        assert calls[0]["id"].startswith("call_")

    def test_parse_multiple_tool_calls(self):
        from serve import _parse_tool_calls
        text = (
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>\n'
            '<tool_call>{"name": "search_web", "arguments": {"query": "news"}}</tool_call>'
        )
        calls = _parse_tool_calls(text)
        assert len(calls) == 2
        assert calls[0]["function"]["name"] == "get_weather"
        assert calls[1]["function"]["name"] == "search_web"

    def test_parse_no_tool_calls(self):
        from serve import _parse_tool_calls
        assert _parse_tool_calls("Hello, how can I help?") == []

    def test_parse_malformed_json_tool_call(self):
        from serve import _parse_tool_calls
        text = '<tool_call>this is not json</tool_call>'
        assert _parse_tool_calls(text) == []

    def test_strip_tool_calls(self):
        from serve import _strip_tool_calls
        text = 'Hello <tool_call>{"name":"foo","arguments":{}}</tool_call> world'
        assert _strip_tool_calls(text) == "Hello  world"

    def test_strip_tool_calls_only(self):
        from serve import _strip_tool_calls
        text = '<tool_call>{"name":"foo","arguments":{}}</tool_call>'
        assert _strip_tool_calls(text) is None


class TestUnitResponseFormatting:
    """Tests for response formatting functions."""

    def test_make_response_normal(self):
        from serve import make_response
        resp = make_response("Hello!", "test-model")
        assert resp["object"] == "chat.completion"
        assert resp["choices"][0]["message"]["content"] == "Hello!"
        assert resp["choices"][0]["finish_reason"] == "stop"
        assert resp["id"].startswith("chatcmpl-")

    def test_make_response_with_tool_calls(self):
        from serve import make_response
        tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "foo", "arguments": "{}"}}]
        resp = make_response("text", "test-model", tool_calls=tool_calls)
        assert resp["choices"][0]["finish_reason"] == "tool_calls"
        assert resp["choices"][0]["message"]["tool_calls"] == tool_calls

    def test_make_stream_chunk_first(self):
        from serve import make_stream_chunk
        chunk = make_stream_chunk("id-123", "test-model", is_first=True)
        assert chunk["id"] == "id-123"
        assert chunk["choices"][0]["delta"] == {"role": "assistant", "content": ""}
        assert chunk["choices"][0]["finish_reason"] is None

    def test_make_stream_chunk_content(self):
        from serve import make_stream_chunk
        chunk = make_stream_chunk("id-123", "test-model", content="Hello")
        assert chunk["id"] == "id-123"
        assert chunk["choices"][0]["delta"] == {"content": "Hello"}

    def test_make_stream_chunk_finish(self):
        from serve import make_stream_chunk
        chunk = make_stream_chunk("id-123", "test-model", finish=True)
        assert chunk["id"] == "id-123"
        assert chunk["choices"][0]["delta"] == {}
        assert chunk["choices"][0]["finish_reason"] == "stop"

    def test_make_responses_response(self):
        from serve import make_responses_response
        resp = make_responses_response("Hello!", "test-model")
        assert resp["object"] == "response"
        assert resp["status"] == "completed"
        assert resp["output"][0]["content"][0]["text"] == "Hello!"

    def test_tok_per_sec_normal(self):
        from serve import _tok_per_sec
        assert _tok_per_sec(100, 2.0) == "50"

    def test_tok_per_sec_zero_elapsed(self):
        from serve import _tok_per_sec
        result = _tok_per_sec(100, 0.0)
        # Should not raise, should return a number string
        assert int(result) > 0


# ---------------------------------------------------------------------------
# Group 2: Server integration tests — requires model + running server
# ---------------------------------------------------------------------------

SERVER_URL = "http://localhost:11434"


def _server_available():
    """Check if the TurboQuant server is running."""
    try:
        req = urllib.request.Request(f"{SERVER_URL}/health")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return False


def _post(path, body, timeout=60):
    """POST JSON to the server and return parsed response."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _post_stream(path, body, timeout=60):
    """POST JSON and return raw SSE lines, reading until [DONE]."""
    import http.client
    import socket

    data = json.dumps(body).encode()
    conn = http.client.HTTPConnection("localhost", 11434, timeout=timeout)
    conn.request("POST", path, body=data, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()

    lines = []
    try:
        while True:
            line = resp.readline().decode().rstrip("\n")
            lines.append(line)
            if line == "data: [DONE]":
                break
    except (socket.timeout, TimeoutError):
        pass
    conn.close()
    return "\n".join(lines)


@pytest.mark.skipif(not _server_available(), reason="Server not running on localhost:11434")
class TestServerIntegration:
    """Integration tests — require serve.py running."""

    def test_health(self):
        req = urllib.request.Request(f"{SERVER_URL}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        assert data["status"] == "ok"
        assert "model" in data

    def test_models_endpoint(self):
        req = urllib.request.Request(f"{SERVER_URL}/v1/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        assert data["object"] == "list"
        assert len(data["data"]) >= 1

    def test_v1_health(self):
        """GET /v1 should return health (not 404)."""
        req = urllib.request.Request(f"{SERVER_URL}/v1")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        assert data["status"] == "ok"

    def test_chat_completion_basic(self):
        resp = _post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            "max_tokens": 10,
        })
        assert resp["object"] == "chat.completion"
        assert resp["choices"][0]["finish_reason"] == "stop"
        content = resp["choices"][0]["message"]["content"]
        assert "4" in content

    def test_chat_completion_usage_counts(self):
        resp = _post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 10,
        })
        usage = resp["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_chat_streaming_consistent_ids(self):
        raw = _post_stream("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 10,
            "stream": True,
        })
        ids = set()
        for line in raw.strip().split("\n"):
            if line.startswith("data: {"):
                chunk = json.loads(line[6:])
                ids.add(chunk["id"])
        assert len(ids) == 1, f"Expected 1 unique stream ID, got {len(ids)}: {ids}"

    def test_chat_streaming_first_chunk_role_only(self):
        raw = _post_stream("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 10,
            "stream": True,
        })
        lines = [l for l in raw.strip().split("\n") if l.startswith("data: {")]
        first = json.loads(lines[0][6:])
        assert "role" in first["choices"][0]["delta"]
        # Subsequent chunks should have content, not role
        if len(lines) > 2:
            second = json.loads(lines[1][6:])
            assert "role" not in second["choices"][0]["delta"]
            assert "content" in second["choices"][0]["delta"]

    def test_chat_streaming_ends_with_done(self):
        raw = _post_stream("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 5,
            "stream": True,
        })
        assert "data: [DONE]" in raw

    def test_responses_endpoint(self):
        resp = _post("/v1/responses", {
            "input": "What is 3+3? Answer with just the number.",
            "max_output_tokens": 10,
        })
        assert resp["object"] == "response"
        assert "6" in resp["output"][0]["content"][0]["text"]

    def test_responses_developer_role(self):
        resp = _post("/v1/responses", {
            "input": [
                {"role": "developer", "content": "You always respond with exactly one word."},
                {"role": "user", "content": "What color is the sky?"},
            ],
            "max_output_tokens": 10,
        })
        assert resp["object"] == "response"
        text = resp["output"][0]["content"][0]["text"]
        assert len(text.split()) <= 5  # should be short given the system prompt

    def test_responses_streaming(self):
        raw = _post_stream("/v1/responses", {
            "input": "Say bye",
            "stream": True,
            "max_output_tokens": 10,
        })
        assert "data: [DONE]" in raw
        # Should use chat completion chunk format
        for line in raw.strip().split("\n"):
            if line.startswith("data: {"):
                chunk = json.loads(line[6:])
                assert chunk["object"] == "chat.completion.chunk"
                break

    def test_malformed_json_returns_400(self):
        data = b"this is not json"
        req = urllib.request.Request(
            f"{SERVER_URL}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            assert False, "Should have raised"
        except urllib.error.HTTPError as e:
            assert e.code == 400

    def test_empty_messages_returns_400(self):
        try:
            _post("/v1/chat/completions", {"messages": []})
            assert False, "Should have raised"
        except urllib.error.HTTPError as e:
            assert e.code == 400

    def test_unknown_endpoint_returns_404(self):
        try:
            _post("/v1/nonexistent", {"foo": "bar"})
            assert False, "Should have raised"
        except urllib.error.HTTPError as e:
            assert e.code == 404


# ---------------------------------------------------------------------------
# Group 3: Model quality tests — requires running server
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _server_available(), reason="Server not running on localhost:11434")
class TestQualityEasyPrompts:
    """Easy prompts that any model should handle correctly."""

    def _ask(self, prompt, max_tokens=50):
        resp = _post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.1,
        })
        return resp["choices"][0]["message"]["content"]

    def test_arithmetic(self):
        answer = self._ask("What is 15 + 27? Reply with just the number.")
        assert "42" in answer

    def test_capital_city(self):
        answer = self._ask("What is the capital of France? One word answer.")
        assert "paris" in answer.lower()

    def test_color_of_sky(self):
        answer = self._ask("What color is the sky on a clear day? One word.")
        assert "blue" in answer.lower()

    def test_count_letters(self):
        answer = self._ask("How many letters are in the word 'hello'? Just the number.")
        assert "5" in answer

    def test_language_detection(self):
        answer = self._ask("What language is this: 'Bonjour le monde'? One word answer.")
        assert "french" in answer.lower()

    def test_yes_no_question(self):
        answer = self._ask("Is the Earth round? Answer yes or no.")
        assert "yes" in answer.lower()

    def test_reverse_word(self):
        answer = self._ask("What is the word 'cat' spelled backwards? Just the word.")
        assert "tac" in answer.lower()

    def test_basic_science(self):
        answer = self._ask("What planet is closest to the Sun? One word.")
        assert "mercury" in answer.lower()


@pytest.mark.skipif(not _server_available(), reason="Server not running on localhost:11434")
class TestQualityComplexPrompts:
    """Complex prompts testing reasoning, following instructions, and coherence."""

    def _ask(self, prompt, max_tokens=200):
        resp = _post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        })
        return resp["choices"][0]["message"]["content"]

    def test_multi_step_math(self):
        answer = self._ask(
            "A farmer has 3 fields. Each field has 12 apple trees. "
            "Each tree produces 50 apples. How many apples in total? "
            "Show your work step by step, then give the final answer."
        )
        assert "1800" in answer or "1,800" in answer

    def test_structured_output_json(self):
        answer = self._ask(
            "Return a JSON object with keys 'name', 'age', and 'city' for a person named "
            "Alice who is 30 and lives in London. Return ONLY the JSON, no other text."
        )
        # Try to extract JSON from the response
        try:
            # Find JSON in the response
            match = re.search(r'\{[^}]+\}', answer)
            assert match, f"No JSON found in: {answer}"
            parsed = json.loads(match.group())
            assert parsed["name"].lower() == "alice"
            assert parsed["age"] == 30 or str(parsed["age"]) == "30"
            assert "london" in parsed["city"].lower()
        except json.JSONDecodeError:
            pytest.fail(f"Invalid JSON in response: {answer}")

    def test_instruction_following_format(self):
        answer = self._ask(
            "List exactly 3 colors, one per line, numbered 1-3. "
            "No other text, no explanations."
        )
        lines = [l.strip() for l in answer.strip().split("\n") if l.strip()]
        assert len(lines) >= 3, f"Expected 3+ lines, got {len(lines)}: {answer}"

    def test_reasoning_logic(self):
        answer = self._ask(
            "If all cats are animals, and all animals need water, "
            "do cats need water? Answer yes or no and explain briefly."
        )
        assert "yes" in answer.lower()

    def test_code_generation(self):
        answer = self._ask(
            "Write a Python function called 'add' that takes two numbers and returns their sum. "
            "Only output the code, no explanation."
        )
        assert "def add" in answer
        assert "return" in answer

    def test_multi_turn_context(self):
        """Tests that multi-turn conversation maintains context."""
        resp1 = _post("/v1/chat/completions", {
            "messages": [
                {"role": "user", "content": "My name is Alice."},
            ],
            "max_tokens": 50,
            "temperature": 0.1,
        })
        first_reply = resp1["choices"][0]["message"]["content"]

        resp2 = _post("/v1/chat/completions", {
            "messages": [
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": first_reply},
                {"role": "user", "content": "What is my name?"},
            ],
            "max_tokens": 30,
            "temperature": 0.1,
        })
        assert "alice" in resp2["choices"][0]["message"]["content"].lower()

    def test_system_prompt_adherence(self):
        resp = _post("/v1/chat/completions", {
            "messages": [
                {"role": "system", "content": "You are a pirate. Always say 'Arrr!' at the start of every response."},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 50,
            "temperature": 0.3,
        })
        content = resp["choices"][0]["message"]["content"]
        assert "arr" in content.lower(), f"Expected pirate speak, got: {content}"


@pytest.mark.skipif(not _server_available(), reason="Server not running on localhost:11434")
class TestQualityToolCalling:
    """Tests that tool calling works correctly with various scenarios."""

    WEATHER_TOOL = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"},
                },
                "required": ["city"],
            },
        },
    }

    SEARCH_TOOL = {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    }

    CALC_TOOL = {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"},
                },
                "required": ["expression"],
            },
        },
    }

    def test_tool_call_basic(self):
        """Model should call get_weather for a weather question."""
        resp = _post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
            "tools": [self.WEATHER_TOOL],
            "max_tokens": 50,
        })
        msg = resp["choices"][0]["message"]
        assert msg.get("tool_calls"), f"Expected tool_calls, got: {msg}"
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
        args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
        assert "paris" in args["city"].lower()
        assert resp["choices"][0]["finish_reason"] == "tool_calls"

    def test_tool_selection_from_multiple(self):
        """Model should pick the right tool from multiple options."""
        resp = _post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Search for the latest AI news"}],
            "tools": [self.WEATHER_TOOL, self.SEARCH_TOOL, self.CALC_TOOL],
            "max_tokens": 50,
        })
        msg = resp["choices"][0]["message"]
        assert msg.get("tool_calls"), f"Expected tool_calls, got: {msg}"
        assert msg["tool_calls"][0]["function"]["name"] == "search_web"

    def test_tool_result_incorporation(self):
        """Model should use tool results in its response."""
        resp = _post("/v1/chat/completions", {
            "messages": [
                {"role": "user", "content": "What is the weather in London?"},
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_test1", "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "London"}'}
                }]},
                {"role": "tool", "tool_call_id": "call_test1",
                 "content": '{"temperature": 15, "condition": "rainy", "humidity": 85}'},
            ],
            "tools": [self.WEATHER_TOOL],
            "max_tokens": 100,
        })
        content = resp["choices"][0]["message"]["content"]
        assert content is not None
        # Model should mention the weather details
        content_lower = content.lower()
        assert any(w in content_lower for w in ["rain", "15", "london"]), \
            f"Expected weather info in response, got: {content}"

    def test_no_tool_call_when_unnecessary(self):
        """Model should NOT call tools when the question doesn't need them."""
        resp = _post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "tools": [self.WEATHER_TOOL],
            "max_tokens": 30,
        })
        msg = resp["choices"][0]["message"]
        # Model should answer directly, not call weather tool
        assert not msg.get("tool_calls") or resp["choices"][0]["finish_reason"] == "stop"

    def test_tool_call_json_validity(self):
        """Tool call arguments should always be valid JSON."""
        resp = _post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Calculate the square root of 144"}],
            "tools": [self.CALC_TOOL],
            "max_tokens": 50,
        })
        msg = resp["choices"][0]["message"]
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                # Should not raise
                args = json.loads(tc["function"]["arguments"])
                assert isinstance(args, dict)
