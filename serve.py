"""TurboQuant Local LLM Server — OpenAI-compatible API with KV-cache compression.

Serves MLX models with TurboQuant V2/V3 KV-cache compression via a local
HTTP endpoint. Compatible with OpenClaw, Continue.dev, or any OpenAI client.

Usage:
    python serve.py                                    # defaults: Llama 3.2 3B, V2 4-bit
    python serve.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit
    python serve.py --model mlx-community/Llama-3.1-8B-Instruct-4bit --bits 3 --strategy v2
    python serve.py --strategy v3 --bits 3 --port 8800

Environment:
    TURBOQUANT_MODEL      — model name (default: mlx-community/Llama-3.2-3B-Instruct-4bit)
    TURBOQUANT_PORT       — port (default: 11434)
    TURBOQUANT_STRATEGY   — v2 or v3 (default: v2)
    TURBOQUANT_BITS       — quantization bits (default: 4)
    TURBOQUANT_LEAN       — set to 1 for LEAN mode (default: 0)
    TURBOQUANT_GROUP_SIZE — group size (default: 64)
"""

import argparse
import json
import os
import re
import time
import threading
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step

from turboquant.utils import get_head_dim, make_cache
import turboquant.patch as tq_patch

tq_patch.apply()

# --- Constants ---
MAX_BODY_SIZE = 10 * 1024 * 1024  # 10 MB
LOCK_TIMEOUT = 30  # seconds

# --- Globals set at startup ---
MODEL = None
TOKENIZER = None
MODEL_NAME = ""
HEAD_DIM = 0
N_LAYERS = 0
STRATEGY = "v2"
BITS = 4
GROUP_SIZE = 64
LEAN = False


def _make_sampler(temperature=0.7):
    """Creates a temperature sampler for generate_step."""
    def sampler(logits):
        if temperature <= 0:
            return mx.argmax(logits, axis=-1)
        return mx.random.categorical(logits / temperature)
    return sampler


def generate(messages, max_tokens=512, temperature=0.7, stream=False, tools=None):
    """Generates a response from chat messages."""
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        template_kwargs["tools"] = tools
    formatted = TOKENIZER.apply_chat_template(messages, **template_kwargs)
    input_ids = mx.array(TOKENIZER.encode(formatted))
    cache = make_cache(N_LAYERS, HEAD_DIM, STRATEGY, BITS, GROUP_SIZE, LEAN)

    tokens = []
    token_buffer = []
    for token, _ in generate_step(
        prompt=input_ids,
        model=MODEL,
        max_tokens=max_tokens,
        prompt_cache=cache,
        sampler=_make_sampler(temperature),
    ):
        tok = token.item() if hasattr(token, "item") else int(token)
        if tok == TOKENIZER.eos_token_id:
            break
        tokens.append(tok)
        if stream:
            # Buffer tokens and only yield when we have valid UTF-8 (#8)
            token_buffer.append(tok)
            text = TOKENIZER.decode(token_buffer)
            if text and not text.endswith("\ufffd"):
                yield text
                token_buffer = []

    # Flush any remaining buffered tokens
    if stream and token_buffer:
        text = TOKENIZER.decode(token_buffer)
        if text:
            yield text

    if not stream:
        yield TOKENIZER.decode(tokens)


# --- Tool call parsing ---

def _parse_tool_calls(text):
    """Parses <tool_call> blocks from model output into OpenAI tool_calls format."""
    tool_calls = []
    pattern = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)
    for match in pattern.finditer(text):
        try:
            call = json.loads(match.group(1))
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": call.get("name", ""),
                    "arguments": json.dumps(call.get("arguments", {})),
                },
            })
        except json.JSONDecodeError:
            continue
    return tool_calls


def _strip_tool_calls(text):
    """Removes <tool_call> blocks from text, returns remaining content."""
    cleaned = re.sub(r'<tool_call>\s*\{.*?\}\s*</tool_call>', '', text, flags=re.DOTALL).strip()
    return cleaned if cleaned else None


# --- Response formatters ---

def make_response(content, model_name, usage=None, tool_calls=None):
    """Formats an OpenAI-compatible chat completion response."""
    message = {"role": "assistant", "content": content}
    finish_reason = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = _strip_tool_calls(content) if content else None
        finish_reason = "tool_calls"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def make_stream_chunk(stream_id, model_name, content=None, finish=False, is_first=False):
    """Formats an OpenAI-compatible streaming chunk.

    Uses consistent stream_id across all chunks (#4).
    First chunk sends role only, subsequent send content only (#5).
    """
    if finish:
        delta = {}
    elif is_first:
        delta = {"role": "assistant", "content": ""}
    else:
        delta = {"content": content}
    return {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": "stop" if finish else None,
        }],
    }


def make_responses_response(content, model_name):
    """Formats an OpenAI Responses API response."""
    resp_id = f"resp-{uuid.uuid4().hex[:12]}"
    return {
        "id": resp_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": model_name,
        "output": [{
            "type": "message",
            "id": f"msg-{uuid.uuid4().hex[:12]}",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content}],
            "status": "completed",
        }],
        "status": "completed",
    }


def _tok_per_sec(tokens, elapsed):
    """Safe tokens/second calculation (#3)."""
    return f"{tokens / max(elapsed, 1e-9):.0f}"


# --- HTTP Server ---

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle each request in a new thread so health checks don't block."""
    daemon_threads = True
    allow_reuse_address = True


# Lock to serialize MLX inference (not thread-safe)
_inference_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"  {self.client_address[0]} {format % args}")

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_cors(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_OPTIONS(self):
        self._send_cors()

    def do_GET(self):
        print(f"\n>> GET {self.path}")
        if self.path == "/v1/models":
            self._send_json({
                "object": "list",
                "data": [{
                    "id": MODEL_NAME,
                    "object": "model",
                    "owned_by": "local",
                    "meta": {
                        "strategy": STRATEGY,
                        "bits": BITS,
                        "head_dim": HEAD_DIM,
                        "layers": N_LAYERS,
                    },
                }],
            })
        elif self.path in ("/health", "/v1", "/"):
            self._send_json({"status": "ok", "model": MODEL_NAME})
        else:
            self._send_json({"error": "not found"}, 404)

    def _parse_body(self):
        """Parse JSON body with error handling (#2) and size limit (#9)."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                return None
            if length > MAX_BODY_SIZE:
                return None
            raw = self.rfile.read(length)
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None

    def _stream_response(self, messages, max_tokens, temperature, model_id, tools=None):
        """Shared streaming logic for chat and responses endpoints.

        Handles BrokenPipeError (#1), consistent chunk IDs (#4),
        proper role delta (#5), and forces non-streaming for tool calls (#6).
        """
        # Force non-streaming when tools are present (#6)
        if tools:
            return self._non_stream_response(messages, max_tokens, temperature, model_id, tools)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        stream_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        try:
            # First chunk: role only (#5)
            first = make_stream_chunk(stream_id, model_id, is_first=True)
            self.wfile.write(f"data: {json.dumps(first)}\n\n".encode())
            self.wfile.flush()

            for chunk_text in generate(messages, max_tokens, temperature, stream=True, tools=None):
                chunk = make_stream_chunk(stream_id, model_id, content=chunk_text)
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

            final = make_stream_chunk(stream_id, model_id, finish=True)
            self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            print("  [client disconnected]")

    def _non_stream_response(self, messages, max_tokens, temperature, model_id, tools=None):
        """Non-streaming response with tool call parsing."""
        full_text = next(generate(messages, max_tokens, temperature, stream=False, tools=tools))
        elapsed = time.perf_counter() - self._request_start

        tool_calls = _parse_tool_calls(full_text) if tools else []

        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            template_kwargs["tools"] = tools
        prompt_tokens = len(TOKENIZER.encode(
            TOKENIZER.apply_chat_template(messages, **template_kwargs)
        ))
        completion_tokens = len(TOKENIZER.encode(full_text))
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        response = make_response(full_text, model_id, usage, tool_calls or None)
        self._send_json(response)
        if tool_calls:
            names = [tc["function"]["name"] for tc in tool_calls]
            print(f"  [tool_calls: {names}, {elapsed:.1f}s]")
        else:
            print(f"  [{completion_tokens} tokens, {elapsed:.1f}s, {_tok_per_sec(completion_tokens, elapsed)} tok/s]")

    def _handle_chat(self, body):
        """Handles /v1/chat/completions requests."""
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 0.7)
        stream = body.get("stream", False)
        tools = body.get("tools", None)

        if not messages:
            self._send_json({"error": "messages required"}, 400)
            return

        model_id = body.get("model", MODEL_NAME)
        self._request_start = time.perf_counter()

        if stream:
            self._stream_response(messages, max_tokens, temperature, model_id, tools)
        else:
            self._non_stream_response(messages, max_tokens, temperature, model_id, tools)

    def _handle_responses(self, body):
        """Handles /v1/responses requests (OpenAI Responses API)."""
        input_data = body.get("input", "")
        max_tokens = body.get("max_output_tokens", body.get("max_tokens", 512))
        temperature = body.get("temperature", 0.7)
        model_id = body.get("model", MODEL_NAME)
        stream = body.get("stream", False)

        def _map_role(role):
            return "system" if role == "developer" else role

        if isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        elif isinstance(input_data, list):
            messages = []
            for item in input_data:
                if isinstance(item, dict) and "role" in item:
                    role = _map_role(item["role"])
                    content = item.get("content", "")
                    if isinstance(content, list):
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict):
                                text_parts.append(block.get("text", block.get("content", "")))
                            else:
                                text_parts.append(str(block))
                        content = "\n".join(text_parts)
                    messages.append({"role": role, "content": content})
                elif isinstance(item, dict) and "type" in item:
                    if item.get("type") == "message":
                        messages.append({
                            "role": _map_role(item.get("role", "user")),
                            "content": item.get("content", ""),
                        })
                    elif item.get("type") == "input_text":
                        messages.append({"role": "user", "content": item.get("text", "")})
                else:
                    messages.append({"role": "user", "content": str(item)})
            if not messages:
                messages = [{"role": "user", "content": str(input_data)}]
        else:
            messages = [{"role": "user", "content": str(input_data)}]

        if not messages:
            self._send_json({"error": "input required"}, 400)
            return

        self._request_start = time.perf_counter()

        if stream:
            self._stream_response(messages, max_tokens, temperature, model_id)
        else:
            full_text = next(generate(messages, max_tokens, temperature, stream=False))
            elapsed = time.perf_counter() - self._request_start
            response = make_responses_response(full_text, model_id)
            self._send_json(response)
            tok_count = len(TOKENIZER.encode(full_text))
            print(f"  [responses] [{tok_count} tokens, {elapsed:.1f}s, {_tok_per_sec(tok_count, elapsed)} tok/s]")

    def do_POST(self):
        print(f"\n>> POST {self.path}")
        body = self._parse_body()

        # (#2) Handle malformed/oversized requests
        if body is None:
            self._send_json({"error": "invalid or missing JSON body"}, 400)
            return

        print(f"   stream: {body.get('stream', 'NOT SET')}, model: {body.get('model', 'NOT SET')}")

        if self.path not in ("/v1/chat/completions", "/v1/responses"):
            self._send_json({"error": "not found"}, 404)
            return

        # (#7) Lock with timeout — return 503 if busy
        if not _inference_lock.acquire(timeout=LOCK_TIMEOUT):
            self._send_json({"error": "server busy, try again later"}, 503)
            return
        try:
            if self.path == "/v1/chat/completions":
                self._handle_chat(body)
            else:
                self._handle_responses(body)
        finally:
            _inference_lock.release()


def main():
    global MODEL, TOKENIZER, MODEL_NAME, HEAD_DIM, N_LAYERS, STRATEGY, BITS, GROUP_SIZE, LEAN

    parser = argparse.ArgumentParser(description="TurboQuant Local LLM Server")
    parser.add_argument("--model", default=os.environ.get("TURBOQUANT_MODEL", "mlx-community/Llama-3.2-3B-Instruct-4bit"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TURBOQUANT_PORT", "11434")))
    parser.add_argument("--strategy", default=os.environ.get("TURBOQUANT_STRATEGY", "v2"), choices=["v2", "v3"])
    parser.add_argument("--bits", type=int, default=int(os.environ.get("TURBOQUANT_BITS", "4")), choices=[2, 3, 4])
    parser.add_argument("--group-size", type=int, default=int(os.environ.get("TURBOQUANT_GROUP_SIZE", "64")))
    parser.add_argument("--lean", action="store_true",
                        default=os.environ.get("TURBOQUANT_LEAN", "0") == "1",
                        help="LEAN mode: no rotation, maximum speed")
    args = parser.parse_args()

    MODEL_NAME = args.model
    STRATEGY = args.strategy
    BITS = args.bits
    GROUP_SIZE = args.group_size
    LEAN = args.lean

    print(f"Loading model: {MODEL_NAME}")
    MODEL, TOKENIZER = mlx_lm.load(MODEL_NAME)
    HEAD_DIM = get_head_dim(MODEL)
    N_LAYERS = len(MODEL.layers)
    print(f"  {N_LAYERS} layers, head_dim={HEAD_DIM}")
    mode = "LEAN" if LEAN else "rotated"
    print(f"  Strategy: {STRATEGY.upper()} {BITS}-bit {mode} (group_size={GROUP_SIZE})")

    # Warmup
    print("Warming up...")
    cache = make_cache(N_LAYERS, HEAD_DIM, STRATEGY, BITS, GROUP_SIZE, LEAN)
    warmup_ids = mx.array(TOKENIZER.encode("Hello"))
    for tok, _ in generate_step(prompt=warmup_ids, model=MODEL, max_tokens=1, prompt_cache=cache):
        break
    print("Ready.")

    server = ThreadedHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"\nServing on http://localhost:{args.port}")
    print(f"  POST /v1/chat/completions  — OpenAI-compatible chat")
    print(f"  POST /v1/responses         — OpenAI Responses API")
    print(f"  GET  /v1/models            — list models")
    print(f"  GET  /health               — health check")
    print(f"\nOpenClaw config:")
    print(f'  Provider URL: http://localhost:{args.port}/v1')
    print(f'  Model: {MODEL_NAME}')
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
