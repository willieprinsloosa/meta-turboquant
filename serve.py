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
import logging
import os
import re
import sys
import time
import threading
import traceback
import uuid
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step

from turboquant.utils import get_head_dim, make_cache
import turboquant.patch as tq_patch

tq_patch.apply()

# --- Logging ---
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log = logging.getLogger("turboquant")
log.setLevel(logging.DEBUG)

# File handler — detailed logs with timestamps
_log_file = os.path.join(LOG_DIR, f"server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
_fh = logging.FileHandler(_log_file)
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
log.addHandler(_fh)

# Console handler — concise
_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(_ch)

# Crash log — append-only, survives restarts
_crash_file = os.path.join(LOG_DIR, "crashes.log")
_crash_fh = logging.FileHandler(_crash_file)
_crash_fh.setLevel(logging.ERROR)
_crash_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
log.addHandler(_crash_fh)

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


def generate(messages, max_tokens=512, temperature=0.7, stream=False, tools=None, stop=None):
    """Generates a response from chat messages."""
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        template_kwargs["tools"] = tools
    formatted = TOKENIZER.apply_chat_template(messages, **template_kwargs)
    input_ids = mx.array(TOKENIZER.encode(formatted))
    cache = make_cache(N_LAYERS, HEAD_DIM, STRATEGY, BITS, GROUP_SIZE, LEAN)

    tokens = []
    token_buffer = []
    full_decoded = ""
    stop_hit = False

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
            token_buffer.append(tok)
            text = TOKENIZER.decode(token_buffer)
            if text and not text.endswith("\ufffd"):
                # Check stop sequences before yielding
                if stop:
                    full_decoded += text
                    for s in stop:
                        if s in full_decoded:
                            # Yield text up to the stop sequence
                            idx = full_decoded.rfind(s)
                            trimmed = full_decoded[:idx]
                            if trimmed:
                                yield trimmed
                            stop_hit = True
                            break
                    if stop_hit:
                        break
                yield text
                token_buffer = []

    if stop_hit:
        return

    # Flush remaining buffer
    if stream and token_buffer:
        text = TOKENIZER.decode(token_buffer)
        if text:
            yield text

    if not stream:
        decoded = TOKENIZER.decode(tokens)
        # Trim at stop sequence
        if stop:
            for s in stop:
                idx = decoded.find(s)
                if idx != -1:
                    decoded = decoded[:idx]
                    break
        yield decoded


# --- Output post-processing ---

def _clean_react_output(text):
    """Strips ReAct reasoning traces and extracts the final answer.

    1-bit models sometimes emit raw ReAct format:
      Thought: I need to...
      Action: search
      Action Input: ...
      Observation: ...
      Final Answer: The actual response

    This extracts just the final answer, or cleans up partial reasoning.
    """
    if not text:
        return text

    # If there's a "Final Answer:" extract everything after it
    for marker in ("Final Answer:", "Final Answer :", "final answer:"):
        idx = text.lower().find(marker.lower())
        if idx != -1:
            return text[idx + len(marker):].strip()

    # If there's reasoning but no final answer, check for common patterns
    has_react = any(m in text for m in ("Thought:", "Action:", "Observation:", "Action Input:"))
    if not has_react:
        return text

    # Strip Thought/Action/Observation lines and keep the rest
    lines = text.split("\n")
    clean_lines = []
    skip_next = False
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(p) for p in ("Thought:", "Action:", "Action Input:", "Observation:")):
            skip_next = False
            continue
        if stripped:
            clean_lines.append(line)

    cleaned = "\n".join(clean_lines).strip()

    # If everything was reasoning and nothing remains, return the last
    # Observation or the original text as fallback
    if not cleaned:
        for line in reversed(lines):
            if line.strip().startswith("Observation:"):
                return line.strip()[len("Observation:"):].strip()
        return text

    return cleaned


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
        log.debug(f"{self.client_address[0]} {format % args}")

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
        log.info(f"\n>> GET {self.path}")
        if self.path == "/chat":
            self._serve_chat_ui()
        elif self.path == "/v1/models":
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
        elif self.path in ("/health", "/v1"):
            self._send_json({"status": "ok", "model": MODEL_NAME})
        elif self.path == "/":
            # Redirect root to chat UI
            self.send_response(302)
            self.send_header("Location", "/chat")
            self.end_headers()
        else:
            self._send_json({"error": "not found"}, 404)

    def _serve_chat_ui(self):
        """Serves the built-in chat UI from samples/chat.html."""
        chat_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", "chat.html")
        try:
            with open(chat_path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self._send_json({"error": "chat UI not found"}, 404)

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

    def _stream_response(self, messages, max_tokens, temperature, model_id, tools=None, stop=None):
        """Shared streaming logic for chat and responses endpoints.

        Handles BrokenPipeError (#1), consistent chunk IDs (#4),
        proper role delta (#5), and forces non-streaming for tool calls (#6).
        """
        # Force non-streaming when tools are present (#6)
        if tools:
            return self._non_stream_response(messages, max_tokens, temperature, model_id, tools, stop)

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

            for chunk_text in generate(messages, max_tokens, temperature, stream=True, tools=None, stop=stop):
                chunk = make_stream_chunk(stream_id, model_id, content=chunk_text)
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

            final = make_stream_chunk(stream_id, model_id, finish=True)
            self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            log.info("  [client disconnected during streaming]")

    def _non_stream_response(self, messages, max_tokens, temperature, model_id, tools=None, stop=None):
        """Non-streaming response with tool call parsing and ReAct cleanup."""
        full_text = next(generate(messages, max_tokens, temperature, stream=False, tools=tools, stop=stop))
        elapsed = time.perf_counter() - self._request_start

        tool_calls = _parse_tool_calls(full_text) if tools else []

        # Clean up ReAct reasoning traces from 1-bit model output
        if not tool_calls:
            full_text = _clean_react_output(full_text)

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
            log.info(f"  [tool_calls: {names}, {elapsed:.1f}s]")
        else:
            log.info(f"  [{completion_tokens} tokens, {elapsed:.1f}s, {_tok_per_sec(completion_tokens, elapsed)} tok/s]")

    def _handle_chat(self, body):
        """Handles /v1/chat/completions requests."""
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 0.7)
        stream = body.get("stream", False)
        tools = body.get("tools", None)
        stop = body.get("stop", None)
        # Normalize stop to a list
        if isinstance(stop, str):
            stop = [stop]

        if not messages:
            self._send_json({"error": "messages required"}, 400)
            return

        model_id = body.get("model", MODEL_NAME)
        self._request_start = time.perf_counter()

        if stream:
            self._stream_response(messages, max_tokens, temperature, model_id, tools, stop)
        else:
            self._non_stream_response(messages, max_tokens, temperature, model_id, tools, stop)

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
            full_text = _clean_react_output(full_text)
            elapsed = time.perf_counter() - self._request_start
            response = make_responses_response(full_text, model_id)
            self._send_json(response)
            tok_count = len(TOKENIZER.encode(full_text))
            log.info(f"  [responses] [{tok_count} tokens, {elapsed:.1f}s, {_tok_per_sec(tok_count, elapsed)} tok/s]")

    def do_POST(self):
        log.info(f"\n>> POST {self.path}")
        body = self._parse_body()

        # (#2) Handle malformed/oversized requests
        if body is None:
            log.warning(f"  400 — invalid or missing JSON body from {self.client_address[0]}")
            self._send_json({"error": "invalid or missing JSON body"}, 400)
            return

        log.info(f"   stream: {body.get('stream', 'NOT SET')}, model: {body.get('model', 'NOT SET')}")

        if self.path not in ("/v1/chat/completions", "/v1/responses"):
            log.warning(f"  404 — unknown endpoint: {self.path}")
            self._send_json({"error": "not found"}, 404)
            return

        # (#7) Lock with timeout — return 503 if busy
        if not _inference_lock.acquire(timeout=LOCK_TIMEOUT):
            log.warning(f"  503 — server busy, lock timeout after {LOCK_TIMEOUT}s")
            self._send_json({"error": "server busy, try again later"}, 503)
            return
        try:
            if self.path == "/v1/chat/completions":
                self._handle_chat(body)
            else:
                self._handle_responses(body)
        except (BrokenPipeError, ConnectionResetError):
            log.info("  [client disconnected during response]")
        except Exception:
            tb = traceback.format_exc()
            log.error(f"CRASH in {self.path}:\n{tb}")
            try:
                self._send_json({"error": "internal server error"}, 500)
            except Exception:
                pass
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

    log.info(f"Loading model: {MODEL_NAME}")
    MODEL, TOKENIZER = mlx_lm.load(MODEL_NAME)
    HEAD_DIM = get_head_dim(MODEL)
    N_LAYERS = len(MODEL.layers)
    log.info(f"  {N_LAYERS} layers, head_dim={HEAD_DIM}")
    mode = "LEAN" if LEAN else "rotated"
    log.info(f"  Strategy: {STRATEGY.upper()} {BITS}-bit {mode} (group_size={GROUP_SIZE})")

    # Warmup
    log.info("Warming up...")
    cache = make_cache(N_LAYERS, HEAD_DIM, STRATEGY, BITS, GROUP_SIZE, LEAN)
    warmup_ids = mx.array(TOKENIZER.encode("Hello"))
    for tok, _ in generate_step(prompt=warmup_ids, model=MODEL, max_tokens=1, prompt_cache=cache):
        break
    log.info("Ready.")

    log.info(f"Log file: {_log_file}")
    log.info(f"Crash log: {_crash_file}")

    server = ThreadedHTTPServer(("0.0.0.0", args.port), Handler)
    log.info(f"\nServing on http://localhost:{args.port}")
    log.info(f"  POST /v1/chat/completions  — OpenAI-compatible chat")
    log.info(f"  POST /v1/responses         — OpenAI Responses API")
    log.info(f"  GET  /v1/models            — list models")
    log.info(f"  GET  /health               — health check")
    log.info(f"\nOpenClaw config:")
    log.info(f'  Provider URL: http://localhost:{args.port}/v1')
    log.info(f'  Model: {MODEL_NAME}')
    log.info("")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
