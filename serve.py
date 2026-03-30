"""TurboQuant Local LLM Server — OpenAI-compatible API with KV-cache compression.

Serves MLX models with TurboQuant V2/V3 KV-cache compression via a local
HTTP endpoint. Compatible with OpenClaw, Continue.dev, or any OpenAI client.

Usage:
    python serve.py                                    # defaults: Llama 3.2 3B, V2 4-bit
    python serve.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit
    python serve.py --model mlx-community/Llama-3.1-8B-Instruct-4bit --bits 3 --strategy v2
    python serve.py --strategy v3 --bits 3 --port 8800

Environment:
    TURBOQUANT_MODEL    — model name (default: mlx-community/Llama-3.2-3B-Instruct-4bit)
    TURBOQUANT_PORT     — port (default: 11434)
    TURBOQUANT_STRATEGY — v2 or v3 (default: v2)
    TURBOQUANT_BITS     — quantization bits (default: 4)
"""

import argparse
import json
import os
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import threading

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step

from turboquant.cache_v2 import TurboQuantKVCacheV2
from turboquant.cache_v3 import TurboQuantKVCacheV3
import turboquant.patch as tq_patch

tq_patch.apply()

# Globals set at startup
MODEL = None
TOKENIZER = None
MODEL_NAME = ""
HEAD_DIM = 0
N_LAYERS = 0
STRATEGY = "v2"
BITS = 4
GROUP_SIZE = 64


def make_cache():
    """Creates a fresh TurboQuant cache for all layers."""
    if STRATEGY == "v3":
        return [
            TurboQuantKVCacheV3(
                head_dim=HEAD_DIM, bits=BITS, seed=42 + i,
            )
            for i in range(N_LAYERS)
        ]
    else:
        return [
            TurboQuantKVCacheV2(
                head_dim=HEAD_DIM, bits=BITS, group_size=GROUP_SIZE,
                use_rotation=True, use_normalization=True, seed=42 + i,
            )
            for i in range(N_LAYERS)
        ]


def _make_sampler(temperature=0.7):
    """Creates a temperature sampler for generate_step."""
    def sampler(logits):
        if temperature <= 0:
            return mx.argmax(logits, axis=-1)
        return mx.random.categorical(logits / temperature)
    return sampler


def generate(messages, max_tokens=512, temperature=0.7, stream=False):
    """Generates a response from chat messages."""
    formatted = TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = mx.array(TOKENIZER.encode(formatted))
    cache = make_cache()

    tokens = []
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
            yield TOKENIZER.decode([tok])

    if not stream:
        yield TOKENIZER.decode(tokens)


def make_response(content, model_name, usage=None):
    """Formats an OpenAI-compatible chat completion response."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def make_stream_chunk(content, model_name, finish=False):
    """Formats an OpenAI-compatible streaming chunk."""
    delta = {} if finish else {"role": "assistant", "content": content}
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
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
            "content": [{
                "type": "output_text",
                "text": content,
            }],
            "status": "completed",
        }],
        "status": "completed",
    }


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle each request in a new thread so health checks don't block."""
    daemon_threads = True
    allow_reuse_address = True


# Lock to serialize MLX inference (not thread-safe)
_inference_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quieter logging
        pass

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
        elif self.path == "/health":
            self._send_json({"status": "ok", "model": MODEL_NAME})
        else:
            self._send_json({"error": "not found"}, 404)

    def _parse_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length))

    def _handle_chat(self, body):
        """Handles /v1/chat/completions requests."""
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 0.7)
        stream = body.get("stream", False)

        if not messages:
            self._send_json({"error": "messages required"}, 400)
            return

        model_id = body.get("model", MODEL_NAME)
        start = time.perf_counter()

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            for chunk_text in generate(messages, max_tokens, temperature, stream=True):
                chunk = make_stream_chunk(chunk_text, model_id)
                line = f"data: {json.dumps(chunk)}\n\n"
                self.wfile.write(line.encode())
                self.wfile.flush()

            final = make_stream_chunk("", model_id, finish=True)
            self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            full_text = ""
            for text in generate(messages, max_tokens, temperature, stream=False):
                full_text = text

            elapsed = time.perf_counter() - start
            response = make_response(full_text, model_id)
            self._send_json(response)
            tok_count = len(TOKENIZER.encode(full_text))
            print(f"  [{tok_count} tokens, {elapsed:.1f}s, {tok_count/elapsed:.0f} tok/s]")

    def _handle_responses(self, body):
        """Handles /v1/responses requests (OpenAI Responses API)."""
        # The Responses API uses 'input' (string or messages array)
        input_data = body.get("input", "")
        max_tokens = body.get("max_output_tokens", body.get("max_tokens", 512))
        temperature = body.get("temperature", 0.7)
        model_id = body.get("model", MODEL_NAME)
        stream = body.get("stream", False)

        # Convert input to messages format
        if isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        elif isinstance(input_data, list):
            # Could be messages array or content blocks
            messages = []
            for item in input_data:
                if isinstance(item, dict) and "role" in item:
                    messages.append(item)
                elif isinstance(item, dict) and "type" in item:
                    # Content block format
                    if item.get("type") == "message":
                        messages.append({
                            "role": item.get("role", "user"),
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

        start = time.perf_counter()

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Responses API streaming uses different event types
            resp_id = f"resp-{uuid.uuid4().hex[:12]}"
            # Send response.created
            self.wfile.write(f"data: {json.dumps({'type': 'response.created', 'response': {'id': resp_id, 'status': 'in_progress'}})}\n\n".encode())
            self.wfile.flush()

            for chunk_text in generate(messages, max_tokens, temperature, stream=True):
                event = {
                    "type": "response.output_text.delta",
                    "delta": chunk_text,
                }
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()

            # Send response.completed
            self.wfile.write(f"data: {json.dumps({'type': 'response.completed', 'response': {'id': resp_id, 'status': 'completed'}})}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            full_text = ""
            for text in generate(messages, max_tokens, temperature, stream=False):
                full_text = text

            elapsed = time.perf_counter() - start
            response = make_responses_response(full_text, model_id)
            self._send_json(response)
            tok_count = len(TOKENIZER.encode(full_text))
            print(f"  [responses] [{tok_count} tokens, {elapsed:.1f}s, {tok_count/elapsed:.0f} tok/s]")

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            body = self._parse_body()
            with _inference_lock:
                self._handle_chat(body)
        elif self.path == "/v1/responses":
            body = self._parse_body()
            with _inference_lock:
                self._handle_responses(body)
        else:
            self._send_json({"error": "not found"}, 404)


def main():
    global MODEL, TOKENIZER, MODEL_NAME, HEAD_DIM, N_LAYERS, STRATEGY, BITS, GROUP_SIZE

    parser = argparse.ArgumentParser(description="TurboQuant Local LLM Server")
    parser.add_argument("--model", default=os.environ.get("TURBOQUANT_MODEL", "mlx-community/Llama-3.2-3B-Instruct-4bit"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TURBOQUANT_PORT", "11434")))
    parser.add_argument("--strategy", default=os.environ.get("TURBOQUANT_STRATEGY", "v2"), choices=["v2", "v3"])
    parser.add_argument("--bits", type=int, default=int(os.environ.get("TURBOQUANT_BITS", "4")), choices=[2, 3, 4])
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    MODEL_NAME = args.model
    STRATEGY = args.strategy
    BITS = args.bits
    GROUP_SIZE = args.group_size

    print(f"Loading model: {MODEL_NAME}")
    MODEL, TOKENIZER = mlx_lm.load(MODEL_NAME)
    HEAD_DIM = MODEL.layers[0].self_attn.head_dim
    N_LAYERS = len(MODEL.layers)
    print(f"  {N_LAYERS} layers, head_dim={HEAD_DIM}")
    print(f"  Strategy: {STRATEGY.upper()} {BITS}-bit (group_size={GROUP_SIZE})")

    # Warmup
    print("Warming up...")
    cache = make_cache()
    warmup_ids = mx.array(TOKENIZER.encode("Hello"))
    for tok, _ in generate_step(prompt=warmup_ids, model=MODEL, max_tokens=1, prompt_cache=cache):
        break
    print("Ready.")

    server = ThreadedHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"\nServing on http://localhost:{args.port}")
    print(f"  POST /v1/chat/completions  — OpenAI-compatible chat")
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
