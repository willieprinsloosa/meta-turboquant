"""TurboQuant Chat — Interactive local LLM chat with KV-cache compression.

Simple terminal chat that runs directly on MLX with TurboQuant compression.
No server needed.

Usage:
    python chat.py
    python chat.py --model mlx-community/Llama-3.1-8B-Instruct-4bit
    python chat.py --bits 3
    python chat.py --lean
"""

import argparse
import time

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step

from turboquant.utils import get_head_dim, make_cache
import turboquant.patch as tq_patch

tq_patch.apply()

# Maximum prompt tokens before trimming history (#14)
MAX_PROMPT_TOKENS = 4096


def _trim_history(history, tokenizer, max_tokens, has_system):
    """Trims oldest messages (keeping system prompt) to fit context window."""
    while len(history) > (2 if has_system else 1):
        formatted = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        if len(tokenizer.encode(formatted)) <= max_tokens:
            break
        # Remove oldest non-system message
        start = 1 if has_system else 0
        history.pop(start)
    return history


def main():
    parser = argparse.ArgumentParser(description="TurboQuant Chat")
    parser.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    parser.add_argument("--strategy", default="v2", choices=["v2", "v3"])
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--lean", action="store_true", help="No rotation, max speed")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--system", default=None, help="System prompt")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model, tokenizer = mlx_lm.load(args.model)
    head_dim = get_head_dim(model)
    n_layers = len(model.layers)
    mode = "LEAN" if args.lean else "rotated"
    print(f"  {n_layers} layers, head_dim={head_dim}")
    print(f"  {args.strategy.upper()} {args.bits}-bit {mode}\n")
    print("Type your message. Press Ctrl+C to quit.\n")

    history = []
    has_system = False
    if args.system:
        history.append({"role": "system", "content": args.system})
        has_system = True

    # Cumulative stats for /stats command (#13)
    total_tokens = 0
    total_time = 0.0
    turn_count = 0

    while True:
        try:
            user_input = input("\033[1;36mYou:\033[0m ")
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input.strip():
            continue

        cmd = user_input.strip().lower()
        if cmd in ("/quit", "/exit", "/q"):
            print("Bye!")
            break

        if cmd == "/clear":
            history = history[:1] if has_system else []
            total_tokens = 0
            total_time = 0.0
            turn_count = 0
            print("  (history and stats cleared)\n")
            continue

        if cmd == "/stats":
            avg_tps = total_tokens / max(total_time, 1e-9)
            print(f"  Turns: {turn_count}")
            print(f"  Total tokens: {total_tokens}")
            print(f"  Total time: {total_time:.1f}s")
            print(f"  Average: {avg_tps:.0f} tok/s")
            print(f"  History: {len(history)} messages")
            print()
            continue

        if cmd == "/help":
            print("  /clear  — clear conversation history and stats")
            print("  /stats  — show session statistics")
            print("  /quit   — exit")
            print()
            continue

        history.append({"role": "user", "content": user_input})

        # Trim history if too long (#14)
        history = _trim_history(history, tokenizer, MAX_PROMPT_TOKENS, has_system)

        cache = make_cache(n_layers, head_dim, args.strategy, args.bits, args.group_size, args.lean)

        formatted = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        input_ids = mx.array(tokenizer.encode(formatted))

        # Generate with streaming output, buffering for UTF-8 (#8)
        print("\033[1;32mAssistant:\033[0m ", end="", flush=True)
        tokens = []
        token_buffer = []
        start = time.perf_counter()

        for token, _ in generate_step(
            prompt=input_ids, model=model,
            max_tokens=args.max_tokens, prompt_cache=cache,
        ):
            tok = token.item() if hasattr(token, "item") else int(token)
            if tok == tokenizer.eos_token_id:
                break
            tokens.append(tok)
            token_buffer.append(tok)
            text = tokenizer.decode(token_buffer)
            if text and not text.endswith("\ufffd"):
                print(text, end="", flush=True)
                token_buffer = []

        # Flush remaining buffer
        if token_buffer:
            text = tokenizer.decode(token_buffer)
            if text:
                print(text, end="", flush=True)

        elapsed = time.perf_counter() - start
        response_text = tokenizer.decode(tokens)
        history.append({"role": "assistant", "content": response_text})

        # Update cumulative stats
        total_tokens += len(tokens)
        total_time += elapsed
        turn_count += 1

        # Per-turn stats (#3 — guard division by zero)
        cache_bytes = sum(c.nbytes for c in cache)
        fp16_bytes = sum(c.nbytes_equivalent_fp16 for c in cache)
        ratio = fp16_bytes / cache_bytes if cache_bytes > 0 else 0
        tps = len(tokens) / max(elapsed, 1e-9)
        print(f"\n\033[2m  [{len(tokens)} tokens, {elapsed:.1f}s, {tps:.0f} tok/s, "
              f"cache: {cache_bytes/1024:.0f}KB, {ratio:.1f}x compression]\033[0m\n")


if __name__ == "__main__":
    main()
