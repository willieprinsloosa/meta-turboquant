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
import sys
import time

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step

from turboquant.cache_v2 import TurboQuantKVCacheV2
from turboquant.cache_v3 import TurboQuantKVCacheV3
import turboquant.patch as tq_patch

tq_patch.apply()


def _get_head_dim(model):
    attn = model.layers[0].self_attn
    hd = getattr(attn, 'head_dim', None)
    if hd is None:
        hidden = getattr(model.args, 'hidden_size', getattr(model.args, 'model_dim', 0))
        hd = hidden // attn.n_heads if hidden else 128
    return hd


def make_cache(model, strategy="v2", bits=4, group_size=64, lean=False):
    head_dim = _get_head_dim(model)
    n_layers = len(model.layers)
    if strategy == "v3":
        return [
            TurboQuantKVCacheV3(head_dim=head_dim, bits=bits, seed=42 + i)
            for i in range(n_layers)
        ]
    return [
        TurboQuantKVCacheV2(
            head_dim=head_dim, bits=bits, group_size=group_size,
            use_rotation=not lean, use_normalization=not lean, seed=42 + i,
        )
        for i in range(n_layers)
    ]


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
    head_dim = _get_head_dim(model)
    n_layers = len(model.layers)
    mode = "LEAN" if args.lean else "rotated"
    print(f"  {n_layers} layers, head_dim={head_dim}")
    print(f"  {args.strategy.upper()} {args.bits}-bit {mode}\n")
    print("Type your message. Press Ctrl+C to quit.\n")

    history = []
    if args.system:
        history.append({"role": "system", "content": args.system})

    while True:
        try:
            user_input = input("\033[1;36mYou:\033[0m ")
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input.strip():
            continue

        if user_input.strip().lower() in ("/quit", "/exit", "/q"):
            print("Bye!")
            break

        if user_input.strip().lower() == "/clear":
            history = history[:1] if args.system else []
            print("  (history cleared)\n")
            continue

        if user_input.strip().lower() == "/help":
            print("  /clear  — clear conversation history")
            print("  /quit   — exit")
            print("  /stats  — show cache stats")
            print()
            continue

        history.append({"role": "user", "content": user_input})

        # Build new cache each turn (stateless — simpler and avoids KV drift)
        cache = make_cache(model, args.strategy, args.bits, args.group_size, args.lean)

        formatted = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        input_ids = mx.array(tokenizer.encode(formatted))

        # Generate with streaming output
        print("\033[1;32mAssistant:\033[0m ", end="", flush=True)
        tokens = []
        start = time.perf_counter()

        for token, _ in generate_step(
            prompt=input_ids, model=model,
            max_tokens=args.max_tokens, prompt_cache=cache,
        ):
            tok = token.item() if hasattr(token, "item") else int(token)
            if tok == tokenizer.eos_token_id:
                break
            tokens.append(tok)
            print(tokenizer.decode([tok]), end="", flush=True)

        elapsed = time.perf_counter() - start
        response_text = tokenizer.decode(tokens)
        history.append({"role": "assistant", "content": response_text})

        # Stats
        cache_bytes = sum(c.nbytes for c in cache)
        fp16_bytes = sum(c.nbytes_equivalent_fp16 for c in cache)
        ratio = fp16_bytes / cache_bytes if cache_bytes > 0 else 0
        print(f"\n\033[2m  [{len(tokens)} tokens, {elapsed:.1f}s, {len(tokens)/elapsed:.0f} tok/s, "
              f"cache: {cache_bytes/1024:.0f}KB, {ratio:.1f}x compression]\033[0m\n")


if __name__ == "__main__":
    main()
