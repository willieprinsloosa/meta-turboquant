"""Hybrid attention — fp16 sinks + compressed middle + fp16 window.

Uses standard mx.matmul for sink and window regions (fp16),
mx.quantized_matmul for the compressed middle, then concatenates
scores and applies softmax over the full sequence.
"""

import mlx.core as mx


def hybrid_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cache,
    scale: float,
    mask=None,
) -> mx.array:
    """SDPA for HybridKVCache — combines fp16 and quantized attention."""
    B, n_q_heads, T_q, D = queries.shape

    # Get all KV pairs in order (sink fp16 + middle dequant + window fp16)
    all_keys, all_values = cache.get_all_keys_values()

    if all_keys is None:
        # Empty cache — shouldn't happen, but handle gracefully
        return mx.zeros_like(queries)

    n_kv_heads = all_keys.shape[1]
    n_repeats = n_q_heads // n_kv_heads
    T_kv = all_keys.shape[2]

    # GQA expand
    if n_repeats > 1:
        all_keys = mx.repeat(all_keys, n_repeats, axis=1)
        all_values = mx.repeat(all_values, n_repeats, axis=1)

    # Standard scaled dot-product attention on the combined KV
    q_scaled = queries * scale
    scores = q_scaled @ all_keys.transpose(0, 1, 3, 2)

    # Apply mask
    if mask is not None:
        if isinstance(mask, str):
            q_indices = mx.arange(T_kv - T_q, T_kv)
            k_indices = mx.arange(T_kv)
            causal = q_indices[:, None] >= k_indices[None]
            scores = mx.where(causal, scores, mx.finfo(scores.dtype).min)
        elif mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores = scores + mask

    weights = mx.softmax(scores, axis=-1, precise=True)
    output = weights @ all_values

    return output
