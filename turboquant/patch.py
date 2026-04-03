"""Monkey-patch for mlx-lm's SDPA dispatch.

Supports TurboQuant V2 (mx.quantized_matmul) and V3 (Lloyd-Max codebook).
V1 (legacy fused Metal kernel) is deprecated — use V2 or V3 instead.
"""

import warnings

import mlx.core as mx
import mlx_lm.models.base as _base

from turboquant.attention_v2 import turboquant_v2_sdpa
from turboquant.attention_v3 import turboquant_v3_sdpa
from turboquant.cache_v2 import TurboQuantKVCacheV2
from turboquant.cache_v3 import TurboQuantKVCacheV3

_original_sdpa = _base.scaled_dot_product_attention
_patched = False


def _patched_sdpa(queries, keys, values, cache, scale, mask, **kwargs):
    # KVTC cache (PCA transform coding)
    from turboquant.cache_kvtc import KVTCCache
    if isinstance(cache, KVTCCache):
        from turboquant.attention_hybrid import hybrid_sdpa
        return hybrid_sdpa(queries, keys, values, cache, scale, mask)
    # Hybrid cache (sliding window)
    from turboquant.cache_hybrid import HybridKVCache
    if isinstance(cache, HybridKVCache):
        from turboquant.attention_hybrid import hybrid_sdpa
        return hybrid_sdpa(queries, keys, values, cache, scale, mask)
    if isinstance(cache, TurboQuantKVCacheV3):
        return turboquant_v3_sdpa(queries, cache, scale, mask)
    if isinstance(cache, TurboQuantKVCacheV2):
        return turboquant_v2_sdpa(queries, keys, values, cache, scale, mask)
    # V1 legacy fallback — lazy import to avoid loading dead code by default
    from turboquant.cache import TurboQuantKVCache
    if isinstance(cache, TurboQuantKVCache):
        warnings.warn(
            "TurboQuantKVCache (V1) is deprecated. Use TurboQuantKVCacheV2 or V3.",
            DeprecationWarning,
            stacklevel=2,
        )
        from turboquant.attention_fused import turboquant_fused_sdpa
        return turboquant_fused_sdpa(queries, cache, scale, mask)
    return _original_sdpa(queries, keys, values, cache, scale, mask, **kwargs)


def apply():
    """Activates the TurboQuant SDPA patch. Idempotent."""
    global _patched
    if _patched:
        return
    _base.scaled_dot_product_attention = _patched_sdpa
    _patched = True


def revert():
    """Removes the patch."""
    global _patched
    _base.scaled_dot_product_attention = _original_sdpa
    _patched = False
