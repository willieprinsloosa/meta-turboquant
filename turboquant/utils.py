"""Shared utilities for serve.py and chat.py."""

from turboquant.cache_v2 import TurboQuantKVCacheV2
from turboquant.cache_v3 import TurboQuantKVCacheV3


def get_head_dim(model):
    """Gets head_dim from a model, handling different architectures."""
    attn = model.layers[0].self_attn
    hd = getattr(attn, 'head_dim', None)
    if hd is None:
        hidden = getattr(model.args, 'hidden_size', getattr(model.args, 'model_dim', 0))
        hd = hidden // attn.n_heads if hidden else 128
    return hd


def make_cache(n_layers, head_dim, strategy="v2", bits=4, group_size=64, lean=False):
    """Creates a fresh TurboQuant cache for all layers."""
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
