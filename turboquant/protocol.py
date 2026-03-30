"""Cache protocol defining the interface all TurboQuant cache versions must implement."""

from typing import Protocol, runtime_checkable

import mlx.core as mx


@runtime_checkable
class TurboQuantCache(Protocol):
    """Protocol for TurboQuant KV cache implementations.

    All cache versions (V1, V2, V3) must implement this interface
    to work with the patch.py SDPA dispatch.
    """

    offset: int
    head_dim: int

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Quantizes and stores new KV pairs. Returns data needed for attention."""
        ...

    def make_mask(self, N: int, return_array: bool = False, window_size=None, **kwargs):
        """Creates a causal attention mask."""
        ...

    @property
    def state(self) -> list:
        """Returns the current cache state as a list of tensors."""
        ...

    @property
    def nbytes(self) -> int:
        """Returns the compressed cache size in bytes."""
        ...

    @property
    def nbytes_equivalent_fp16(self) -> int:
        """Returns the equivalent fp16 cache size for compression ratio."""
        ...

    def is_trimmable(self) -> bool:
        """Whether the cache supports trimming."""
        ...

    def trim(self, n: int) -> int:
        """Trims n tokens from the cache. Returns actual number trimmed."""
        ...

    def empty(self) -> bool:
        """Whether the cache has any stored tokens."""
        ...
