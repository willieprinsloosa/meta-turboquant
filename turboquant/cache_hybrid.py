"""Sliding Window Hybrid KV-Cache — fp16 sinks + fp16 window + compressed middle.

Keeps the first N_SINK tokens (attention sinks) and the last N_WINDOW recent
tokens at full fp16 precision. Only the middle region is compressed with
TurboQuant V2 or V3.

This exploits two findings:
  1. Attention sinks: the first few tokens receive disproportionate attention
     regardless of content (StreamingLLM, ICLR 2024). Keeping them lossless
     prevents quality degradation.
  2. Recent tokens: the most recent tokens are critical for next-token prediction.
     Full precision here gives the best local quality.
  3. Middle region: older context is accessed less frequently and tolerates
     quantization with minimal quality impact.

Usage:
    from turboquant.cache_hybrid import HybridKVCache
    cache = [HybridKVCache(head_dim=128, bits=4, n_sink=4, n_window=128)
             for _ in range(n_layers)]
"""

import mlx.core as mx

from turboquant.cache import make_causal_mask
from turboquant.cache_v2 import TurboQuantKVCacheV2
from turboquant.rotation import safe_normalize


class HybridKVCache:
    """Sliding Window Hybrid: fp16 sinks + fp16 window + TurboQuant middle.

    Token layout in the cache:
        [SINK tokens (fp16)] [COMPRESSED middle (TurboQuant)] [WINDOW tokens (fp16)]
        |-- n_sink --|       |-- variable length --|          |-- n_window --|

    When total tokens < n_sink + n_window, everything stays fp16 (no compression).
    Once the window fills, the oldest window token is moved to the compressed middle.
    """

    step = 256

    def __init__(
        self,
        head_dim: int = 128,
        bits: int = 4,
        group_size: int = 64,
        n_sink: int = 4,
        n_window: int = 128,
        use_rotation: bool = True,
        use_normalization: bool = True,
        seed: int = 42,
    ):
        self.head_dim = head_dim
        self.bits = bits
        self.group_size = group_size
        self.n_sink = n_sink
        self.n_window = n_window
        self.offset = 0

        # fp16 regions
        self._sink_keys = None    # (B, H, n_sink, D) fp16
        self._sink_values = None
        self._sink_count = 0

        self._window_keys = None  # (B, H, n_window, D) fp16 — ring buffer
        self._window_values = None
        self._window_count = 0
        self._window_start = 0    # ring buffer pointer

        # Compressed middle region
        self._middle = TurboQuantKVCacheV2(
            head_dim=head_dim, bits=bits, group_size=group_size,
            use_rotation=use_rotation, use_normalization=use_normalization,
            seed=seed,
        )

        # Track total for mask generation
        self.use_rotation = use_rotation
        self.use_normalization = use_normalization

    def _ensure_sink(self, B, n_kv_heads, D, dtype):
        if self._sink_keys is None:
            self._sink_keys = mx.zeros((B, n_kv_heads, self.n_sink, D), dtype=dtype)
            self._sink_values = mx.zeros((B, n_kv_heads, self.n_sink, D), dtype=dtype)

    def _ensure_window(self, B, n_kv_heads, D, dtype):
        if self._window_keys is None:
            self._window_keys = mx.zeros((B, n_kv_heads, self.n_window, D), dtype=dtype)
            self._window_values = mx.zeros((B, n_kv_heads, self.n_window, D), dtype=dtype)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Adds new KV pairs with hybrid storage strategy."""
        B, n_kv_heads, num_steps, D = keys.shape
        dtype = keys.dtype

        self._ensure_sink(B, n_kv_heads, D, dtype)
        self._ensure_window(B, n_kv_heads, D, dtype)

        for step in range(num_steps):
            k = keys[:, :, step:step+1, :]
            v = values[:, :, step:step+1, :]

            if self._sink_count < self.n_sink:
                # Fill sink region first
                self._sink_keys[:, :, self._sink_count:self._sink_count+1, :] = k
                self._sink_values[:, :, self._sink_count:self._sink_count+1, :] = v
                self._sink_count += 1
            elif self._window_count < self.n_window:
                # Fill window region
                self._window_keys[:, :, self._window_count:self._window_count+1, :] = k
                self._window_values[:, :, self._window_count:self._window_count+1, :] = v
                self._window_count += 1
            else:
                # Window is full — evict oldest window token to compressed middle
                evict_idx = self._window_start
                evict_k = self._window_keys[:, :, evict_idx:evict_idx+1, :]
                evict_v = self._window_values[:, :, evict_idx:evict_idx+1, :]

                # Compress evicted token into middle
                self._middle.update_and_fetch(evict_k, evict_v)

                # Place new token in the vacated window slot
                self._window_keys[:, :, evict_idx:evict_idx+1, :] = k
                self._window_values[:, :, evict_idx:evict_idx+1, :] = v
                self._window_start = (self._window_start + 1) % self.n_window

            self.offset += 1

        # Return the original keys/values (for V2 attention compatibility)
        return keys, values

    def get_all_keys_values(self):
        """Returns all keys and values in order: sink + middle(dequant) + window.

        For attention computation, we need all KV pairs in the correct order.
        Sink and window are fp16, middle is dequantized from TurboQuant.
        """
        parts_k = []
        parts_v = []

        # 1. Sink tokens (fp16)
        if self._sink_count > 0:
            parts_k.append(self._sink_keys[:, :, :self._sink_count, :])
            parts_v.append(self._sink_values[:, :, :self._sink_count, :])

        # 2. Middle tokens (dequantized from compressed)
        if self._middle.offset > 0:
            mid_keys, mid_values = self._middle.state[:2]  # Get raw quantized state
            # Dequantize using mx.dequantize
            mid_k = mx.dequantize(*self._middle.keys[:3], group_size=self.group_size, bits=self.bits)
            mid_v = mx.dequantize(*self._middle.values[:3], group_size=self.group_size, bits=self.bits)
            T_mid = self._middle.offset
            parts_k.append(mid_k[:, :, :T_mid, :])
            parts_v.append(mid_v[:, :, :T_mid, :])

        # 3. Window tokens (fp16, in ring buffer order)
        if self._window_count > 0:
            if self._window_count < self.n_window or self._window_start == 0:
                # Not wrapped yet — simple slice
                parts_k.append(self._window_keys[:, :, :self._window_count, :])
                parts_v.append(self._window_values[:, :, :self._window_count, :])
            else:
                # Ring buffer wrapped — reorder
                parts_k.append(mx.concatenate([
                    self._window_keys[:, :, self._window_start:, :],
                    self._window_keys[:, :, :self._window_start, :],
                ], axis=2))
                parts_v.append(mx.concatenate([
                    self._window_values[:, :, self._window_start:, :],
                    self._window_values[:, :, :self._window_start, :],
                ], axis=2))

        if not parts_k:
            return None, None

        all_keys = mx.concatenate(parts_k, axis=2) if len(parts_k) > 1 else parts_k[0]
        all_values = mx.concatenate(parts_v, axis=2) if len(parts_v) > 1 else parts_v[0]
        return all_keys, all_values

    def make_mask(self, N, return_array=False, window_size=None, **kwargs):
        return make_causal_mask(self.offset, N, return_array, window_size)

    @property
    def state(self):
        parts = []
        if self._sink_keys is not None:
            parts.extend([
                self._sink_keys[:, :, :self._sink_count, :],
                self._sink_values[:, :, :self._sink_count, :],
            ])
        if self._middle.keys is not None:
            parts.extend(self._middle.state)
        if self._window_keys is not None:
            parts.extend([
                self._window_keys[:, :, :self._window_count, :],
                self._window_values[:, :, :self._window_count, :],
            ])
        return parts

    @state.setter
    def state(self, v):
        raise NotImplementedError("HybridKVCache does not support state restoration.")

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        pass

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def empty(self):
        return self.offset == 0

    @property
    def nbytes(self):
        total = 0
        # Sink (fp16)
        if self._sink_keys is not None:
            total += self._sink_keys[:, :, :self._sink_count, :].nbytes
            total += self._sink_values[:, :, :self._sink_count, :].nbytes
        # Middle (compressed)
        total += self._middle.nbytes
        # Window (fp16)
        if self._window_keys is not None:
            total += self._window_keys[:, :, :self._window_count, :].nbytes
            total += self._window_values[:, :, :self._window_count, :].nbytes
        return total

    @property
    def nbytes_equivalent_fp16(self):
        B = 1
        n_kv_heads = 1
        if self._sink_keys is not None:
            B = self._sink_keys.shape[0]
            n_kv_heads = self._sink_keys.shape[1]
        D = self.head_dim
        T = self.offset
        return B * n_kv_heads * T * D * 2 * 2  # 2 bytes per fp16, K+V

    @property
    def compression_ratio(self):
        fp16 = self.nbytes_equivalent_fp16
        actual = self.nbytes
        if actual == 0:
            return 0
        return fp16 / actual

    @property
    def stats(self):
        """Returns breakdown of token allocation."""
        return {
            "total_tokens": self.offset,
            "sink_tokens": self._sink_count,
            "middle_tokens": self._middle.offset,
            "window_tokens": self._window_count,
            "sink_bytes": (self._sink_keys[:, :, :self._sink_count, :].nbytes * 2) if self._sink_keys is not None else 0,
            "middle_bytes": self._middle.nbytes,
            "window_bytes": (self._window_keys[:, :, :self._window_count, :].nbytes * 2) if self._window_keys is not None else 0,
            "compression_ratio": self.compression_ratio,
        }
