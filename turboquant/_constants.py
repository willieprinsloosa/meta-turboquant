"""Shared constants and helpers used across TurboQuant modules."""

import math

import mlx.core as mx


def qjl_scale(head_dim: int) -> float:
    """QJL correction scale factor: sqrt(pi/2) / D."""
    return math.sqrt(math.pi / 2.0) / head_dim


def qjl_scale_array(head_dim: int) -> mx.array:
    """QJL correction scale factor as a pre-computed mx.array."""
    arr = mx.array([qjl_scale(head_dim)], dtype=mx.float32)
    mx.eval(arr)
    return arr
