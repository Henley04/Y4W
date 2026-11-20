"""Portable W4A4 quantization primitives.

These are pure-PyTorch reference implementations of the OrbitQuant W4A4 data
layout. They serve two purposes:

1. They are the correctness oracle against which native CUDA / TensorRT-RTX /
   Vulkan kernels are validated (bit-exact reference).
2. They are the execution path used by the CPU backend when no accelerator is
   present, so the engine can run (slowly) anywhere, including CI.
"""
from __future__ import annotations

from .codebook import (
    BITS,
    LEVELS,
    lloyd_max_codebook,
    codebook_from_manifest,
    quantize_to_codes,
    dequantize_codes,
)
from .packed import (
    pack_nibbles,
    unpack_nibbles,
    dequantize_w4a4,
    quantized_matmul,
    w4a4_linear,
)
from .int8 import (
    bake_int8_from_packed,
    bake_int8_from_float,
    quantize_activation_int8,
    int8_linear,
)

__all__ = [
    "BITS", "LEVELS", "lloyd_max_codebook", "codebook_from_manifest",
    "quantize_to_codes", "dequantize_codes", "pack_nibbles", "unpack_nibbles",
    "dequantize_w4a4", "quantized_matmul", "w4a4_linear",
    "bake_int8_from_packed", "bake_int8_from_float",
    "quantize_activation_int8", "int8_linear",
]
