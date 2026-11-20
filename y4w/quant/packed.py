"""Packed 4-bit weight layout and portable W4A4 GEMM reference.

OrbitQuant packs two 4-bit codes into one ``uint8`` byte (low nibble first),
matching the ``weight_pack_dtype=uint8`` field of the manifest and the DP4A
byte-pair lookup kernels shipped with the model. Dequantization applies a
per-row (bfloat16) scale to each code's codebook value.

Everything here is plain PyTorch so it runs on CPU (CI) and on CUDA/Torch
without any custom compiled extension. Native backends may override the hot
path but must stay numerically close to ``quantized_matmul``.
"""
from __future__ import annotations

import torch
from .codebook import lloyd_max_codebook

_NIBBLE_MASK = 0x0F
_SIGN_BIT = 0x08


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """Pack integer codes (0..15) two-per-byte into uint8.

    Args:
        codes: integer tensor whose last dimension is even.

    Returns:
        uint8 tensor whose last dimension is ``codes.shape[-1] // 2``.
        The first code occupies the low nibble, the second the high nibble.
    """
    c = codes.to(torch.int64)
    if c.shape[-1] % 2 != 0:
        raise ValueError(f"pack_nibbles needs an even last dimension, got {c.shape[-1]}")
    low = c[..., 0::2] & _NIBBLE_MASK
    high = c[..., 1::2] & _NIBBLE_MASK
    return (low | (high << 4)).to(torch.uint8)


def unpack_nibbles(packed: torch.Tensor, out_features: int | None = None) -> torch.Tensor:
    """Unpack uint8 bytes into one code per nibble (0..15).

    Args:
        packed: uint8 tensor, last dim = number of bytes.
        out_features: optional expected last-dim size (2 * bytes) for validation.

    Returns:
        uint8 codes with last dimension ``2 * packed.shape[-1]``.
    """
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed weights must be uint8, got {packed.dtype}")
    low = packed & _NIBBLE_MASK
    high = (packed >> 4) & _NIBBLE_MASK
    # Interleave low/high so ordering matches the original (low nibble first).
    codes = torch.stack((low, high), dim=-1).flatten(start_dim=-2).to(torch.uint8)
    if out_features is not None and codes.shape[-1] != out_features:
        from ..errors import WeightFormatError
        raise WeightFormatError(
            f"unpacked {codes.shape[-1]} codes but expected {out_features} output features"
        )
    return codes


def dequantize_w4a4(packed: torch.Tensor, scales: torch.Tensor,
                    codebook: torch.Tensor | None = None) -> torch.Tensor:
    """Fully dequantize a packed W4A4 weight tensor to float.

    Args:
        packed: uint8 ``(out_features, in_features // 2)``.
        scales: per-row scales ``(out_features, 1)`` or ``(out_features,)``.
        codebook: 16-entry codebook; defaults to the canonical Lloyd-Max one.

    Returns:
        float32 tensor ``(out_features, in_features)``.
    """
    if codebook is None:
        codebook = lloyd_max_codebook()
    codes = unpack_nibbles(packed)
    values = codebook.float()[codes.long()]
    s = scales.float().reshape(-1, 1)
    if s.shape[0] != values.shape[0]:
        from ..errors import WeightFormatError
        raise WeightFormatError(
            f"scale rows {s.shape[0]} != weight rows {values.shape[0]}"
        )
    return values * s


def quantized_matmul(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
                     codebook: torch.Tensor | None = None,
                     activation_bits: int = 4) -> torch.Tensor:
    """Portable W4A4 linear: x @ dequantized(packed weights).

    The activation is quantized to ``activation_bits`` via a per-token RMS
    scale to mirror the runtime's activation quantizer; disable by passing
    ``activation_bits=0`` (useful as a lossless oracle).

    Args:
        x: ``(..., in_features)`` float tensor.
        packed: ``(out_features, in_features // 2)`` uint8 weights.
        scales: per-output-row scales.
        codebook: weight codebook (16 values).
        activation_bits: 0 to skip activation quantization.

    Returns:
        ``(..., out_features)`` float tensor.
    """
    if codebook is None:
        codebook = lloyd_max_codebook()
    w = dequantize_w4a4(packed, scales, codebook)  # (N, K)
    xf = x.float()
    if activation_bits:
        # Per-token symmetric scale using RMSNorm-style reduction.
        rms = xf.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        levels = 1 << activation_bits
        qscale = rms / (levels - 1)
        xq = torch.round(xf / qscale).clamp(-(levels // 2), levels // 2 - 1)
        xf = xq * qscale
    return xf @ w.t()


def w4a4_linear(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
                codebook: torch.Tensor | None = None) -> torch.Tensor:
    """Alias of :func:`quantized_matmul` with activation quantization enabled."""
    return quantized_matmul(x, packed, scales, codebook, activation_bits=4)
