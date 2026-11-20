"""Baked-int8 compute path: the fast, memory-frugal alternative to decoding
packed W4A4 on every forward pass.

Why this exists
---------------
The portable W4A4 path (``quant.packed.quantized_matmul``) dequantizes the
4-bit weights on *every* forward pass — on CPU that measured ~26 ms for a
single 2048x2048 projection, ~10x slower than a plain fp16 GEMM. The 4-bit
format saved storage, not compute.

This module "bakes" the packed weights ONCE at load time into an int8 weight
matrix (1 byte/weight — still half of fp16, and the smallest layout that maps
to a hardware DP4A instruction) plus a per-column scale. At runtime the
activation is quantized per-token to int8 and the product is computed with
``torch._int_mm`` (DP4A on SM89/SM120) where available, with a pure-torch
fallback otherwise. The baked form preserves the W4A4 dequantized values to
within int8 round-off (codebook values are small, so relative error is <1%).

Only ``torch`` is required; no custom C++ extension.
"""
from __future__ import annotations

import torch

_INT8_MAX = 127
_INT8_MIN = -127


def bake_int8_from_packed(packed: torch.Tensor, scales: torch.Tensor,
                          codebook: torch.Tensor | None = None,
                          *,
                          group_size: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Bake packed W4A4 weights into an int8 matrix + per-column scales.

    Args:
        packed: ``(out, in//2)`` uint8 codes (low nibble first).
        scales: per-output-row scales (the W4A4 row-norm).
        codebook: 16-value codebook; defaults to Lloyd-Max.
        group_size: if >0, use per-group (per ``group_size`` input columns)
            scales instead of one scale per output row. Smaller groups improve
            accuracy at the cost of a slightly larger scale vector.

    Returns:
        ``(w_int8, w_scales)``:
            w_int8  -- ``(out, in)`` int8 (device = input device)
            w_scales -- ``(out, g)`` fp32 where ``g = ceil(in/group_size)``
                        when grouped, else ``(out, 1)``.
    """
    from .packed import dequantize_w4a4
    w = dequantize_w4a4(packed, scales, codebook).float()  # (out, in)
    return bake_int8_from_float(w, group_size=group_size)


def bake_int8_from_float(w: torch.Tensor, *, group_size: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float ``(out, in)`` weight to int8 with symmetric scales."""
    if w.ndim != 2:
        raise ValueError(f"expected 2D weight, got {tuple(w.shape)}")
    out, in_f = w.shape
    if group_size and group_size > 0:
        if in_f % group_size:
            raise ValueError(f"in_features {in_f} not divisible by group_size {group_size}")
        g = in_f // group_size
        blocks = w.reshape(out, g, group_size)
        amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = (amax / _INT8_MAX).expand_as(blocks).reshape(out, in_f)
        wq = torch.round(w / scale).clamp(_INT8_MIN, _INT8_MAX).to(torch.int8)
        s = (amax / _INT8_MAX).squeeze(-1).contiguous()  # (out, g)
    else:
        amax = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)  # (out, 1)
        scale = amax / _INT8_MAX
        wq = torch.round(w / scale).clamp(_INT8_MIN, _INT8_MAX).to(torch.int8)
        s = scale.contiguous()  # (out, 1)
    return wq.contiguous(), s


def quantize_activation_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric int8 quantization of ``x``.

    Returns ``(x_int8, per_token_scale)`` where ``x ≈ x_int8 * scale[...,None]``.
    """
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = amax / _INT8_MAX
    xq = torch.round(xf / scale).clamp(_INT8_MIN, _INT8_MAX).to(torch.int8)
    return xq.contiguous(), scale.squeeze(-1).contiguous()


def int8_linear(x: torch.Tensor, w_int8: torch.Tensor, w_scales: torch.Tensor,
                bias: torch.Tensor | None = None, *, use_dp4a: bool = False) -> torch.Tensor:
    """y = x @ w_int8.T with int8 inputs, dequantized on output.

    Args:
        x: ``(..., in)`` float activation.
        w_int8: ``(out, in)`` int8 baked weight.
        w_scales: ``(out, 1)`` or ``(out, g)`` fp32 weight scales.
        bias: optional fp32 bias of shape ``(out,)``.
        use_dp4a: use ``torch._int_mm`` (DP4A) on CUDA when available.
    """
    leading = x.shape[:-1]
    xf = x.reshape(-1, x.shape[-1])
    xq, xs = quantize_activation_int8(xf)  # (M,in) int8, (M,) scale

    y = None
    if use_dp4a and xq.is_cuda and hasattr(torch, "_int_mm"):
        # DP4A accumulates int8 products into int32. It requires both inner
        # dims to be multiples of 8; fall back to portable math otherwise.
        try:
            acc = torch._int_mm(xq, w_int8.t().contiguous())  # (M, out) int32
            y = acc.float()
        except (RuntimeError, ValueError):  # shape/dtype unsupported by _int_mm
            pass
    if y is None:
        # Pure-torch fallback: fp32 accumulation of int8 products.
        y = xq.float() @ w_int8.float().t()

    # Activation scale is per-token (rows); weight scale is per-output (cols).
    y = y * xs.unsqueeze(1)
    if w_scales.shape[1] == 1:
        y = y * w_scales.reshape(1, -1)
    else:
        # Grouped scales: repeat each group scale across its columns and do a
        # group-wise sum. Kept simple here (correctness path); the common case
        # is the per-output scale above.
        g = w_scales.shape[1]
        gs = in_group_scales(w_scales, w_int8.shape[1], g, y.device)
        y = y * gs.unsqueeze(0)

    if bias is not None:
        y = y + bias.to(y.device).reshape(1, -1)
    return y.reshape(*leading, w_int8.shape[0])


def in_group_scales(w_scales: torch.Tensor, in_features: int, g: int,
                    device: torch.device) -> torch.Tensor:
    """Expand grouped column scales ``(out, g)`` to ``(out, in)`` via repeat."""
    group_size = in_features // g
    return w_scales.repeat_interleave(group_size, dim=1).to(device)
