"""Backend-dispatched W4A4 linear layer.

``W4A4Linear`` stores packed uint8 weights + per-row scales and delegates the
compute to whichever backend the engine selected. It is deliberately simple:
the heavy model file (modeling_yue2.py) is not vendored here; the engine loads
quantized projections from the checkpoint into these layers by name.
"""
from __future__ import annotations

import torch
from torch import nn

from ..backends.base import Backend
from ..quant import pack_nibbles, quantize_to_codes


class W4A4Linear(nn.Module):
    """A linear layer backed by OrbitQuant packed 4-bit weights.

    Args:
        in_features: input width (must be even).
        out_features: output width.
        backend: backend used for the forward pass.
        codebook: 16-entry float codebook.
        bias: optional FP32 bias.
    """

    def __init__(self, in_features: int, out_features: int, backend: Backend,
                 codebook: torch.Tensor | None = None, bias: bool = False) -> None:
        super().__init__()
        if in_features % 2:
            raise ValueError("in_features must be even for packed W4A4")
        from ..quant import lloyd_max_codebook
        self.in_features = in_features
        self.out_features = out_features
        self.backend = backend
        self.register_buffer("codebook", (codebook if codebook is not None
                                          else lloyd_max_codebook()).float())
        nbytes = in_features // 2
        self.register_buffer("packed", torch.zeros(out_features, nbytes, dtype=torch.uint8))
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False) if bias else None

    @torch.no_grad()
    def load_dense_weight(self, dense: torch.Tensor, group_scale: float | None = None) -> None:
        """Quantize an FP dense ``(out, in)`` weight into this layer's buffers.

        We follow OrbitQuant's per-row scaling: each output row is divided by
        ``amax(row) / max(|codebook|)`` before being mapped to nearest codes, so
        values land inside the codebook's dynamic range. ``group_scale`` can
        override the automatic per-row factor with a fixed one (caller must
        ensure the values fit the codebook), which is useful for test fixtures.
        """
        if dense.shape != (self.out_features, self.in_features):
            raise ValueError(f"Expected {(self.out_features, self.in_features)}, got {tuple(dense.shape)}")
        w = dense.detach().float()
        if group_scale is None:
            cb_max = self.codebook.abs().max().clamp_min(1e-12)
            row_amax = w.abs().amax(dim=1).clamp_min(1e-12)
            scale = (row_amax / cb_max)  # (out,)
        else:
            scale = torch.full((self.out_features,), float(group_scale))
        codes = quantize_to_codes(w / scale.reshape(-1, 1), self.codebook)
        self.packed.copy_(pack_nibbles(codes).cpu())
        self.scales.copy_(scale.cpu())

    @torch.no_grad()
    def load_packed(self, packed: torch.Tensor, scales: torch.Tensor) -> None:
        """Load pre-packed ``(out, in//2)`` uint8 weights and per-row scales."""
        if tuple(packed.shape) != (self.out_features, self.in_features // 2):
            raise ValueError(f"Packed shape mismatch: {tuple(packed.shape)}")
        self.packed.copy_(packed.to(torch.uint8).cpu())
        self.scales.copy_(scales.float().reshape(-1).cpu())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.backend.linear(x, self.packed, self.scales, self.codebook)
        if self.bias is not None:
            y = y + self.bias.to(y.device)
        return y

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, backend={self.backend.name}"
