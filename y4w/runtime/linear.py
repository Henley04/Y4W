"""Backend-dispatched W4A4 linear layer with optional baked-int8 fast path.

``W4A4Linear`` stores packed uint8 weights + per-row scales (the compact
OrbitQuant source format). On CUDA/TRT-RTX backends that expose DP4A, the
weights are *baked once* at load time into an int8 matrix, removing the ~10x
per-forward dequantization overhead while keeping memory at 1 byte/weight
(half of fp16). On CPU/Vulkan the compact packed form is kept and decoded via
the portable path.
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
        bake_int8: if True (and the backend prefers it), bake weights to int8
            on load for a DP4A fast path. Disabled automatically when the
            backend does not benefit (CPU/Vulkan) to preserve minimal memory.
            Set to False explicitly to always keep the packed form.
    """

    def __init__(self, in_features: int, out_features: int, backend: Backend,
                 codebook: torch.Tensor | None = None, bias: bool = False,
                 bake_int8: bool | None = None) -> None:
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

        # None => follow the backend's preference; explicit bool overrides.
        self.bake_int8 = (backend.prefer_baked_int8 if bake_int8 is None else bool(bake_int8))
        # Baked buffers are registered lazily so state_dict stays compatible.
        self._baked = False
        self.register_buffer("w_int8", torch.zeros(0, dtype=torch.int8), persistent=False)
        self.register_buffer("w_scales", torch.zeros(0, dtype=torch.float32), persistent=False)

    @torch.no_grad()
    def load_dense_weight(self, dense: torch.Tensor, group_scale: float | None = None) -> None:
        """Quantize an FP dense ``(out, in)`` weight into this layer's buffers.

        We follow OrbitQuant's per-row scaling: each output row is divided by
        ``amax(row) / max(|codebook|)`` before being mapped to nearest codes.
        """
        if dense.shape != (self.out_features, self.in_features):
            raise ValueError(f"Expected {(self.out_features, self.in_features)}, got {tuple(dense.shape)}")
        w = dense.detach().float()
        if group_scale is None:
            cb_max = self.codebook.abs().max().clamp_min(1e-12)
            row_amax = w.abs().amax(dim=1).clamp_min(1e-12)
            scale = (row_amax / cb_max)
        else:
            scale = torch.full((self.out_features,), float(group_scale))
        codes = quantize_to_codes(w / scale.reshape(-1, 1), self.codebook)
        self.packed.copy_(pack_nibbles(codes).cpu())
        self.scales.copy_(scale.cpu())
        self._baked = False

    @torch.no_grad()
    def load_packed(self, packed: torch.Tensor, scales: torch.Tensor) -> None:
        """Load pre-packed ``(out, in//2)`` uint8 weights and per-row scales."""
        if tuple(packed.shape) != (self.out_features, self.in_features // 2):
            raise ValueError(f"Packed shape mismatch: {tuple(packed.shape)}")
        self.packed.copy_(packed.to(torch.uint8).cpu())
        self.scales.copy_(scales.float().reshape(-1).cpu())
        self._baked = False

    @torch.no_grad()
    def bake(self) -> None:
        """Bake the packed W4A4 weights into an int8 matrix (fast path)."""
        from ..quant.int8 import bake_int8_from_packed
        wi, ws = bake_int8_from_packed(self.packed, self.scales, self.codebook)
        self.w_int8 = wi
        self.w_scales = ws
        self._baked = True

    @torch.no_grad()
    def to_device(self, device: torch.device | str) -> "W4A4Linear":
        """Move compute tensors to ``device`` and bake if configured."""
        device = torch.device(device)
        if self.bake_int8:
            if not self._baked:
                self.bake()
            self.w_int8 = self.w_int8.to(device)
            self.w_scales = self.w_scales.to(device)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(device)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._baked and self.w_int8.numel() > 0:
            y = self.backend.linear_int8(x, self.w_int8, self.w_scales, self.bias)
        else:
            y = self.backend.linear(x, self.packed, self.scales, self.codebook)
            if self.bias is not None:
                y = y + self.bias.to(y.device)
        return y

    def memory_bytes(self) -> int:
        """Approximate compute-weight memory footprint for this layer."""
        if self._baked and self.w_int8.numel() > 0:
            return self.w_int8.numel() + self.w_scales.numel() * 4
        return self.packed.numel() + self.scales.numel() * 4

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"backend={self.backend.name}, baked={self._baked}")
