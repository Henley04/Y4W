"""Portable CPU / Torch backend (always available reference path)."""
from __future__ import annotations

import torch

from ..errors import BackendUnavailable
from ..quant import quantized_matmul
from .base import Backend, BackendInfo, Capability


class CPUBackend(Backend):
    """Fallback backend using pure PyTorch W4A4 dequantization.

    This is the path used on GitHub-hosted Windows runners (no GPU) and is the
    numerical reference all accelerated backends are validated against.
    """

    name = "cpu"
    capability = Capability.REFERENCE

    @classmethod
    def probe(cls) -> BackendInfo:
        return BackendInfo(
            name=cls.name,
            available=True,
            capability=cls.capability,
            device="cpu",
            reason="portable PyTorch reference path",
            detail={"torch": torch.__version__},
        )

    def initialize(self) -> None:
        # Nothing to acquire; guard against a deliberately disabled CPU path.
        if torch is None:  # pragma: no cover
            raise BackendUnavailable("PyTorch is not importable")

    def linear(self, x, packed, scales, codebook=None):
        # Ensure all operands are on CPU for the reference path.
        return quantized_matmul(
            x.cpu() if isinstance(x, torch.Tensor) else x,
            packed.cpu() if isinstance(packed, torch.Tensor) else packed,
            scales.cpu() if isinstance(scales, torch.Tensor) else scales,
            codebook=codebook,
            activation_bits=4,
        )
