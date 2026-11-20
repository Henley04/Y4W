"""Backend interface and metadata types."""
from __future__ import annotations

import dataclasses
from enum import IntEnum
from typing import Any

import torch


class Capability(IntEnum):
    """Qualitative backend capability tier (higher is faster)."""
    REFERENCE = 0   # CPU / portable torch
    PORTABLE_GPU = 1  # Vulkan
    ACCELERATED = 2   # CUDA native kernels
    COMPILED = 3      # TensorRT-RTX engine


@dataclasses.dataclass(frozen=True)
class BackendInfo:
    """Result of probing whether a backend can be used."""
    name: str
    available: bool
    capability: Capability = Capability.REFERENCE
    device: str = "cpu"
    reason: str = ""
    detail: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __bool__(self) -> bool:  # convenient: `if detect_cuda():`
        return self.available


class Backend:
    """Base class every y4w backend implements.

    Subclasses must be safe to instantiate on a machine without their hardware:
    construction only binds metadata; actual context/GPU work happens in
    :meth:`initialize` and fails with ``BackendUnavailable`` when unsupported.
    """

    name = "base"
    capability = Capability.REFERENCE

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    @classmethod
    def probe(cls) -> BackendInfo:
        """Return availability information without side effects."""
        raise NotImplementedError

    def initialize(self) -> None:
        """Acquire device context / load kernels. No-op for CPU."""

    def linear(self, x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
               codebook: torch.Tensor | None = None) -> torch.Tensor:
        """Execute a W4A4 linear layer."""
        raise NotImplementedError

    def linear_int8(self, x: torch.Tensor, w_int8: torch.Tensor,
                    w_scales: torch.Tensor,
                    bias: torch.Tensor | None = None) -> torch.Tensor:
        """Execute a linear layer with baked int8 weights.

        Default implementation uses the portable torch path; CUDA overrides it
        with DP4A (``torch._int_mm``). ``use_dp4a`` is implied by the backend.
        """
        from ..quant.int8 import int8_linear
        return int8_linear(x, w_int8, w_scales, bias, use_dp4a=False)

    @property
    def prefer_baked_int8(self) -> bool:
        """Whether this backend benefits from baked-int8 weights."""
        return False

    def synchronize(self) -> None:
        """Wait for all queued work (used for timing). No-op on CPU."""

    def info(self) -> BackendInfo:
        return self.probe()
