"""Exception hierarchy for y4w."""
from __future__ import annotations


class Y4WError(Exception):
    """Base class for all y4w errors."""


class BackendUnavailable(Y4WError, RuntimeError):
    """A requested backend cannot be used on this machine."""


class BackendInitError(Y4WError, RuntimeError):
    """A backend was detected but failed to initialize."""


class UnsupportedGPU(Y4WError, RuntimeError):
    """GPU compute capability is outside the supported set (SM89/SM120)."""

    def __init__(self, capability: tuple[int, int]):
        self.capability = capability
        super().__init__(
            f"Unsupported GPU compute capability {capability[0]}.{capability[1]}; "
            "this runtime targets SM89 (Ada) and SM120 (Blackwell). "
            "Pass backend='cpu' for a portable reference path."
        )


class ManifestError(Y4WError, ValueError):
    """Malformed or incompatible OrbitQuant quantization manifest."""


class WeightFormatError(Y4WError, ValueError):
    """Packed weight tensor has an unexpected shape / dtype / size."""
