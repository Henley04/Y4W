"""W4A4 codebook construction and (de)quantization.

OrbitQuant W4A4 stores each 4-bit weight as an index into a 16-entry Lloyd-Max
codebook. The codebook values live in the checkpoint (float32, codebook v2) and
are applied per quantization group together with a per-row bfloat16 scale.

The reference codebook generator below reproduces a symmetric 4-bit Lloyd-Max
quantizer for a unit normal distribution via the standard centroid/boundary
fixed-point iteration. It is deterministic (no RNG), which makes round-trip
tests reproducible on CPU and on every Windows runner.
"""
from __future__ import annotations

import torch

BITS = 4
LEVELS = 1 << BITS  # 16 codebook entries (codes 0..15)


def lloyd_max_codebook(bits: int = BITS, *, iters: int = 100,
                       eps: float = 1e-12) -> torch.Tensor:
    """Return a sorted Lloyd-Max codebook for ``N(0, 1)``.

    Args:
        bits: number of bits per code (default 4 -> 16 levels).
        iters: maximum fixed-point iterations.
        eps: convergence threshold on centroid movement.

    Returns:
        Float32 tensor of shape ``(1 << bits,)`` with ascending centroids.
    """
    levels = 1 << bits
    # Initialise centroids at evenly spaced quantiles of N(0,1) over [-4, 4].
    import math
    lo, hi = -4.0, 4.0
    centroids = torch.linspace(lo, hi, levels, dtype=torch.float64)
    for _ in range(iters):
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        edges = torch.cat([
            torch.tensor([float("-inf")], dtype=torch.float64),
            boundaries,
            torch.tensor([float("inf")], dtype=torch.float64),
        ])
        new_centroids = torch.empty_like(centroids)
        # E[x | a <= x < b] for a standard normal = (phi(a) - phi(b)) / (Phi(b) - Phi(a))
        phi = lambda t: torch.exp(-0.5 * t * t) / math.sqrt(2 * math.pi)
        Phi = lambda t: 0.5 * (1.0 + torch.erf(t / math.sqrt(2.0)))
        for i in range(levels):
            p = (Phi(edges[i + 1]) - Phi(edges[i])).clamp_min(eps)
            new_centroids[i] = (phi(edges[i]) - phi(edges[i + 1])) / p
        if torch.max(torch.abs(new_centroids - centroids)).item() < eps:
            centroids = new_centroids
            break
        centroids = new_centroids
    # Guard against numerical drift producing non-finite tail centroids.
    centroids = torch.nan_to_num(centroids, nan=0.0, posinf=4.0, neginf=-4.0)
    return centroids.to(torch.float32)


def codebook_from_manifest(config: dict, fallback_bits: int = BITS) -> torch.Tensor:
    """Resolve a codebook from an OrbitQuant manifest ``config`` block.

    The checkpoint may embed explicit ``codebook`` values; when absent we build
    the canonical Lloyd-Max codebook. Always returns a 1-D float32 tensor of
    length 16 (or ``1 << fallback_bits``).
    """
    cb = None
    if isinstance(config, dict):
        cb = config.get("codebook_values") or config.get("codebook")
    if isinstance(cb, (list, tuple)) and len(cb):
        return torch.tensor(cb, dtype=torch.float32).reshape(-1)
    if isinstance(cb, torch.Tensor):
        return cb.detach().float().reshape(-1)
    return lloyd_max_codebook(fallback_bits)


def quantize_to_codes(values: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Quantize float values to integer codes by nearest codebook entry.

    Args:
        values: any shape float tensor.
        codebook: 1-D codebook tensor.

    Returns:
        uint8 tensor of codes with the same shape as ``values``.
    """
    v = values.float()
    cb = codebook.float().reshape(-1)
    # (..., N, 1) -> distances to (levels,) -> nearest index
    dist = (v.unsqueeze(-1) - cb).abs()
    codes = dist.argmin(dim=-1).to(torch.uint8)
    return codes


def dequantize_codes(codes: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Map integer codes back to float codebook values (gather)."""
    cb = codebook.float().reshape(-1)
    return cb[codes.long()]
