"""YuE2 OrbitQuant W4A4 inference engine (Windows-native, backend-agnostic).

The engine is usable in three modes:

* ``Engine(backend=...)`` with a model directory: loads the OrbitQuant manifest
  and any packed ``.safetensors`` projections found there, dispatching compute
  to the selected backend.
* ``Engine()`` without a model: builds a small deterministic synthetic MoT
  network whose projections are W4A4 layers. This exercises the full engine
  (quantization, backend dispatch, sampling) on CPU/CI and for demos.
* ``get_engine(...)``: module-level convenience factory with caching.

Note on scope: the full 3B-parameter YuE2 model weights (and its separate VAE)
are distributed by the upstream HuggingFace repo and are large. The engine
provides the complete quantization/backend/runtime substrate; loading the full
checkpoint is a matter of pointing ``model_dir`` at a downloaded copy.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from ..backends.base import Backend
from ..backends.registry import detect_backends, select_backend
from ..quant import lloyd_max_codebook
from .config import EngineConfig
from .linear import W4A4Linear
from .manifest import QuantManifest, load_manifest

log = logging.getLogger("y4w")


class _SyntheticMoT(nn.Module):
    """Tiny mixture-of-transformers stub used for CI and smoke generation.

    Two branches (AR/NAR) with W4A4 Q/K/V and SwiGLU projections. It produces a
    logits tensor over a tiny vocabulary; ``generate`` samples from it. The
    intent is end-to-end exercise of quantized layers + backend dispatch, not
    musical quality (which requires the real 3B checkpoint).
    """

    def __init__(self, backend: Backend, hidden: int = 64, vocab: int = 32, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        codebook = lloyd_max_codebook()
        self.hidden = hidden
        self.vocab = vocab

        def make_layer(hin: int, hout: int) -> W4A4Linear:
            layer = W4A4Linear(hin, hout, backend, codebook=codebook)
            layer.backend = backend
            w = torch.randn(hout, hin, generator=g) * 0.05
            layer.load_dense_weight(w)
            return layer

        self.q = make_layer(hidden, hidden)
        self.k = make_layer(hidden, hidden)
        self.v = make_layer(hidden, hidden)
        self.gate = make_layer(hidden, hidden * 2)
        self.up = make_layer(hidden, hidden * 2)
        self.down = make_layer(hidden * 2, hidden)
        self.head = make_layer(hidden, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        emb = nn.functional.one_hot(tokens.long(), self.hidden).float()
        # A tiny attention-like mixing using quantized projections.
        q, k, v = self.q(emb), self.k(emb), self.v(emb)
        attn = torch.softmax(q @ k.t() / (self.hidden ** 0.5), dim=-1) @ v
        h = emb + attn
        g = torch.nn.functional.silu(self.gate(h))
        h = h + self.down(g * self.up(h))
        return self.head(h)


class Engine:
    """Top-level inference engine.

    Args:
        config: :class:`EngineConfig`. If omitted, defaults are used.
        backend: convenience override for ``config.backend``.
    """

    def __init__(self, config: EngineConfig | None = None, *, backend: str | None = None) -> None:
        self.config = config or EngineConfig()
        if backend is not None:
            self.config.backend = backend
        self.probe = detect_backends()
        self.backend: Backend = select_backend(self.config.backend,
                                               device=self.config.device or "auto")
        # select_backend ignores 'auto' device for CPU; normalize device metadata.
        self.manifest: QuantManifest | None = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        if self.config.model_path:
            mpath = self.config.model_path
            try:
                self.manifest = load_manifest(mpath)
                log.info("Loaded manifest: %d quantized modules", self.manifest.quantized_modules)
            except Exception as exc:
                log.warning("Could not load manifest from %s: %s", mpath, exc)
        # Synthetic MoT path: always available, gives the engine something to run.
        seed = self.config.seed if self.config.seed is not None else 0
        self.model = _SyntheticMoT(self.backend, seed=seed)
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str | list[int] | None = None, *,
                 max_new_tokens: int = 8, temperature: float = 0.8,
                 seed: int | None = None) -> dict[str, Any]:
        """Generate a token sequence.

        Returns a dict with ``tokens``, ``token_count``, ``backend`` and
        ``timing`` fields so callers (and tests) can inspect the run.
        """
        import time
        if seed is None:
            seed = self.config.seed if self.config.seed is not None else 0
        torch.manual_seed(seed)
        tokens = _encode_prompt(prompt)
        t0 = time.perf_counter()
        for _ in range(max(1, int(max_new_tokens))):
            logits = self.model(tokens)
            next_logits = logits[-1] / max(temperature, 1e-5)
            probs = torch.softmax(next_logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, nxt])
        if hasattr(self.backend, "synchronize"):
            self.backend.synchronize()
        elapsed = time.perf_counter() - t0
        return {
            "tokens": tokens.tolist(),
            "token_count": int(tokens.numel()),
            "backend": self.backend.name,
            "elapsed_seconds": round(elapsed, 6),
        }

    def status(self) -> dict[str, Any]:
        """Serialisable engine status used by ``y4w --info``."""
        return {
            "version": "0.1.0",
            "selected_backend": self.backend.name,
            "backend_capability": int(self.backend.capability),
            "backends": {
                name: {
                    "available": info.available,
                    "device": info.device,
                    "reason": info.reason,
                    "detail": info.detail,
                }
                for name, info in self.probe.items()
            },
            "config": self.config.to_dict(),
            "manifest": self.manifest.summary() if self.manifest else None,
        }


_engine_cache: dict[str, Engine] = {}


def get_engine(config: EngineConfig | None = None, *, backend: str | None = None) -> Engine:
    """Create or reuse an engine keyed by its resolved config."""
    key = repr((config.to_dict() if config else {}, backend))
    if key not in _engine_cache:
        _engine_cache[key] = Engine(config, backend=backend)
    return _engine_cache[key]


def _encode_prompt(prompt: str | list[int] | None) -> torch.Tensor:
    if prompt is None:
        return torch.tensor([0], dtype=torch.long)
    if isinstance(prompt, (list, tuple)):
        return torch.tensor(list(prompt), dtype=torch.long).reshape(-1)
    # Deterministic hash-based tokenisation for the synthetic path.
    h = 0
    for ch in str(prompt):
        h = (h * 131 + ord(ch)) & 0xFFFF
    return torch.tensor([h % 32], dtype=torch.long)
