"""Engine configuration dataclass with validation."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class EngineConfig:
    """Runtime configuration for a YuE2 OrbitQuant W4A4 engine.

    Attributes mirror the knobs exposed by the upstream release while staying
    backend-agnostic and safe to construct without a GPU.
    """
    model_dir: str | None = None
    backend: str = "auto"           # auto | cuda | tensorrt_rtx | vulkan | cpu
    profile: str = "fast"           # reference | fused | fast | memory
    device: str | None = None       # explicit torch device override
    kv_cache_dtype: str = "bf16"    # bf16 | int8
    ode_steps: int = 8
    vae_precision: str = "fp16"     # fp32 | fp16 | bf16
    fuse_rms_quant: bool = False
    max_seq_len: int = 24576
    seed: int | None = None
    strict_capability: bool = True  # reject non-SM89/SM120 GPUs in CUDA auto mode
    # Acceleration knobs (all safe to enable; degrade gracefully without GPU).
    optimize: bool = True            # master switch: enable all available accelerations
    compile_model: bool = True       # torch.compile(mode="reduce-overhead") + CUDA graphs
    bake_int8: bool = True           # bake W4A4 -> int8 once for DP4A fast path
    sdpa_attention: bool = True      # use F.scaled_dot_product_attention (FlashAttention)
    int8_kv_cache: bool = False      # store KV cache as int8 (saves ~50% KV memory)
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    PROFILES = ("reference", "fused", "fast", "memory")

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "cuda", "tensorrt_rtx", "vulkan", "cpu"}:
            raise ValueError(f"Unknown backend {self.backend!r}")
        if self.profile not in self.PROFILES:
            raise ValueError(f"Unknown profile {self.profile!r}")
        if self.kv_cache_dtype not in {"bf16", "int8"}:
            raise ValueError("kv_cache_dtype must be 'bf16' or 'int8'")
        if self.vae_precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("vae_precision must be fp32/fp16/bf16")
        if not isinstance(self.ode_steps, int) or self.ode_steps < 1:
            raise ValueError("ode_steps must be a positive integer")
        if self.kv_cache_dtype == "int8" and not self.fuse_rms_quant:
            # Upstream requires RMS fusion for INT8 KV; keep parity but do not
            # hard-fail here so CPU reference mode can still exercise the path.
            self.extra.setdefault("warnings", []).append(
                "INT8 KV cache normally requires fuse_rms_quant=True"
            )

    @property
    def model_path(self) -> Path | None:
        return Path(self.model_dir).expanduser() if self.model_dir else None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
