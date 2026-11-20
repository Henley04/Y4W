"""OrbitQuant W4A4 manifest parsing and validation.

Reads ``orbitquant_manifest.json`` shipped with WaveCut/YuE2-3B-OrbitQuant-W4A4
and turns it into typed module specs the engine can provision. Parsing is done
without touching any weight bytes, so it is fast and fully CPU-testable.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from ..errors import ManifestError
from ..quant.codebook import codebook_from_manifest


@dataclasses.dataclass(frozen=True)
class ModuleSpec:
    name: str
    in_features: int
    out_features: int
    bias: bool = False
    weight_bits: int = 4
    activation_bits: int = 4
    weight_parameters: int = 0

    @property
    def packed_shape(self) -> tuple[int, int]:
        """Shape of the uint8 packed weight tensor for this layer."""
        if self.in_features % 2:
            raise ManifestError(f"{self.name}: in_features must be even")
        return (self.out_features, self.in_features // 2)


@dataclasses.dataclass
class QuantManifest:
    format_version: int
    config: dict[str, Any]
    modules: list[ModuleSpec]
    quantized_modules: int
    quantized_weight_parameters: int
    kept_precision: list[str] = dataclasses.field(default_factory=list)
    source: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def codebook(self):
        """Resolve the 16-entry codebook described by this manifest."""
        return codebook_from_manifest(self.config)

    @property
    def weight_bits(self) -> int:
        return int(self.config.get("weight_bits", 4))

    @property
    def activation_bits(self) -> int:
        return int(self.config.get("activation_bits", 4))

    @property
    def method(self) -> str:
        return str(self.config.get("quant_method", "orbitquant"))

    def validate(self) -> None:
        if self.method != "orbitquant":
            raise ManifestError(f"Unsupported quant_method {self.method!r}")
        if self.weight_bits != 4 or self.activation_bits != 4:
            raise ManifestError("y4w only supports W4A4 manifests (4/4 bits)")
        if not self.modules:
            raise ManifestError("Manifest has no quantized modules")
        if self.quantized_modules and len(self.modules) != self.quantized_modules:
            raise ManifestError(
                f"Module count {len(self.modules)} != quantized_modules {self.quantized_modules}"
            )

    def summary(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "method": self.method,
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
            "codebook": self.config.get("codebook"),
            "rotation": self.config.get("rotation"),
            "quantized_modules": self.quantized_modules,
            "quantized_weight_parameters": self.quantized_weight_parameters,
            "kept_precision": self.kept_precision,
        }


def load_manifest(path: str | Path) -> QuantManifest:
    """Load and validate an OrbitQuant manifest from JSON.

    Accepts either a path to the JSON file or to a directory containing
    ``orbitquant_manifest.json``.
    """
    path = Path(path).expanduser()
    if path.is_dir():
        path = path / "orbitquant_manifest.json"
    if not path.is_file():
        raise ManifestError(f"Manifest not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Cannot read manifest {path}: {exc}") from exc

    modules: list[ModuleSpec] = []
    for entry in raw.get("modules", []):
        try:
            modules.append(ModuleSpec(
                name=str(entry["name"]),
                in_features=int(entry["in_features"]),
                out_features=int(entry["out_features"]),
                bias=bool(entry.get("bias", False)),
                weight_bits=int(entry.get("weight_bits", 4)),
                activation_bits=int(entry.get("activation_bits", 4)),
                weight_parameters=int(entry.get("weight_parameters", 0)),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"Malformed module entry {entry!r}: {exc}") from exc

    manifest = QuantManifest(
        format_version=int(raw.get("format_version", 1)),
        config=dict(raw.get("config", {})),
        modules=modules,
        quantized_modules=int(raw.get("quantized_modules", len(modules))),
        quantized_weight_parameters=int(raw.get("quantized_weight_parameters", 0)),
        kept_precision=list(raw.get("kept_precision", [])),
        source=dict(raw.get("source", {})),
    )
    manifest.validate()
    return manifest
