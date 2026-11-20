"""Manifest parsing against the real upstream orbitquant_manifest.json shape."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from y4w.errors import ManifestError
from y4w.runtime.manifest import ModuleSpec, load_manifest

REF = Path(__file__).resolve().parent / "fixtures" / "orbitquant_manifest.json"
if not REF.exists():  # fallback for source trees that keep reference data in _ref
    REF = Path(__file__).resolve().parents[2] / "_ref" / "orbitquant_manifest.json"


def _write(tmp_path, data):
    p = tmp_path / "orbitquant_manifest.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _minimal(**overrides):
    data = {
        "format_version": 1,
        "config": {"quant_method": "orbitquant", "weight_bits": 4,
                   "activation_bits": 4, "codebook": "lloyd_max", "rotation": "rpbh"},
        "modules": [{"name": "layer.0.q", "in_features": 2048, "out_features": 2048,
                     "weight_bits": 4, "activation_bits": 4, "weight_parameters": 4194304}],
        "quantized_modules": 1,
        "quantized_weight_parameters": 4194304,
        "kept_precision": ["lm_head"],
    }
    data.update(overrides)
    return data


def test_module_spec_packed_shape():
    spec = ModuleSpec("x", in_features=2048, out_features=1024)
    assert spec.packed_shape == (1024, 1024)


def test_module_spec_odd_in_features_raises():
    spec = ModuleSpec("x", in_features=7, out_features=8)
    with pytest.raises(ManifestError):
        _ = spec.packed_shape


def test_load_real_manifest(tmp_path):
    if not REF.exists():
        pytest.skip("reference manifest not bundled")
    m = load_manifest(REF)
    assert m.method == "orbitquant"
    assert m.weight_bits == 4 and m.activation_bits == 4
    assert m.quantized_modules == 392
    assert m.quantized_weight_parameters == 2_818_572_288
    assert len(m.modules) == 392
    assert len(m.codebook) == 16
    assert "model.layers.0.self_attn.q_proj" in {mod.name for mod in m.modules}


def test_load_manifest_from_directory(tmp_path):
    _write(tmp_path, _minimal())
    m = load_manifest(tmp_path)
    assert m.quantized_modules == 1
    s = m.summary()
    assert s["quantized_modules"] == 1 and s["method"] == "orbitquant"


def test_reject_wrong_method(tmp_path):
    p = _write(tmp_path, _minimal(config={"quant_method": "gptq", "weight_bits": 4, "activation_bits": 4}))
    with pytest.raises(ManifestError):
        load_manifest(p)


def test_reject_wrong_bitwidth(tmp_path):
    p = _write(tmp_path, _minimal(config={"quant_method": "orbitquant", "weight_bits": 8, "activation_bits": 8}))
    with pytest.raises(ManifestError):
        load_manifest(p)


def test_reject_module_count_mismatch(tmp_path):
    p = _write(tmp_path, _minimal(quantized_modules=5))
    with pytest.raises(ManifestError):
        load_manifest(p)


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "nope.json")


def test_codebook_from_manifest_explicit_values(tmp_path):
    cb = [float(i) - 7.5 for i in range(16)]
    p = _write(tmp_path, _minimal(config={"quant_method": "orbitquant", "weight_bits": 4,
                                          "activation_bits": 4, "codebook_values": cb}))
    m = load_manifest(p)
    assert len(m.codebook) == 16
    assert abs(float(m.codebook[0]) - (-7.5)) < 1e-6
