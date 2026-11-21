"""Model download helpers backed by ModelScope.

The YuE2 OrbitQuant W4A4 weights are distributed from ModelScope at
``syxppp/YuE2-3B-OrbitQuant-W4A4``.  This module keeps the ModelScope SDK as an
*optional* dependency: it is imported lazily so the core engine, server and
CLI keep working on machines (and CI runners) without it installed.
"""
from __future__ import annotations

import os
from typing import Any

# Default mirror of WaveCut/YuE2-3B-OrbitQuant-W4A4 hosted by the y4w author.
DEFAULT_MODEL_ID = "syxppp/YuE2-3B-OrbitQuant-W4A4"
# Allow downstream users / tests to point at a different mirror or local id.
MODEL_ID_ENV = "Y4W_MODEL_ID"

# Files that the y4w runtime actually consumes; restricting the snapshot to
# these keeps the download to ~3 GB (the single safetensors blob + configs)
# instead of pulling the upstream's 560-file source/kernel tree.
NEEDED_FILES = (
    "model.safetensors",
    "orbitquant_manifest.json",
    "config.json",
    "generation_config.json",
    "yue2_generation_config.json",
    "weights_manifest.json",
    "source-lock.json",
    "qwen.tiktoken",
    "SHA256SUMS",
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
)


def get_model_id() -> str:
    """Return the configured ModelScope model id."""
    return os.environ.get(MODEL_ID_ENV, "").strip() or DEFAULT_MODEL_ID


def is_available() -> bool:
    """Whether the ModelScope SDK is importable."""
    try:
        import modelscope  # noqa: F401
        return True
    except ImportError:
        return False


def download_model(
    model_id: str | None = None,
    cache_dir: str | None = None,
    local_dir: str | None = None,
    allow_patterns: list[str] | None = None,
    **kwargs: Any,
) -> str:
    """Download the model snapshot from ModelScope and return its local path.

    Parameters
    ----------
    model_id:
        ModelScope repo id. Defaults to :data:`DEFAULT_MODEL_ID` (overridable
        via ``Y4W_MODEL_ID``).
    cache_dir:
        Optional ModelScope cache directory.
    local_dir:
        If given, download files directly into this directory (handy for
        pointing ``--model-dir`` at a stable path).
    allow_patterns:
        Glob patterns limiting which files are fetched. Defaults to the runtime
        file set in :data:`NEEDED_FILES`.
    **kwargs:
        Extra keyword arguments forwarded to ``modelscope.snapshot_download``.

    Returns
    -------
    str
        Absolute path to the downloaded snapshot directory.
    """
    try:
        from modelscope import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "ModelScope SDK not installed. Install it with "
            "`pip install -e .[hub]` or `pip install modelscope`."
        ) from exc

    rid = model_id or get_model_id()
    patterns = list(allow_patterns) if allow_patterns is not None else list(NEEDED_FILES)
    dl_kwargs: dict[str, Any] = dict(kwargs)
    if cache_dir:
        dl_kwargs["cache_dir"] = cache_dir
    if local_dir:
        dl_kwargs["local_dir"] = local_dir
    if patterns:
        dl_kwargs["allow_patterns"] = patterns

    return snapshot_download(rid, **dl_kwargs)


def model_url(model_id: str | None = None) -> str:
    """Human-facing ModelScope page URL for the model."""
    return f"https://www.modelscope.cn/models/{model_id or get_model_id()}"
