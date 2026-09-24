# ruff: noqa: E501
from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .config import (
    CONFIDENCE_THRESHOLD,
    MAX_EXEMPLARS,
    MAX_SIDE,
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    MODEL_SHA256,
    MODEL_SIZE_BYTES,
    SHORT_SIDE,
    TARGET_BOX_PX,
    TOKENIZER_LICENSE,
    TOKENIZER_MODEL_ID,
    TOKENIZER_REVISION,
    UPSTREAM_COMMIT,
    UPSTREAM_LICENSE,
    UPSTREAM_REPOSITORY,
)

_RUNTIME_PACKAGES = ("torch", "torchvision", "transformers", "safetensors", "scipy", "numpy", "pillow", "huggingface-hub")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def build_provenance(
    *,
    pipeline: Any | None = None,
    checkpoint_path: str | Path | None = None,
    include_runtime: bool = True,
) -> dict[str, Any]:
    checkpoint_source = None
    manifest_verified = False
    weight_sha256 = MODEL_SHA256
    weight_size = MODEL_SIZE_BYTES
    device_str = None
    resolved_checkpoint_path = None
    adapter: dict[str, Any] | None = None

    if pipeline is not None:
        if getattr(pipeline, "checkpoint_path", None) is not None:
            resolved_checkpoint_path = str(pipeline.checkpoint_path)
        checkpoint_source = getattr(pipeline, "checkpoint_source", None)
        manifest_verified = bool(getattr(pipeline, "manifest_verified", False))
        if getattr(pipeline, "weight_sha256", None):
            weight_sha256 = pipeline.weight_sha256
        if getattr(pipeline, "weight_size_bytes", None):
            weight_size = pipeline.weight_size_bytes
        if getattr(pipeline, "device", None) is not None:
            device_str = str(pipeline.device)
        if getattr(pipeline, "adapter", None):
            skip = ("history", "trainable_names")
            adapter = {k: v for k, v in pipeline.adapter.items() if k not in skip}

    if checkpoint_path is not None:
        resolved_checkpoint_path = str(checkpoint_path)
        if checkpoint_source is None:
            checkpoint_source = "explicit_path"

    model_record: dict[str, Any] = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "license": MODEL_LICENSE,
        "weight_file": MODEL_FILENAME,
        "weight_sha256": weight_sha256,
        "weight_size_bytes": weight_size,
        "weight_format": "safetensors",
    }
    if checkpoint_source is not None:
        model_record["checkpoint_source"] = checkpoint_source
    if resolved_checkpoint_path is not None:
        model_record["checkpoint_path"] = resolved_checkpoint_path
    if pipeline is not None or checkpoint_path is not None:
        model_record["manifest_verified"] = manifest_verified

    inference_record: dict[str, Any] = {
        "counting": (
            "open-world counting: a text prompt and/or up to three exemplar boxes describe the object; 900 decoder "
            "queries each predict a box and per-token similarity logits; the queries whose best token score "
            "exceeds the threshold are counted and their box centres returned as points"
        ),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "score_semantics": "sigmoid token similarities, uncalibrated; the count is a threshold decision, not a probability",
        "training_box": f"{TARGET_BOX_PX:g} x {TARGET_BOX_PX:g} px boxes centred on FSC-147 points (upstream), so predicted boxes are read as points",
        "max_exemplars": MAX_EXEMPLARS,
        "deformable_attention": "pure-PyTorch grid_sample path (no compiled op) on CPU and CUDA alike",
    }
    if device_str is not None:
        inference_record["device"] = device_str

    provenance: dict[str, Any] = {
        "schema_version": 1,
        "model": model_record,
        "processor": {
            "resize": f"shortest side {SHORT_SIDE} px, longest at most {MAX_SIDE} px (upstream's test transform)",
            "normalization": "ImageNet mean / std",
            "tokenizer": {"id": TOKENIZER_MODEL_ID, "revision": TOKENIZER_REVISION, "license": TOKENIZER_LICENSE, "weights": "inside the CountGD checkpoint"},
        },
        "code": {"repository": UPSTREAM_REPOSITORY, "commit": UPSTREAM_COMMIT, "license": UPSTREAM_LICENSE, "carried_as": "modeling.py (vendored)"},
        "inference": inference_record,
        "adapter": adapter,
    }
    if include_runtime:
        provenance["runtime"] = {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": sys.platform,
            "packages": {name: _package_version(name) for name in _RUNTIME_PACKAGES},
        }
    return provenance


def write_provenance(
    path: str | Path,
    *,
    pipeline: Any | None = None,
    checkpoint_path: str | Path | None = None,
    include_runtime: bool = True,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            build_provenance(pipeline=pipeline, checkpoint_path=checkpoint_path, include_runtime=include_runtime),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return target
