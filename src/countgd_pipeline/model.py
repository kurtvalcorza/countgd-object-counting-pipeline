# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer, BertConfig, BertModel

from .config import (
    ALLOWED_CHECKPOINT_FILES,
    DEFAULT_MODEL_KEY,
    MODEL_CONFIG,
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    MODEL_SIZE_BYTES,
    TOKENIZER_FILES,
    TOKENIZER_KEY,
    TOKENIZER_MODEL_ID,
    TOKENIZER_REVISION,
    UNSAFE_WEIGHT_EXTENSIONS,
)
from .modeling import build_groundingdino

MANIFEST_NAME = "dimer-base-manifest.json"
#: Fleet snapshot scheme (DIMER NOTEBOOK_SPEC 1.1 MOD13): the pinned files live in repository-local
#: snapshot directories named by their keys and described by committed manifests; a standalone notebook
#: carries the manifests inline and stages/verifies working-directory copies. Two snapshots here: the
#: CountGD weights and the BERT tokenizer (vocabulary + configurations, no weights).
_WEIGHTS_ROOT = Path(__file__).resolve().parents[2] / "weights"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / DEFAULT_MODEL_KEY
TOKENIZER_WEIGHTS_DIR = _WEIGHTS_ROOT / TOKENIZER_KEY


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_manifest_files(root: Path, manifest: dict[str, Any]) -> None:
    files = manifest.get("files") or []
    if not files:
        raise RuntimeError(f"Manifest {MANIFEST_NAME} at {root} contains no files")
    for entry in files:
        rel_path = entry.get("path")
        if not rel_path:
            continue
        target = root / rel_path
        if not target.is_file():
            raise RuntimeError(f"Manifest file missing: {rel_path}")
        exp_bytes = entry.get("bytes")
        if exp_bytes is not None and target.stat().st_size != exp_bytes:
            raise RuntimeError(f"Size mismatch for {rel_path}: {target.stat().st_size} != {exp_bytes}")
        exp_sha = entry.get("sha256")
        if exp_sha is not None and _sha256(target) != exp_sha:
            raise RuntimeError(f"SHA-256 mismatch for {rel_path}")


def verify_checkpoint(
    snapshot_path: str | Path,
    *,
    require_configs: bool = False,
    return_manifest_verified: bool = False,
) -> Path | tuple[Path, bool]:
    """Verify the CountGD snapshot: no unsafe formats, the manifest's sizes and digests when present, and the
    pinned `model.safetensors` byte count and SHA-256 always."""
    root = Path(snapshot_path)
    if not root.is_dir():
        raise RuntimeError(f"Checkpoint directory does not exist: {root}")
    weight_path = root / MODEL_FILENAME
    if not weight_path.is_file():
        raise RuntimeError(f"Pinned checkpoint is missing {MODEL_FILENAME}")
    unsafe = sorted(p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in UNSAFE_WEIGHT_EXTENSIONS)
    if unsafe:
        raise RuntimeError(f"Refusing unsafe weight files: {unsafe}")
    manifest_path = root / MANIFEST_NAME
    manifest_verified = False
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Corrupt manifest {MANIFEST_NAME}: {exc}") from exc
        _check_manifest_files(root, manifest)
        manifest_verified = True
    size = weight_path.stat().st_size
    if size != MODEL_SIZE_BYTES:
        raise RuntimeError(f"Unexpected {MODEL_FILENAME} size: {size}; expected {MODEL_SIZE_BYTES}")
    digest = _sha256(weight_path)
    if digest != MODEL_SHA256:
        raise RuntimeError(f"Unexpected {MODEL_FILENAME} SHA-256: {digest}; expected {MODEL_SHA256}")
    if require_configs and not (root / "config.json").is_file():
        raise RuntimeError("Missing required configuration file: config.json")
    if return_manifest_verified:
        return root, manifest_verified
    return root


def _read_manifest(root: Path, model_id: str, revision: str) -> dict[str, Any]:
    """Load and identity-check ``<root>/dimer-base-manifest.json``."""
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise RuntimeError(f"Corrupt manifest {MANIFEST_NAME}: {exc}") from exc
    if manifest.get("modelId") != model_id or manifest.get("revision") != revision:
        raise ValueError(f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, package pins {model_id}@{revision}; refusing")
    if not manifest.get("files"):
        raise RuntimeError(f"Manifest {MANIFEST_NAME} contains no files")
    return manifest


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Manifest-driven verification of the CountGD snapshot directory; raise on the first mismatch.

    The identity in the manifest must be the pinned one; every manifest entry is then size- and SHA-256-checked
    by :func:`verify_checkpoint` (which also asserts the weight file's pinned digest and byte count and refuses
    unsafe formats). Returns ``{"path": ..., **manifest}``."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root, MODEL_ID, MODEL_REVISION)
    _, manifest_verified = verify_checkpoint(root, require_configs=True, return_manifest_verified=True)
    if not manifest_verified:
        raise RuntimeError(f"manifest at {root} was not verified")  # pragma: no cover
    return {"path": str(root), **manifest}


def verify_tokenizer_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Manifest-driven verification of the BERT tokenizer snapshot (vocabulary and configurations; no weights)."""
    root = Path(path) if path is not None else TOKENIZER_WEIGHTS_DIR
    manifest = _read_manifest(root, TOKENIZER_MODEL_ID, TOKENIZER_REVISION)
    unsafe = sorted(p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in UNSAFE_WEIGHT_EXTENSIONS)
    if unsafe:
        raise RuntimeError(f"Refusing unsafe weight files in the tokenizer snapshot: {unsafe}")
    _check_manifest_files(root, manifest)
    for name in ("config.json", "vocab.txt", "tokenizer_config.json"):
        if not (root / name).is_file():
            raise RuntimeError(f"tokenizer snapshot is missing {name}")
    return {"path": str(root), **manifest}


def _hub_download(model_id: str, revision: str) -> Callable[[str, Path], None]:
    def fetch(relative_path: str, root: Path) -> None:
        from huggingface_hub import hf_hub_download

        hf_hub_download(model_id, relative_path, revision=revision, local_dir=str(root))

    return fetch


def _stage_missing(root: Path, model_id: str, revision: str, *, allow_download: bool, downloader: Callable[[str, Path], None] | None) -> list[str]:
    manifest = _read_manifest(root, model_id, revision)
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(f"snapshot at {root} is missing {missing}; pass allow_download=True to fetch them at {revision}")
    fetch = downloader or _hub_download(model_id, revision)
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch CountGD manifest-listed files that are absent locally (a fresh clone commits the manifest and the
    small files but git-ignores the weights). Returns the relative paths fetched; :func:`verify_snapshot` still
    runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    return _stage_missing(root, MODEL_ID, MODEL_REVISION, allow_download=allow_download, downloader=downloader)


def stage_missing_tokenizer_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """The same for the BERT tokenizer snapshot (all of its files are committed; a standalone notebook fetches
    them at the pinned revision)."""
    root = Path(path) if path is not None else TOKENIZER_WEIGHTS_DIR
    return _stage_missing(root, TOKENIZER_MODEL_ID, TOKENIZER_REVISION, allow_download=allow_download, downloader=downloader)


def resolve_weights_path(
    weights_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[Path, str]:
    """Resolve the CountGD weights path with precedence:

    1. Explicit argument `weights_path` -> 'explicit_path'
    2. Environment variable `COUNTGD_WEIGHTS_DIR` -> 'env_var'
    3. Source checkout convention `weights/countgd` -> 'repo_offline'
       (only if pyproject.toml exists at repo root and weights/ contains model.safetensors)
    4. Hugging Face Hub snapshot download -> 'hf_hub'
    """
    if weights_path is not None:
        return Path(weights_path), "explicit_path"
    env_dir = os.environ.get("COUNTGD_WEIGHTS_DIR")
    if env_dir:
        return Path(env_dir), "env_var"
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "pyproject.toml").is_file():
        repo_weights = repo_root / "weights" / DEFAULT_MODEL_KEY
        if (repo_weights / MODEL_FILENAME).is_file():
            return repo_weights, "repo_offline"
    hub_path = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=list(ALLOWED_CHECKPOINT_FILES),
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    )
    return hub_path, "hf_hub"


def resolve_tokenizer_path(tokenizer_path: str | Path | None = None, cache_dir: str | Path | None = None) -> tuple[Path, str]:
    """The tokenizer snapshot: explicit path, `COUNTGD_TOKENIZER_DIR`, the checkout's `weights/bert-base-uncased`, or the Hub."""
    if tokenizer_path is not None:
        return Path(tokenizer_path), "explicit_path"
    env_dir = os.environ.get("COUNTGD_TOKENIZER_DIR")
    if env_dir:
        return Path(env_dir), "env_var"
    if (TOKENIZER_WEIGHTS_DIR / "vocab.txt").is_file():
        return TOKENIZER_WEIGHTS_DIR, "repo_offline"
    hub_path = Path(
        snapshot_download(
            repo_id=TOKENIZER_MODEL_ID,
            revision=TOKENIZER_REVISION,
            allow_patterns=list(TOKENIZER_FILES),
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    )
    return hub_path, "hf_hub"


_resolve_weights_path = resolve_weights_path


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_args() -> SimpleNamespace:
    """The vendored builders read an attribute namespace of the upstream config."""
    return SimpleNamespace(device="cpu", **MODEL_CONFIG)


def load_state_dict(weight_path: str | Path) -> dict[str, torch.Tensor]:
    """Read the safetensors file and restore the shared box-head aliases its metadata records (safetensors
    stores a tied tensor once; `dec_pred_bbox_embed_share=True` ties the six decoder box heads and the encoder's)."""
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(weight_path), "pt") as handle:
        metadata = handle.metadata() or {}
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    for alias, canonical in metadata.items():
        if canonical not in tensors:
            raise RuntimeError(f"safetensors metadata aliases {alias} to a missing tensor {canonical}")
        tensors[alias] = tensors[canonical]
    return tensors


def build_model(tokenizer_dir: str | Path) -> tuple[Any, Any, Any]:
    """The GroundingDINO/CountGD network, its criterion and the tokenizer, from the vendored code and the
    tokenizer snapshot's configuration (BERT at random init: its weights come from the checkpoint)."""
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    bert = BertModel(BertConfig.from_pretrained(str(tokenizer_dir), local_files_only=True))
    model, criterion, _ = build_groundingdino(model_args(), tokenizer, bert)
    return model, criterion, tokenizer


def load_components(
    *,
    device: str | torch.device | None = None,
    cache_dir: str | Path | None = None,
    weights_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    return_metadata: bool = False,
) -> tuple[Any, Any, Any, torch.device, Path] | tuple[Any, Any, Any, torch.device, Path, dict[str, Any]]:
    """Acquire, verify, and load the one supported CountGD checkpoint into the vendored network, strict=True."""
    candidate_path, source = resolve_weights_path(weights_path=weights_path, cache_dir=cache_dir)
    verified, manifest_verified = verify_checkpoint(candidate_path, require_configs=True, return_manifest_verified=True)
    tokenizer_dir, tokenizer_source = resolve_tokenizer_path(tokenizer_path=tokenizer_path, cache_dir=cache_dir)
    for name in ("config.json", "vocab.txt", "tokenizer_config.json"):
        if not (tokenizer_dir / name).is_file():
            raise RuntimeError(f"tokenizer snapshot at {tokenizer_dir} is missing {name}")
    target_device = _resolve_device(device)
    model, criterion, tokenizer = build_model(tokenizer_dir)
    weight_file = verified / MODEL_FILENAME
    model.load_state_dict(load_state_dict(weight_file), strict=True)
    model = model.eval().to(target_device)
    criterion = criterion.to(target_device)
    metadata: dict[str, Any] = {
        "checkpoint_path": verified,
        "checkpoint_source": source,
        "tokenizer_path": tokenizer_dir,
        "tokenizer_source": tokenizer_source,
        "manifest_verified": manifest_verified,
        "weight_sha256": _sha256(weight_file),
        "weight_size_bytes": weight_file.stat().st_size,
        "device": str(target_device),
    }
    if return_metadata:
        return model, criterion, tokenizer, target_device, verified, metadata
    return model, criterion, tokenizer, target_device, verified
