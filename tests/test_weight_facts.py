"""The weight prose may only quote SHA-256 digests and byte counts that a committed manifest records
or that tools/validate_release_assets.py declares, with a label, in EXTERNAL_WEIGHT_BYTES / _DIGESTS."""
# ruff: noqa: E501

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRONG_SIZE = 987_654_321_013
WRONG_DIGEST = "ab" * 32


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_release_assets", ROOT / "tools" / "validate_release_assets.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


def _copy_docs(tmp_path: Path) -> Path:
    for name in validator.WEIGHT_DOCS:
        if (ROOT / name).exists():
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / name, tmp_path / name)
    for manifest in ROOT.glob("weights/*/dimer-base-manifest.json"):
        target = tmp_path / manifest.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(manifest, target)
    return tmp_path


def _append(root: Path, text: str) -> None:
    doc = next(root / name for name in validator.WEIGHT_DOCS if (root / name).exists())
    doc.write_text(doc.read_text(encoding="utf-8") + text, encoding="utf-8")


def test_repository_weight_facts_match_manifests():
    validator.validate_weight_facts()


def test_nested_manifest_facts_are_accepted(tmp_path):
    root = _copy_docs(tmp_path)
    manifest_path = next(root.glob("weights/*/dimer-base-manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit_digest = "ef" * 32
    manifest["files"].append(
        {
            "path": "nested-source.bin",
            "bytes": 1,
            "sha256": "cd" * 32,
            "pickleAuditSha256": audit_digest,
            "convertsTo": {"path": "nested.bin", "bytes": WRONG_SIZE, "sha256": WRONG_DIGEST},
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _append(root, f"\n{WRONG_SIZE:,} bytes; SHA-256 `{WRONG_DIGEST}`; audit `{audit_digest}`\n")
    validator.validate_weight_facts(root)


def test_wrong_byte_count_is_rejected(tmp_path):
    assert WRONG_SIZE not in validator.EXTERNAL_WEIGHT_BYTES
    root = _copy_docs(tmp_path)
    _append(root, f"\n- stray file ({WRONG_SIZE:,} bytes)\n")
    with pytest.raises(validator.ValidationError, match="byte counts"):
        validator.validate_weight_facts(root)


def test_wrong_digest_is_rejected(tmp_path):
    assert WRONG_DIGEST not in validator.EXTERNAL_WEIGHT_DIGESTS
    root = _copy_docs(tmp_path)
    _append(root, f"\nSHA-256: `{WRONG_DIGEST}`\n")
    with pytest.raises(validator.ValidationError, match="SHA-256"):
        validator.validate_weight_facts(root)


@pytest.mark.parametrize("wrong_digest", [WRONG_DIGEST.upper(), "aB" * 32])
def test_wrong_digest_case_variants_are_rejected(tmp_path, wrong_digest):
    root = _copy_docs(tmp_path)
    _append(root, f"\nSHA-256: `{wrong_digest}`\n")
    with pytest.raises(validator.ValidationError, match="SHA-256"):
        validator.validate_weight_facts(root)


def test_grouped_total_bytes_is_read_whole(tmp_path):
    root = _copy_docs(tmp_path)
    _append(root, "\n- stray manifest `totalBytes` 987,654,321,013\n")
    with pytest.raises(validator.ValidationError, match="987654321013"):
        validator.validate_weight_facts(root)


def test_spaced_byte_count_is_read_whole(tmp_path):
    root = _copy_docs(tmp_path)
    _append(root, "\n- stray file (987 654 321 013 bytes)\n")
    with pytest.raises(validator.ValidationError, match="987654321013"):
        validator.validate_weight_facts(root)


def test_abbreviated_byte_suffix_is_rejected(tmp_path):
    root = _copy_docs(tmp_path)
    _append(root, f"\n- stray file ({WRONG_SIZE:,} B)\n")
    with pytest.raises(validator.ValidationError, match="987654321013"):
        validator.validate_weight_facts(root)


@pytest.mark.parametrize(
    "text",
    [
        f'\n- stray manifest `"totalBytes": {WRONG_SIZE:,}`\n',
        f"\n- stray manifest `totalBytes = {WRONG_SIZE:,}`\n",
    ],
)
def test_total_bytes_separators_are_rejected(tmp_path, text):
    root = _copy_docs(tmp_path)
    _append(root, text)
    with pytest.raises(validator.ValidationError, match="987654321013"):
        validator.validate_weight_facts(root)
