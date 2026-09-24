"""The fetch/convert command line: dry run, verify-only, the failure exit code, and the DIMER archive (which
carries the converted safetensors and never the pickle). Stand-in files; no network."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

import countgd_pipeline.model as model_mod
from countgd_pipeline import (
    MANIFEST_NAME,
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_REVISION,
    SOURCE_CKPT_NAME,
)

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("fetch_weights", ROOT / "scripts" / "fetch_weights.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    source, converted = b"pickle-stand-in", b"safetensors-stand-in"
    (tmp_path / SOURCE_CKPT_NAME).write_bytes(source)
    (tmp_path / "README.md").write_bytes(b"# space\n")
    (tmp_path / MODEL_FILENAME).write_bytes(converted)
    files = [{"path": n, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n, b in (("README.md", b"# space\n"), (SOURCE_CKPT_NAME, source))]
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"modelId": MODEL_ID, "revision": MODEL_REVISION, "files": files}), encoding="utf-8")
    for name, value in (("SOURCE_CKPT_SHA256", hashlib.sha256(source).hexdigest()), ("SOURCE_CKPT_SIZE_BYTES", len(source)), ("MODEL_SHA256", hashlib.sha256(converted).hexdigest()), ("MODEL_SIZE_BYTES", len(converted))):
        monkeypatch.setattr(model_mod, name, value)
    return tmp_path


def test_verify_only_and_dry_run(snapshot, capsys):
    script = _script()
    assert script.main(["--dest", str(snapshot), "--verify-only"]) == 0
    (snapshot / "README.md").unlink()
    assert script.main(["--dest", str(snapshot), "--dry-run"]) == 0
    assert "would fetch README.md" in capsys.readouterr().out
    assert script.main(["--dest", str(snapshot), "--verify-only"]) == 1


def test_a_directory_without_the_manifest_is_refused(tmp_path, capsys):
    assert _script().main(["--dest", str(tmp_path), "--verify-only"]) == 1
    assert "no dimer-base-manifest.json" in capsys.readouterr().out


def test_the_default_run_stages_nothing_when_complete_and_zips_only_the_converted_file(snapshot, tmp_path):
    archive = tmp_path / "dimer.zip"
    assert _script().main(["--dest", str(snapshot), "--zip", str(archive)]) == 0
    with zipfile.ZipFile(archive) as z:
        assert z.namelist() == [f"countgd/{MODEL_FILENAME}"]


def test_a_tampered_converted_file_fails_the_run(snapshot):
    (snapshot / MODEL_FILENAME).write_bytes(b"safetensors-stand-iX")
    assert _script().main(["--dest", str(snapshot)]) == 1
