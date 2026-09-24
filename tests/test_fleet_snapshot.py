"""Fleet snapshot scheme (DIMER NOTEBOOK_SPEC 1.1 MOD13): manifest-driven staging and verification of the
authors' Space snapshot.

Nothing here touches the network or the real checkpoint: the snapshot directory is built from stand-in bytes,
the manifest is written by hand, and downloads go through an injected callable.
"""

# ruff: noqa: E501  -- offline fixtures and assertions are kept on single lines for readability
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from countgd_pipeline import (
    DEFAULT_MODEL_KEY,
    DEFAULT_WEIGHTS_DIR,
    MANIFEST_NAME,
    MODEL_ID,
    MODEL_REVISION,
    SOURCE_CKPT_NAME,
    SOURCE_CKPT_SHA256,
    SOURCE_CKPT_SIZE_BYTES,
    CountGDPipeline,
    stage_missing_files,
    verify_snapshot,
)
from countgd_pipeline import model as model_mod
from countgd_pipeline.config import ALLOWED_CHECKPOINT_FILES

OTHER_SHA = "0" * 40
STAND_IN = b"pickle-stand-in"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_manifest(root: Path, files: dict[str, bytes], **overrides) -> dict:
    entries = []
    for name, payload in files.items():
        (root / name).write_bytes(payload)
        entries.append({"path": name, "bytes": len(payload), "sha256": _sha(payload)})
    manifest = {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": DEFAULT_MODEL_KEY,
        "modelId": MODEL_ID,
        "repoType": "space",
        "revision": MODEL_REVISION,
        "files": entries,
        "totalBytes": sum(e["bytes"] for e in entries),
        **overrides,
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


@pytest.fixture
def pinned_to_stand_in(monkeypatch, tmp_path):
    """A snapshot of stand-in bytes with the package's source pins pointed at them."""
    write_manifest(tmp_path, {"README.md": b"# space\n", SOURCE_CKPT_NAME: STAND_IN})
    monkeypatch.setattr(model_mod, "SOURCE_CKPT_SHA256", _sha(STAND_IN))
    monkeypatch.setattr(model_mod, "SOURCE_CKPT_SIZE_BYTES", len(STAND_IN))
    return tmp_path


def test_default_weights_dir_and_committed_manifest_match_the_pins():
    assert DEFAULT_WEIGHTS_DIR.name == DEFAULT_MODEL_KEY
    manifest = json.loads((DEFAULT_WEIGHTS_DIR / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert (manifest["modelId"], manifest["revision"], manifest["repoType"], manifest["modelKey"]) == (MODEL_ID, MODEL_REVISION, "space", DEFAULT_MODEL_KEY)
    by_path = {e["path"]: e for e in manifest["files"]}
    assert set(by_path) == set(ALLOWED_CHECKPOINT_FILES)
    assert (by_path[SOURCE_CKPT_NAME]["sha256"], by_path[SOURCE_CKPT_NAME]["bytes"]) == (SOURCE_CKPT_SHA256, SOURCE_CKPT_SIZE_BYTES)
    assert manifest["totalBytes"] == sum(e["bytes"] for e in manifest["files"])
    readme = DEFAULT_WEIGHTS_DIR / "README.md"
    assert readme.is_file() and _sha(readme.read_bytes()) == by_path["README.md"]["sha256"]


def test_stage_fetches_only_the_absent_entries_through_the_injected_downloader(tmp_path):
    write_manifest(tmp_path, {"README.md": b"# space\n", SOURCE_CKPT_NAME: STAND_IN})
    (tmp_path / SOURCE_CKPT_NAME).unlink()
    calls = []

    def downloader(rel, root):
        calls.append(rel)
        (root / rel).write_bytes(STAND_IN)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=downloader) == [SOURCE_CKPT_NAME]
    assert calls == [SOURCE_CKPT_NAME]
    assert stage_missing_files(tmp_path, allow_download=False, downloader=downloader) == []


def test_stage_refuses_to_download_by_default(tmp_path):
    write_manifest(tmp_path, {"README.md": b"x"})
    (tmp_path / "README.md").unlink()
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)


@pytest.mark.parametrize("override", [{"modelId": "someone/else"}, {"revision": OTHER_SHA}])
def test_stage_and_verify_refuse_a_manifest_with_the_wrong_identity(tmp_path, override):
    write_manifest(tmp_path, {"README.md": b"x"}, **override)
    with pytest.raises(ValueError, match="refusing"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)
    with pytest.raises(ValueError, match="refusing"):
        verify_snapshot(tmp_path)


def test_stage_refuses_a_directory_without_a_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        stage_missing_files(tmp_path)


def test_the_default_downloader_fetches_from_the_space_at_the_pinned_revision(monkeypatch, tmp_path):
    seen = {}

    def fake_hf_hub_download(repo_id, filename, **kwargs):
        seen.update({"repo_id": repo_id, "filename": filename, **kwargs})
        return str(tmp_path / filename)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    write_manifest(tmp_path, {"README.md": b"x", SOURCE_CKPT_NAME: STAND_IN})
    (tmp_path / SOURCE_CKPT_NAME).unlink()
    stage_missing_files(tmp_path, allow_download=True)
    assert seen == {"repo_id": MODEL_ID, "filename": SOURCE_CKPT_NAME, "revision": MODEL_REVISION, "repo_type": "space", "local_dir": str(tmp_path)}


def test_verify_snapshot_returns_the_manifest(pinned_to_stand_in):
    result = verify_snapshot(pinned_to_stand_in)
    assert result["revision"] == MODEL_REVISION and result["path"] == str(pinned_to_stand_in)


def test_verify_snapshot_refuses_a_tampered_manifest_digest(pinned_to_stand_in):
    path = pinned_to_stand_in / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"][0]["sha256"] = "f" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_snapshot(pinned_to_stand_in)


def test_verify_snapshot_refuses_a_tampered_source(pinned_to_stand_in):
    (pinned_to_stand_in / SOURCE_CKPT_NAME).write_bytes(b"pickle-stand-iX")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_snapshot(pinned_to_stand_in)


def test_verify_snapshot_still_asserts_the_pinned_source_digest(monkeypatch, pinned_to_stand_in):
    """A manifest that agrees with the files but not with SOURCE_CKPT_SHA256 is refused (constants win)."""
    monkeypatch.setattr(model_mod, "SOURCE_CKPT_SHA256", "e" * 64)
    with pytest.raises(RuntimeError, match="SHA-256"):
        verify_snapshot(pinned_to_stand_in)


def test_verify_snapshot_refuses_a_missing_manifest_entry(pinned_to_stand_in):
    (pinned_to_stand_in / "README.md").unlink()
    with pytest.raises(RuntimeError, match="missing"):
        verify_snapshot(pinned_to_stand_in)


def test_verify_snapshot_refuses_a_stray_pickle(pinned_to_stand_in):
    (pinned_to_stand_in / "pytorch_model.bin").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="unsafe"):
        verify_snapshot(pinned_to_stand_in)


def test_from_pretrained_weights_dir_stages_verifies_and_loads_the_explicit_path(monkeypatch, pinned_to_stand_in):
    seen: dict = {}

    def fake_load_components(*, device, cache_dir, weights_path, tokenizer_path, manifest_verified, return_metadata):
        seen.update({"weights_path": weights_path, "manifest_verified": manifest_verified})
        meta = {"checkpoint_path": Path(weights_path), "checkpoint_source": "explicit_path", "manifest_verified": manifest_verified, "weight_sha256": "x", "weight_size_bytes": 1}
        return "model", "criterion", "tokenizer", "cpu", Path(weights_path), meta

    import countgd_pipeline.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "load_components", fake_load_components)
    monkeypatch.setattr(pipeline_mod.CountGDPipeline, "__init__", lambda self, *a, **k: self.__dict__.update(k))
    pipe = CountGDPipeline.from_pretrained(device="cpu", weights_dir=pinned_to_stand_in)
    assert seen == {"weights_path": pinned_to_stand_in, "manifest_verified": True}
    assert pipe.checkpoint_source == "explicit_path" and pipe.manifest_verified is True
    (pinned_to_stand_in / SOURCE_CKPT_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        CountGDPipeline.from_pretrained(device="cpu", weights_dir=pinned_to_stand_in)
    with pytest.raises(ValueError, match="not both"):
        CountGDPipeline.from_pretrained(weights_dir=pinned_to_stand_in, weights_path=pinned_to_stand_in)
