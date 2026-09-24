"""Supply-chain tests for the one pinned checkpoint: identity, the static pickle audit, the conversion's refusals,
the converted file's digest and the unsafe-format refusals. Stand-in files only; nothing here reads the 1.25 GB
checkpoint."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import io
import json
import pickle
import zipfile
from pathlib import Path

import pytest
import torch

import countgd_pipeline.model as model_module
from countgd_pipeline.config import (
    ALIAS_METADATA_KEY,
    ALLOWED_CHECKPOINT_FILES,
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_REPO_TYPE,
    MODEL_REVISION,
    PICKLE_ALLOWED_GLOBALS,
    PICKLE_AUDIT_SHA256,
    SOURCE_CKPT_NAME,
    SOURCE_CKPT_SHA256,
    SOURCE_CKPT_SIZE_BYTES,
)


def _pin_converted(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    monkeypatch.setattr(model_module, "MODEL_SIZE_BYTES", len(payload))
    monkeypatch.setattr(model_module, "MODEL_SHA256", hashlib.sha256(payload).hexdigest())


def _pin_source(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    data = path.read_bytes()
    monkeypatch.setattr(model_module, "SOURCE_CKPT_SIZE_BYTES", len(data))
    monkeypatch.setattr(model_module, "SOURCE_CKPT_SHA256", hashlib.sha256(data).hexdigest())


def _torch_archive(path: Path, obj: object) -> Path:
    torch.save(obj, path)
    return path


def test_model_identity_and_revision_are_explicit() -> None:
    assert (MODEL_ID, MODEL_REPO_TYPE) == ("nikigoli/countgd", "space")
    assert MODEL_REVISION == "6e82e59569a84ee5c6aafa35d396f2d2bee57be2"
    assert SOURCE_CKPT_NAME == "checkpoint_best_regular.pth"
    assert (SOURCE_CKPT_SIZE_BYTES, SOURCE_CKPT_SHA256) == (1_250_122_522, "c1bab864b17db345b4c6e3aaabb5765bc2c0a90d0bc8defb5e664a74a50aa126")
    assert MODEL_FILENAME == "countgd.safetensors"
    assert set(ALLOWED_CHECKPOINT_FILES) == {"README.md", SOURCE_CKPT_NAME}


def test_pickle_audit_digest_is_the_digest_of_the_allow_list() -> None:
    assert hashlib.sha256("\n".join(sorted(PICKLE_ALLOWED_GLOBALS)).encode()).hexdigest() == PICKLE_AUDIT_SHA256
    assert "argparse.Namespace" in PICKLE_ALLOWED_GLOBALS and "builtins.eval" not in PICKLE_ALLOWED_GLOBALS


def test_audit_accepts_a_plain_state_dict_archive(tmp_path: Path) -> None:
    path = _torch_archive(tmp_path / "ok.pth", {"model": {"w": torch.zeros(2)}})
    report = model_module.audit_pickle(path)
    assert report["violations"] == [] and set(report["globals"]) <= set(PICKLE_ALLOWED_GLOBALS)


class _Evil:
    def __reduce__(self):
        import os

        return (os.system, ("echo pwned",))


def test_audit_refuses_an_archive_that_names_os_system_without_executing_it(tmp_path: Path) -> None:
    path = _torch_archive(tmp_path / "evil.pth", {"model": {"w": torch.zeros(1)}, "payload": _Evil()})
    with pytest.raises(ValueError, match="outside the allow-list"):
        model_module.audit_pickle(path)


def test_audit_refuses_a_bare_pickle(tmp_path: Path) -> None:
    path = tmp_path / "bare.pth"
    path.write_bytes(pickle.dumps(complex(1, 2)))
    with pytest.raises(ValueError, match="not a torch zip archive"):
        model_module.audit_pickle(path)


def test_audit_refuses_an_archive_without_a_pickle(tmp_path: Path) -> None:
    path = tmp_path / "empty.pth"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data/0", b"\0" * 8)
    with pytest.raises(ValueError, match="no pickle member"):
        model_module.audit_pickle(path)


def test_convert_refuses_a_source_off_its_pins(tmp_path: Path) -> None:
    _torch_archive(tmp_path / SOURCE_CKPT_NAME, {"model": {"w": torch.zeros(1)}})
    with pytest.raises(RuntimeError, match="size"):
        model_module.convert_checkpoint(tmp_path)


def test_convert_refuses_an_audit_digest_that_differs_from_the_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = _torch_archive(tmp_path / SOURCE_CKPT_NAME, {"model": {"w": torch.zeros(1)}})
    _pin_source(monkeypatch, src)
    with pytest.raises(ValueError, match="audit digest"):
        model_module.convert_checkpoint(tmp_path)


def test_convert_ties_aliases_into_one_metadata_key_and_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse

    shared = torch.arange(6, dtype=torch.float32)
    state = {"bbox_embed.0.w": shared, "transformer.decoder.bbox_embed.0.w": shared, "b": torch.ones(3), "feature_map_encoder.x": torch.zeros(2)}
    src = _torch_archive(tmp_path / SOURCE_CKPT_NAME, {"model": state, "args": argparse.Namespace(lr=1.0), "epoch": 3})
    _pin_source(monkeypatch, src)
    monkeypatch.setattr(model_module, "PICKLE_AUDIT_SHA256", model_module.audit_pickle(src)["audit_sha256"])
    monkeypatch.setattr(model_module, "SOURCE_STATE_TENSORS", 4)
    monkeypatch.setattr(model_module, "STATE_TENSORS", 3)
    monkeypatch.setattr(model_module, "MODEL_SIZE_BYTES", -1)
    with pytest.raises(RuntimeError, match="size"):
        model_module.convert_checkpoint(tmp_path)  # writes the file, then refuses the unpinned digest
    first = (tmp_path / MODEL_FILENAME).read_bytes()
    _pin_converted(monkeypatch, first)
    (tmp_path / MODEL_FILENAME).unlink()
    report = model_module.convert_checkpoint(tmp_path)
    assert (tmp_path / MODEL_FILENAME).read_bytes() == first
    assert (report["tensors_stored"], report["aliases"], report["dropped"]) == (2, 1, 1)
    assert report["checkpoint"]["epoch"] == 3 and report["checkpoint"]["unread_entries"] == ["args", "epoch"]
    loaded = model_module.load_state_dict(tmp_path / MODEL_FILENAME)
    assert torch.equal(loaded["transformer.decoder.bbox_embed.0.w"], shared) and "feature_map_encoder.x" not in loaded


def test_load_state_dict_refuses_foreign_metadata(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    save_file({"a": torch.zeros(1)}, str(tmp_path / "x.safetensors"), metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match=ALIAS_METADATA_KEY):
        model_module.load_state_dict(tmp_path / "x.safetensors")
    save_file({"a": torch.zeros(1)}, str(tmp_path / "y.safetensors"), metadata={ALIAS_METADATA_KEY: json.dumps({"b": "missing"})})
    with pytest.raises(RuntimeError, match="missing tensor"):
        model_module.load_state_dict(tmp_path / "y.safetensors")


def test_verify_checkpoint_checks_the_converted_size_and_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"safetensors-stand-in"
    (tmp_path / MODEL_FILENAME).write_bytes(payload)
    _pin_converted(monkeypatch, payload)
    assert model_module.verify_checkpoint(tmp_path) == tmp_path
    monkeypatch.setattr(model_module, "MODEL_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="SHA-256"):
        model_module.verify_checkpoint(tmp_path)


@pytest.mark.parametrize("bad_name", ["extra.pt", "other.pth", "x.ckpt", "x.pkl", "x.h5", "x.msgpack", "pytorch_model.bin"])
def test_verify_checkpoint_refuses_unsafe_files_other_than_the_audited_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_name: str) -> None:
    payload = b"safetensors-stand-in"
    (tmp_path / MODEL_FILENAME).write_bytes(payload)
    _pin_converted(monkeypatch, payload)
    (tmp_path / SOURCE_CKPT_NAME).write_bytes(b"pickle")  # tolerated by name
    (tmp_path / bad_name).write_bytes(b"pickle")
    with pytest.raises(RuntimeError, match="unsafe"):
        model_module.verify_checkpoint(tmp_path)


def test_verify_checkpoint_checks_the_manifest_when_the_source_sits_beside_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"safetensors-stand-in"
    (tmp_path / MODEL_FILENAME).write_bytes(payload)
    _pin_converted(monkeypatch, payload)
    (tmp_path / SOURCE_CKPT_NAME).write_bytes(b"source")
    manifest = {"modelId": MODEL_ID, "revision": MODEL_REVISION, "files": [{"path": SOURCE_CKPT_NAME, "bytes": 6, "sha256": hashlib.sha256(b"source").hexdigest()}]}
    (tmp_path / model_module.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    _, verified = model_module.verify_checkpoint(tmp_path, return_manifest_verified=True)
    assert verified is True
    (tmp_path / SOURCE_CKPT_NAME).write_bytes(b"sourcX")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        model_module.verify_checkpoint(tmp_path)


def test_ensure_converted_refuses_a_tampered_converted_file_instead_of_regenerating(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / MODEL_FILENAME).write_bytes(b"tampered")
    _pin_converted(monkeypatch, b"original")  # same length: the digest check is what refuses it
    with pytest.raises(RuntimeError, match="SHA-256"):
        model_module.ensure_converted(tmp_path)


def test_resolve_weights_path_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = tmp_path / "explicit"
    assert model_module.resolve_weights_path(weights_path=explicit) == (explicit, "explicit_path")
    monkeypatch.setenv("COUNTGD_WEIGHTS_DIR", str(tmp_path / "env"))
    assert model_module.resolve_weights_path() == (tmp_path / "env", "env_var")
    monkeypatch.delenv("COUNTGD_WEIGHTS_DIR")
    monkeypatch.setattr(model_module, "__file__", str(tmp_path / "no-repo" / "src" / "pkg" / "model.py"))
    fetched = []
    monkeypatch.setattr(model_module, "_hub_download", lambda mid, rev, rtype: lambda rel, root: fetched.append((mid, rev, rtype, rel, root)))
    resolved, source = model_module.resolve_weights_path(cache_dir=tmp_path / "cache")
    assert (resolved, source) == (tmp_path / "cache" / "countgd", "hf_hub")
    assert fetched == [(MODEL_ID, MODEL_REVISION, "space", SOURCE_CKPT_NAME, tmp_path / "cache" / "countgd")]


def test_the_source_checkpoint_is_never_served_as_the_weight_file() -> None:
    assert not MODEL_FILENAME.endswith((".pth", ".pt", ".bin"))
    buffer = io.BytesIO()
    torch.save({"x": torch.zeros(1)}, buffer)
    assert zipfile.is_zipfile(io.BytesIO(buffer.getvalue()))  # the audit's precondition holds for torch.save output
