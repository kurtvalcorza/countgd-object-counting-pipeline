"""Role helpers: the validation stage (`validate_inputs`) and the per-call `evaluation_report`. Nothing here
loads the checkpoint."""
# ruff: noqa: E501

from __future__ import annotations

import pytest
from PIL import Image

from test_pipeline import FakeModel, FakeProcessor
from vit_mae_pipeline import (
    DEFAULT_MASK_RATIO,
    INPUT_SCHEMA,
    MODEL_ID,
    MODEL_REVISION,
    NUM_PATCHES,
    ViTMAEPipeline,
    evaluation_report,
    validate_inputs,
)


def _image(side: int = 64) -> Image.Image:
    return Image.new("RGB", (side, side), (10, 120, 200))


def _pipeline() -> ViTMAEPipeline:
    return ViTMAEPipeline(FakeModel(), FakeProcessor(), device="cpu")


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs([_image(), _image(40)], names=["a", "b"])
    assert manifest["schema"] == INPUT_SCHEMA and manifest["verdict"] == "accepted" and manifest["findings"] == []
    assert manifest["inputs"] == [{"id": "a", "mode": "RGB", "size": [64, 64]}, {"id": "b", "mode": "RGB", "size": [40, 40]}]
    assert manifest["mask_ratio"] == DEFAULT_MASK_RATIO and manifest["hidden_patches"] == int(NUM_PATCHES * DEFAULT_MASK_RATIO)
    assert manifest["model_id"] == MODEL_ID and manifest["model_revision"] == MODEL_REVISION


def test_validate_inputs_single_image_default_ids() -> None:
    manifest = validate_inputs(_image(), mask_ratio=0.5)
    assert [i["id"] for i in manifest["inputs"]] == ["image-0"] and manifest["hidden_patches"] == 98


def test_validate_inputs_rejects_like_the_core_methods() -> None:
    pipe = _pipeline()
    for bad, message in (
        ("https://example.invalid/x.png", "remote URLs"),
        ([], "at least one image"),
        (Image.new("RGB", (4, 4)), "sides must be"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_inputs(bad)
        with pytest.raises(ValueError, match=message):
            pipe.reconstruct(bad)
    with pytest.raises(ValueError, match="mask_ratio"):
        validate_inputs(_image(), mask_ratio=0)


def test_evaluation_report_reads_a_reconstruct_result_as_sample_sanity() -> None:
    result = _pipeline().reconstruct([_image(), _image(40)], seed=1)
    report = evaluation_report(result, sample_kind="synthetic (drawn in a test)")
    ids = [m["id"] for m in report["metrics"]]
    assert ids == ["masked_mse", "masked_mse_visible", "mask_ratio"] and report["verdict"] == "sample-sanity"
    assert report["n"] == 2 and report["model_id"] == MODEL_ID and "not a quality judgement" in report["note"]
    assert abs(report["metrics"][0]["value"] - sum(r["masked_mse"] for r in result["results"]) / 2) < 1e-9
    with pytest.raises(ValueError, match="no results"):
        evaluation_report({"results": []})
