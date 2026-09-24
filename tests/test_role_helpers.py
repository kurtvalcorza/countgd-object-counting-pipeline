"""Role helpers: the validation stage (`validate_inputs`) and the per-call `evaluation_report`. Nothing here
loads the checkpoint."""
# ruff: noqa: E501

from __future__ import annotations

import pytest
from PIL import Image

from countgd_pipeline import (
    CONFIDENCE_THRESHOLD,
    DEMO_SEED,
    INPUT_SCHEMA,
    MODEL_ID,
    MODEL_REVISION,
    CountGDPipeline,
    evaluation_report,
    synthetic_scene,
    validate_inputs,
)
from test_pipeline import FakeCriterion, FakeModel, FakeTokenizer, _image


def _pipeline(detections=None) -> CountGDPipeline:
    return CountGDPipeline(FakeModel(detections), FakeCriterion(), FakeTokenizer(), device="cpu")


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    scene = synthetic_scene(DEMO_SEED)
    manifest = validate_inputs([scene["image"], _image()], text=scene["label"], exemplars=[scene["exemplars"], None], names=["scene", "dots"])
    assert manifest["schema"] == INPUT_SCHEMA and manifest["verdict"] == "accepted"
    assert [i["id"] for i in manifest["inputs"]] == ["scene", "dots"] and manifest["inputs"][0]["exemplars"] == 3
    assert manifest["model_id"] == MODEL_ID and manifest["model_revision"] == MODEL_REVISION and manifest["threshold"] == CONFIDENCE_THRESHOLD


def test_validate_inputs_single_image_default_ids() -> None:
    manifest = validate_inputs(_image(), text="dot")
    assert [i["id"] for i in manifest["inputs"]] == ["image-0"]


def test_validate_inputs_rejects_like_the_core_methods() -> None:
    pipe = _pipeline()
    for bad, message in (("https://example.invalid/x.png", "remote URLs"), ([], "at least one image"), (Image.new("RGB", (4, 4)), "sides must be")):
        with pytest.raises(ValueError, match=message):
            validate_inputs(bad, text="dot")
        with pytest.raises(ValueError, match=message):
            pipe.count(bad, text="dot")
    with pytest.raises(ValueError, match="threshold"):
        validate_inputs(_image(), text="dot", threshold=1.5)


def test_evaluation_report_reads_a_count_result_as_sample_sanity() -> None:
    result = _pipeline([(0.5, 0.5), (0.2, 0.2)]).count([_image(), _image()], text="dot")
    report = evaluation_report(result, [2, 3], sample_kind="synthetic (drawn in a test)")
    assert report["verdict"] == "sample-sanity" and report["n"] == 2 and report["model_id"] == MODEL_ID
    assert [m["id"] for m in report["metrics"]] == ["count", "mae", "exact"] and report["metrics"][1]["value"] == 0.5
    with pytest.raises(ValueError, match="no results"):
        evaluation_report({"results": []})
