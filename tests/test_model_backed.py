"""Model-backed regressions on the real converted checkpoint, on the CPU: the three prompt modes on the demo
scene, box-scored evaluation, a one-step adaptation with an artifact round trip, the no-validation rule, the
artifact tensor-set refusal and the transactional rollback. Skipped when `weights/countgd/countgd.safetensors`
has not been produced (`python scripts/fetch_weights.py`). The CUDA variant is opt-in (COUNTGD_TEST_CUDA=1)."""
# ruff: noqa: E501

from __future__ import annotations

import json
import os

import pytest
import torch

from countgd_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    DEMO_SEED,
    MODEL_FILENAME,
    MODEL_SHA256,
    TOKENIZER_WEIGHTS_DIR,
    CountGDPipeline,
    build_synthetic_dataset,
    synthetic_scene,
)

if not (DEFAULT_WEIGHTS_DIR / MODEL_FILENAME).is_file():
    pytest.skip("converted checkpoint absent; run scripts/fetch_weights.py", allow_module_level=True)


@pytest.fixture(scope="module")
def pipe():
    return CountGDPipeline.from_pretrained(device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR, tokenizer_dir=TOKENIZER_WEIGHTS_DIR)


@pytest.fixture(scope="module")
def splits():
    return build_synthetic_dataset({"train": 2, "validation": 1, "test": 2})


def test_the_demo_scene_separates_the_three_prompt_modes(pipe):
    """Scene DEMO_SEED: 35 blue circles among 16 green triangles. Text (and text + exemplars) count the circles;
    exemplars alone count every shape — the behaviour the tutorial shows and the adaptation targets."""
    assert pipe.weight_sha256 == MODEL_SHA256 and pipe.manifest_verified is True
    scene = synthetic_scene(DEMO_SEED)
    assert (scene["label"], scene["count"], scene["distractors"]["count"]) == ("blue circle", 35, 16)
    counts = {}
    for name, kwargs in (("text", {"text": scene["label"]}), ("exemplars", {"exemplars": scene["exemplars"]}), ("both", {"text": scene["label"], "exemplars": scene["exemplars"]})):
        entry = pipe.count(scene["image"], **kwargs)["results"][0]
        assert len(entry["boxes"]) == entry["count"] == len(entry["points"])
        counts[name] = entry["count"]
    assert abs(counts["text"] - 35) <= 3 and abs(counts["both"] - 35) <= 3, counts
    assert counts["exemplars"] >= 35 + 8, counts


def test_evaluate_reports_count_point_and_box_scores(pipe, splits):
    result = pipe.evaluate(splits["test"])
    assert result["n"] == 2 and result["boxes"]["n"] == 2 and result["localisation"]["n"] == 2
    assert 0.0 <= result["boxes"]["f1"] <= 1.0 and result["boxes"]["mean_best_iou"] > 0.2
    assert result["verdict"] == "small-sample" and result["adapted"] is False


def test_one_epoch_continuation_and_artifact_round_trip(pipe, splits, tmp_path):
    base = {k: v.detach().clone() for k, v in pipe.model.named_parameters()}
    try:
        report = pipe.adapt(splits["train"], splits["validation"], epochs=1, lr=1e-4)
        assert len(report["history"]) == 2 and report["best_epoch"] in (0, 1) and report["n_trainable"] > 0
        out = pipe.save_artifact(tmp_path / "adapter")
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["base_model"]["weight_sha256"] == MODEL_SHA256
        adapted = {k: v.detach().clone() for k, v in pipe.model.named_parameters() if k in report_names(pipe)}
        pipe._overlay({k: base[k] for k in adapted})
        pipe.load_artifact(out)
        for name, value in adapted.items():
            assert torch.equal(dict(pipe.model.named_parameters())[name], value)
    finally:
        pipe._overlay({k: v for k, v in base.items() if k in report_names(pipe)})
        pipe.adapter = None


def report_names(pipe):
    return set(pipe._trainable_names(2))


def test_no_validation_keeps_the_final_epoch(pipe, splits):
    base = {k: v.detach().clone() for k, v in pipe.model.named_parameters() if k in report_names(pipe)}
    try:
        report = pipe.adapt(splits["train"][:1], epochs=1, lr=1e-4)
        assert report["best_epoch"] == 1 and report["selection"].startswith("final epoch")
        assert any(not torch.equal(dict(pipe.model.named_parameters())[k], v) for k, v in base.items())
    finally:
        pipe._overlay(base)
        pipe.adapter = None


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_configuration(pipe, tmp_path):
    pipe.adapter = {"trainable_layers": 2, "trainable_names": pipe._trainable_names(2), "history": []}
    try:
        out = pipe.save_artifact(tmp_path / "a")
    finally:
        pipe.adapter = None
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest["adapter"]["trainable_layers"] = 1
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="tensor list"):
        pipe.load_artifact(out)


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe, splits):
    before = {k: v.detach().clone() for k, v in pipe.model.named_parameters() if k in report_names(pipe)}

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        pipe.adapt(splits["train"][:1], epochs=1, lr=1e-3, progress=boom)
    for name, value in before.items():
        assert torch.equal(dict(pipe.model.named_parameters())[name], value)
    assert all(not p.requires_grad for p in pipe.model.parameters())


@pytest.mark.skipif(os.environ.get("COUNTGD_TEST_CUDA") != "1" or not torch.cuda.is_available(), reason="opt-in: COUNTGD_TEST_CUDA=1 and a visible CUDA device")
def test_count_and_adapt_run_on_a_cuda_device(splits):
    gpu = CountGDPipeline.from_pretrained(device="cuda", weights_dir=DEFAULT_WEIGHTS_DIR, tokenizer_dir=TOKENIZER_WEIGHTS_DIR)
    scene = synthetic_scene(DEMO_SEED)
    assert gpu.count(scene["image"], text=scene["label"])["results"][0]["count"] > 0
    assert gpu.adapt(splits["train"][:1], epochs=1, lr=1e-4)["n_train"] == 1
