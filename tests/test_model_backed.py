"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): the
reconstruction contract on the real checkpoint (masked MSE equal to the model's own loss, seeded masks, the
two fills), the linear probe with its k-NN companion, a one-epoch continuation of the decoder and last
encoder block on a dozen drawn scenes, the artifact round trip with the probe head, the loader's scope
check, the transactional guarantee and — where CUDA is visible — the same path on the accelerator. Skipped
when the weights are absent."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from vit_mae_pipeline import DEFAULT_WEIGHTS_DIR, NUM_PATCHES, ViTMAEPipeline
from vit_mae_pipeline.config import MODEL_FILENAME

pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / MODEL_FILENAME).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)

COLOURS = ["red", "green", "blue", "yellow"]
ONE_BLOCK_TRAINABLE = 33_198_592  # decoder (26,109,184) + encoder block 11 (7,087,872) + encoder LayerNorm (1,536)


def _scene(colour, shape, mark):
    image = Image.new("RGB", (256, 192), (135, 206, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 130, 256, 192], fill=(60, 179, 75))
    if shape == "circle":
        draw.ellipse([80, 40, 176, 136], fill=colour)
    else:
        draw.rectangle([80, 40, 176, 136], fill=colour)
    image.putpixel((mark, 0), (mark, 0, 0))
    return image


@pytest.fixture(scope="module")
def records():
    return [
        {
            "id": f"s{i:02d}",
            "image": _scene(COLOURS[i % 4], "circle" if i % 2 else "square", i),
            "label": f"{COLOURS[i % 4]} {'circle' if i % 2 else 'square'}",
        }
        for i in range(16)
    ]


@pytest.fixture(scope="module")
def pipe():
    return ViTMAEPipeline.from_pretrained(device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)


def test_reconstruct_reports_the_models_own_loss_under_a_seeded_mask(pipe, records):
    result = pipe.reconstruct([r["image"] for r in records[:3]], seed=5)
    rows = result["results"]
    assert all(row["hidden_patches"] == int(NUM_PATCHES * 0.75) == 147 for row in rows)
    assert abs(result["model_loss"] - float(np.mean([row["masked_mse"] for row in rows]))) < 1e-4
    assert rows[0]["reconstruction"].size == (224, 224)
    again = pipe.reconstruct([r["image"] for r in records[:3]], seed=5, return_images=False)["results"]
    assert [row["mask"] for row in rows] == [row["mask"] for row in again]
    assert all(abs(a["masked_mse"] - b["masked_mse"]) < 1e-6 for a, b in zip(rows, again, strict=True))
    half = pipe.reconstruct(records[0]["image"], mask_ratio=0.5, seed=5, return_images=False)["results"][0]
    assert half["hidden_patches"] == 98 and pipe.model.config.mask_ratio == 0.75


def test_evaluate_reconstruction_beats_both_fills_on_drawn_scenes(pipe, records):
    result = pipe.evaluate_reconstruction(records[:8], seed=0)
    assert result["n"] == 8 and result["verdict"] == "small-sample" and result["adapted"] is False
    fills = result["baselines"]
    assert result["masked_mse"] < fills["blur_fill"]["masked_mse"] < fills["mean_patch_fill"]["masked_mse"]
    assert result["psnr_masked"] > fills["blur_fill"]["psnr_masked"]
    assert len(result["per_image"]) == 8 and all(row["hidden_patches"] == 147 for row in result["per_image"])
    # the per-record seed is derived from the id, so the same record gets the same mask on a re-run
    rerun = pipe.evaluate_reconstruction(records[:8], seed=0)
    assert abs(rerun["masked_mse"] - result["masked_mse"]) < 1e-6


def test_probe_and_knn_read_the_same_features(pipe, records):
    fit = pipe.fit_probe(records[:12])
    assert fit["classes"] == sorted({r["label"] for r in records[:12]}) and fit["n_train"] == 12
    assert fit["train_loss"][1] < fit["train_loss"][0]
    metrics = pipe.evaluate(records[12:])
    assert metrics["n"] == 4 and 0.0 <= metrics["accuracy"] <= 1.0 and metrics["adapted"] is False
    assert set(metrics["per_class"]) == set(metrics["classes"]) and len(metrics["classes"]) == 4
    assert metrics["verdict"] == "small-sample" and metrics["knn"]["n"] == 4
    scored = pipe.classify(records[12]["image"])["results"][0]
    assert scored["top1"] == metrics["predictions"][0] and abs(sum(scored["scores"].values()) - 1.0) < 1e-5
    with pytest.raises(ValueError, match="not a probe class"):
        pipe.evaluate([{**records[0], "label": "purple hexagon"}, records[1]])


def test_one_epoch_continuation_and_artifact_round_trip(pipe, records, tmp_path):
    pipe.fit_probe(records[:12])
    result = pipe.adapt(records[:12], records[12:], epochs=1, trainable_blocks=1, batch_size=4)
    assert result["n_trainable"] == ONE_BLOCK_TRAINABLE and pipe.probe is None  # the probe is discarded with its features
    assert result["history"][0]["note"] == "frozen model" and result["history"][1]["train_loss"] > 0.0
    assert set(result["history"][1]["val"]) == {"masked_mse", "masked_mse_visible", "psnr_masked", "n"}
    assert all(name.startswith(("vit.encoder.layer.11.", "vit.layernorm.", "decoder.")) for name in result["trainable_names"])
    assert not any(n.startswith(("vit.embeddings.", "vit.encoder.layer.10.")) for n in result["trainable_names"])
    assert result["best_epoch"] in (0, 1) and result["selection"].startswith("lowest validation masked MSE")
    pipe.fit_probe(records[:12])
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tensors"]) == len(result["trainable_names"]) + 4 and manifest["probe"]["classes"] == pipe.probe["classes"]
    reloaded = ViTMAEPipeline.from_artifact(artifact, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    images = [r["image"] for r in records[:4]]
    a = pipe.reconstruct(images, seed=3, return_images=False)["results"]
    b = reloaded.reconstruct(images, seed=3, return_images=False)["results"]
    assert max(abs(x["masked_mse"] - y["masked_mse"]) for x, y in zip(a, b, strict=True)) < 1e-6
    ca, cb = pipe.classify(images)["results"], reloaded.classify(images)["results"]
    assert [x["top1"] for x in ca] == [y["top1"] for y in cb]
    assert max(abs(x["scores"][c] - y["scores"][c]) for x, y in zip(ca, cb, strict=True) for c in x["scores"]) < 1e-5
    assert reloaded.adapter["best_epoch"] == result["best_epoch"] and reloaded.evaluate(records[12:])["knn"] is None
    assert not any(p.requires_grad for p in pipe.model.parameters())


def test_no_validation_keeps_the_final_epoch_and_reloads_it(pipe, records, tmp_path):
    result = pipe.adapt(records[:12], None, epochs=2, trainable_blocks=1, batch_size=4)
    assert result["best_epoch"] == 2 == result["epochs"] and result["selection"].startswith("final epoch")
    assert all(entry["val"] is None for entry in result["history"]) and len(result["history"]) == 3
    artifact = pipe.save_artifact(tmp_path / "final")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["probe"] is None and len(manifest["tensors"]) == len(result["trainable_names"])
    reloaded = ViTMAEPipeline.from_artifact(artifact, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    state, other = pipe.model.state_dict(), reloaded.model.state_dict()
    assert all(torch.equal(state[name], other[name]) for name in result["trainable_names"])
    assert reloaded.adapter["trainable_blocks"] == 1 and reloaded.probe is None


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_configuration(pipe, records, tmp_path):
    from safetensors.torch import load_file, save_file

    pipe.adapt(records[:12], None, epochs=1, trainable_blocks=1, batch_size=4)
    artifact = pipe.save_artifact(tmp_path / "ok")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    fewer = tmp_path / "fewer"
    shutil.copytree(artifact, fewer)
    (fewer / "manifest.json").write_text(json.dumps({**manifest, "tensors": manifest["tensors"][:-1]}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        ViTMAEPipeline.from_artifact(fewer, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    extra = tmp_path / "extra"
    shutil.copytree(artifact, extra)
    tensors = load_file(str(extra / "adapter.safetensors"))
    tensors["zz.extra"] = torch.zeros(1)
    save_file(tensors, str(extra / "adapter.safetensors"), metadata={"format": "pt"})
    digest = hashlib.sha256((extra / "adapter.safetensors").read_bytes()).hexdigest()
    files = [{**manifest["files"][0], "bytes": (extra / "adapter.safetensors").stat().st_size, "sha256": digest}]
    (extra / "manifest.json").write_text(json.dumps({**manifest, "files": files}))
    with pytest.raises(ValueError, match="tensor names differ"):
        ViTMAEPipeline.from_artifact(extra, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    other_blocks = tmp_path / "other_blocks"
    shutil.copytree(artifact, other_blocks)
    (other_blocks / "manifest.json").write_text(json.dumps({**manifest, "adapter": {**manifest["adapter"], "trainable_blocks": 2}}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        ViTMAEPipeline.from_artifact(other_blocks, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe, records):
    before = {k: v.clone() for k, v in pipe.model.state_dict().items()}
    adapter_before = None if pipe.adapter is None else dict(pipe.adapter)

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(records[:12], None, epochs=2, trainable_blocks=1, batch_size=4, progress=boom)
    after = pipe.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before) and pipe.adapter == adapter_before  # the state before the call, adapter record included
    assert not any(p.requires_grad for p in pipe.model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not visible")
def test_reconstruct_adapt_and_reload_run_on_a_cuda_device(records, tmp_path):
    """Every tensor the reconstruction, the probe and the trainer build must land on the model's device."""
    cuda = ViTMAEPipeline.from_pretrained(device="cuda:0", weights_dir=DEFAULT_WEIGHTS_DIR)
    assert str(cuda.device) == "cuda:0"
    result = cuda.evaluate_reconstruction(records[:8], seed=0)
    assert result["masked_mse"] < result["baselines"]["blur_fill"]["masked_mse"]
    cuda.fit_probe(records[:12])
    assert 0.0 <= cuda.evaluate(records[12:])["accuracy"] <= 1.0
    adapt = cuda.adapt(records[:12], records[12:], epochs=1, trainable_blocks=1, batch_size=4)
    assert adapt["best_epoch"] in (0, 1) and adapt["history"][1]["train_loss"] > 0.0
    cuda.fit_probe(records[:12])
    artifact = cuda.save_artifact(tmp_path / "cuda")
    reloaded = ViTMAEPipeline.from_artifact(artifact, device="cuda:0", weights_dir=DEFAULT_WEIGHTS_DIR)
    images = [r["image"] for r in records[:4]]
    a = cuda.reconstruct(images, seed=3, return_images=False)["results"]
    b = reloaded.reconstruct(images, seed=3, return_images=False)["results"]
    assert max(abs(x["masked_mse"] - y["masked_mse"]) for x, y in zip(a, b, strict=True)) < 1e-5
    assert [x["top1"] for x in cuda.classify(images)["results"]] == [y["top1"] for y in reloaded.classify(images)["results"]]
