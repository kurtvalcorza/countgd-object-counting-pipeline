"""Offline tests of the dataset, metric and adaptation contracts: the pinned FSC-147 subset (fetched through an
injected fetcher), the synthetic scenes, record validation (points, exemplars, object boxes), splits, the
count / point / box metrics computed by hand, the non-neural baselines, BYOD round trips, and the adapter's
argument checks, target boxes and artifact refusals on the stand-in network. Nothing here loads the checkpoint."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from countgd_pipeline import (
    CORPUS_BYTES,
    CORPUS_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    SAMPLE_CATEGORIES,
    SAMPLE_RECORDS,
    SAMPLE_SPLIT,
    SYNTHETIC_SEED_BASE,
    SYNTHETIC_SIZE,
    CountGDPipeline,
    box_metrics,
    build_sample_dataset,
    build_synthetic_dataset,
    check_split_disjoint,
    counting_metrics,
    dataset_digest,
    fetch_corpus,
    load_byod_dataset,
    localisation_metrics,
    match_boxes,
    match_points,
    mean_count_baseline,
    read_corpus,
    split_dataset,
    synthetic_scene,
    template_matching_baseline,
    validate_dataset,
    write_dataset_csv,
)
from countgd_pipeline import pipeline as pl
from test_pipeline import FakeCriterion, FakeModel, FakeTokenizer, _image


def _pipeline(detections=None) -> CountGDPipeline:
    return CountGDPipeline(FakeModel(detections), FakeCriterion(), FakeTokenizer(), device="cpu")


def _fake_files() -> dict[str, bytes]:
    """Stand-in corpus bytes: a blank image of each record's pinned original size is enough for read_corpus."""
    out = {}
    for rid, _cat, _file, _size, _digest, w, h, *_ in SAMPLE_RECORDS:
        buffer = io.BytesIO()
        Image.new("RGB", (round(w * 384 / h) if h else w, 384), (200, 200, 200)).save(buffer, format="PNG")
        out[rid] = buffer.getvalue()
    return out


# ------------------------------------------------------------------------------------ the pinned corpus


def test_pinned_corpus_constants():
    assert len(SAMPLE_RECORDS) == 80 and len(SAMPLE_CATEGORIES) == 8
    assert sum(r[3] for r in SAMPLE_RECORDS) == CORPUS_BYTES
    assert len({r[0] for r in SAMPLE_RECORDS}) == 80 and len(CORPUS_REVISION) == 40
    for rid, cat, file, size, digest, *_rest, count, points, boxes in SAMPLE_RECORDS:
        assert cat in SAMPLE_CATEGORIES and file.endswith(".jpg") and size > 0 and len(digest) == 64
        assert len(points) == count and 8 <= count and 1 <= len(boxes) <= 3, rid
    assert SAMPLE_SPLIT == {"train": 5, "validation": 2, "test": 3}


def test_fetch_corpus_pins_every_file_and_caches(tmp_path, monkeypatch):
    import countgd_pipeline.samples as samples_mod

    payload = b"jpeg-bytes"
    monkeypatch.setattr(samples_mod, "SAMPLE_RECORDS", (("r1", "apple", "1.jpg", len(payload), hashlib.sha256(payload).hexdigest(), 10, 10, 1.0, 1.0, 0, [], []),))
    calls = []

    def fetcher(url):
        calls.append(url)
        return payload

    assert samples_mod.fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == {"r1": payload}
    assert calls == [samples_mod.image_url("1.jpg")] and CORPUS_REVISION in calls[0]
    assert samples_mod.fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == {"r1": payload} and len(calls) == 1  # cached
    (tmp_path / "1.jpg").write_bytes(b"tampered!!")
    with pytest.raises(ValueError, match="pinned"):
        samples_mod.fetch_corpus(cache_dir=tmp_path, fetcher=lambda url: b"wrong-bytes")
    with pytest.raises(ValueError, match="unexpected FSC-147 file name"):
        samples_mod.image_url("../etc/passwd")


def test_build_sample_dataset_is_stratified_seeded_and_disjoint():
    records = read_corpus(_fake_files())
    splits = build_sample_dataset(records)
    assert {k: len(v) for k, v in splits.items()} == {"train": 40, "validation": 16, "test": 24}
    for part in splits.values():
        assert sorted({r["label"] for r in part}) == sorted(SAMPLE_CATEGORIES)
    ids = [r["id"] for part in splits.values() for r in part]
    assert len(ids) == len(set(ids)) == 80
    again = build_sample_dataset(read_corpus(_fake_files()))
    assert [r["id"] for r in again["test"]] == [r["id"] for r in splits["test"]]
    assert fetch_corpus is not None


# ------------------------------------------------------------------------------------ record validation


def test_validate_dataset_reports_and_rejects():
    image = _image((100, 80))
    ok = [{"id": "a", "image": image, "label": "Dot", "count": 2, "points": [[10, 10], [20, 20]], "exemplars": [[5, 5, 15, 15]]},
          {"id": "b", "image": image, "label": "dot", "count": 1, "boxes": [[30, 30, 40, 44]]}]
    report = validate_dataset(ok, min_records=1)
    assert report["n_records"] == 2 and report["classes"] == ["dot"] and report["with_points"] == 2 and report["with_boxes"] == 1
    assert report["records"][1]["points"] == [[35.0, 37.0]]  # box centres stand in for points
    assert report["model_id"] == MODEL_ID and len(report["digest"]) == 64
    bad = [
        ({"id": "a", "image": image, "label": "dot", "count": 3, "points": [[1, 1]]}, "does not equal"),
        ({"id": "a", "image": image, "label": "dot", "count": 1, "boxes": [[50, 50, 40, 60]]}, "outside the image or empty"),
        ({"id": "a", "image": image, "label": "dot", "count": 2, "points": [[1, 1]], "boxes": [[1, 1, 5, 5], [6, 6, 9, 9]]}, "1 points but 2 boxes"),
        ({"id": "a", "image": image, "label": "dot", "count": 1, "boxes": [[1, 1, 5, 5], [6, 6, 9, 9]]}, "does not equal"),
        ({"id": "a", "image": image, "label": "dot;", "count": 0}, "label"),
        ({"id": "a", "image": image, "label": "dot", "count": -1}, "count"),
        ({"id": "a", "image": image, "label": "dot", "count": 0, "exemplars": [[1, 1, 5, 5]] * 4}, "at most 3"),
    ]
    for record, message in bad:
        with pytest.raises(ValueError, match=message):
            validate_dataset([record], min_records=1)
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset([ok[0], ok[0]], min_records=1)


def test_split_dataset_is_stratified_seeded_and_deduplicated():
    records = [{"id": f"r{i}", "image": _image(dots=[(10 + i, 10)]), "label": "dot" if i % 2 else "ring", "count": 1, "points": [[10 + i, 10]]} for i in range(20)]
    records.append({**records[0], "id": "dup"})
    splits = split_dataset(records, seed=3)
    assert sum(len(v) for v in splits.values()) == 20  # the duplicate image is dropped
    assert check_split_disjoint(splits) == {k: len(v) for k, v in splits.items()}
    assert [r["id"] for r in split_dataset(records, seed=3)["test"]] == [r["id"] for r in splits["test"]]


# ------------------------------------------------------------------------------------ synthetic scenes


def test_synthetic_scene_is_deterministic_and_its_boxes_hold_the_drawn_targets():
    a, b = synthetic_scene(1234), synthetic_scene(1234)
    assert a["image"].tobytes() == b["image"].tobytes() and a["boxes"] == b["boxes"] and a["label"] == b["label"]
    assert a["image"].size == SYNTHETIC_SIZE and a["count"] == len(a["boxes"]) == len(a["points"]) and len(a["exemplars"]) == 3
    assert all(e in a["boxes"] for e in a["exemplars"]) and a["distractors"]["label"] != a["label"]
    colour, _shape = a["label"].split()
    from countgd_pipeline.synthetic import SYNTHETIC_COLOURS

    pixels = np.asarray(a["image"])
    for x0, y0, x1, y1 in a["boxes"]:
        patch = pixels[int(y0) : int(y1), int(x0) : int(x1)]
        assert (patch == SYNTHETIC_COLOURS[colour]).all(-1).any()  # the target colour is inside its box
    assert validate_dataset([a], min_records=1)["records"][0]["boxes"] == a["boxes"]
    assert synthetic_scene(1235)["image"].tobytes() != a["image"].tobytes()
    with pytest.raises(ValueError, match="non-negative"):
        synthetic_scene(-1)


def test_synthetic_splits_use_disjoint_seeds_and_a_pinned_digest():
    splits = build_synthetic_dataset({"train": 3, "validation": 2, "test": 2})
    assert [r["id"] for r in splits["test"]] == [f"synth-{SYNTHETIC_SEED_BASE['test'] + i:05d}" for i in range(2)]
    check_split_disjoint(splits)
    assert dataset_digest(splits["train"]) == dataset_digest(build_synthetic_dataset({"train": 3})["train"])
    with pytest.raises(ValueError, match="unknown split"):
        build_synthetic_dataset({"holdout": 1})


# ------------------------------------------------------------------------------------ metrics by hand


def test_counting_metrics_by_hand():
    rows = [{"id": "a", "label": "x", "gold": 10, "predicted": 12}, {"id": "b", "label": "y", "gold": 4, "predicted": 2}]
    m = counting_metrics(rows)
    assert (m["mae"], m["rmse"], m["nae"], m["under_count_fraction"], m["exact_fraction"]) == (2.0, 2.0, (0.2 + 0.5) / 2, 0.5, 0.0)
    assert m["per_class"]["x"]["mae"] == 2.0 and m["total_gold"] == 14
    with pytest.raises(ValueError):
        counting_metrics([])


def test_point_and_box_matching_by_hand():
    assert match_points([[0, 0], [10, 10]], [[1, 0], [50, 50]], radius=4) == {"tp": 1, "fp": 1, "fn": 1}
    loc = localisation_metrics([{"tp": 1, "fp": 1, "fn": 1}, None])
    assert loc["n"] == 1 and loc["f1"] == 0.5
    gold = [[0, 0, 10, 10], [20, 20, 30, 30]]
    pred = [[0, 0, 10, 10], [25, 20, 35, 30], [100, 100, 110, 110]]  # exact, IoU 1/3, stray
    row = match_boxes(pred, gold)
    assert (row["tp"], row["fp"], row["fn"]) == (1, 2, 1)
    assert row["matched_iou"] == [1.0] and row["best_iou"] == pytest.approx([1.0, 50 / 150])
    assert match_boxes(pred, gold, threshold=0.3)["tp"] == 2
    m = box_metrics([row, None])
    assert m["n"] == 1 and m["precision"] == pytest.approx(1 / 3) and m["recall"] == 0.5 and m["mean_best_iou"] == pytest.approx((1 + 1 / 3) / 2)
    assert box_metrics([])["f1"] is None and match_boxes([], gold)["best_iou"] == [0.0, 0.0]
    with pytest.raises(ValueError, match="threshold"):
        match_boxes(pred, gold, threshold=0.0)


def test_template_matcher_gives_flat_regions_no_correlation():
    """Flat windows have no defined correlation: they must not become peaks (float error once produced
    tens of thousands of them on a flat synthetic background)."""
    from countgd_pipeline.metrics import _ncc_map, template_match

    flat = torch.full((64, 64), 0.5)
    template = torch.zeros(9, 9)
    template[3:6, 3:6] = 1.0
    ncc = _ncc_map(flat, template)
    assert torch.count_nonzero(ncc) == 0
    assert torch.count_nonzero(_ncc_map(torch.rand(32, 32), torch.full((5, 5), 0.3))) == 0  # flat template
    scene = synthetic_scene(3000)
    result = template_match(scene)
    assert result["count"] <= 3 * scene["count"] and float(_ncc_map(torch.rand(40, 40), torch.rand(7, 7)).abs().max()) <= 1.0


def test_baselines_need_training_records_and_score_with_the_same_code():
    scene = synthetic_scene(3000)
    base = mean_count_baseline([{"count": 10}, {"count": 20}], [scene])
    assert base["rows"][0]["predicted"] == 15 and base["mae"] == abs(15 - scene["count"])
    with pytest.raises(ValueError, match="training records"):
        mean_count_baseline([], [scene])
    tm = template_matching_baseline([scene])
    assert tm["n"] == 1 and tm["localisation"]["n"] == 1 and "no learning" in tm["baseline"]


# ------------------------------------------------------------------------------------ BYOD


def test_byod_directory_zip_round_trip_and_rejections(tmp_path):
    scene = synthetic_scene(42)
    folder = tmp_path / "byod"
    folder.mkdir()
    scene["image"].save(folder / "synth-00042.png")
    write_dataset_csv([{**scene, "source_file": "synth-00042.png"}], folder / "labels.csv")
    records = load_byod_dataset(folder)
    assert records[0]["boxes"] == scene["boxes"] and records[0]["count"] == scene["count"]
    assert validate_dataset(records, min_records=1)["with_boxes"] == 1
    archive = tmp_path / "byod.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for p in folder.iterdir():
            z.write(p, arcname=f"inner/{p.name}")
    assert load_byod_dataset(archive)[0]["points"] == records[0]["points"]
    (folder / "labels.csv").write_text("id,file,label\nx,synth-00042.png,dot\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_byod_dataset(folder)
    with pytest.raises(ValueError, match="directory or a .zip"):
        load_byod_dataset(tmp_path / "nothing.txt")


# ------------------------------------------------------------------------------------ adaptation contract


def test_targets_use_gold_boxes_when_present_and_point_boxes_otherwise():
    pipe = _pipeline()
    boxed = pipe._targets({"id": "a", "image": _image((100, 50)), "boxes": [[10, 10, 30, 20]], "points": [[20, 15]]})
    assert torch.allclose(boxed["boxes"], torch.tensor([[0.2, 0.3, 0.2, 0.2]]))
    pointed = pipe._targets({"id": "b", "image": _image((100, 50)), "points": [[20, 15]]})
    assert torch.allclose(pointed["boxes"], torch.tensor([[0.2, 0.3, 2 / 100, 2 / 50]]))
    with pytest.raises(ValueError, match="gold points"):
        pipe._targets({"id": "c", "image": _image(), "points": []})


def test_adapt_and_artifacts_validate_arguments_before_touching_the_model(tmp_path):
    pipe = _pipeline()
    train = [{"id": "a", "image": _image(dots=[(40, 60)]), "label": "dot", "count": 1, "boxes": [[36, 56, 44, 64]]}]
    for kwargs, message in (({"epochs": 0}, "epochs"), ({"epochs": True}, "epochs"), ({"lr": 0.0}, "lr"), ({"lr": 1.0}, "lr"), ({"trainable_layers": 7}, "trainable_layers")):
        with pytest.raises(ValueError, match=message):
            pipe.adapt(train, **kwargs)
    with pytest.raises(ValueError, match="gold points"):
        pipe.adapt([{"id": "a", "image": _image(), "label": "dot", "count": 1}])
    with pytest.raises(ValueError, match="nothing to save"):
        pipe.save_artifact(tmp_path / "a")
    assert all(not p.requires_grad for p in pipe.model.parameters())


def test_adapt_trains_the_named_tensors_keeps_the_best_epoch_and_round_trips(tmp_path):
    pipe = _pipeline([(0.25, 0.5)])
    train = [{"id": f"t{i}", "image": _image(dots=[(40, 60)]), "label": "dot", "count": 1, "boxes": [[36, 56, 44, 64]]} for i in range(2)]
    val = [{"id": "v0", "image": _image(dots=[(40, 60)]), "label": "dot", "count": 1, "boxes": [[36, 56, 44, 64]]}]
    report = pipe.adapt(train, val, epochs=2, lr=1e-3)
    assert report["best_epoch"] in (0, 1, 2) and report["n_train"] == 2 and len(report["history"]) == 3
    assert report["history"][0]["val"]["box_f1"] is not None and "gold object boxes" in pl.CountGDPipeline.adapt.__doc__
    names = pipe._trainable_names(report["trainable_layers"])
    assert names and all(n.startswith(("transformer.decoder.layers.4.", "transformer.decoder.layers.5.", "bbox_embed.0.", "transformer.decoder.norm.")) for n in names)
    out = pipe.save_artifact(tmp_path / "adapter", metadata={"note": "test"})
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["base_model"] == {"id": MODEL_ID, "revision": MODEL_REVISION, "key": "countgd", "weight_file": "countgd.safetensors", "weight_sha256": MODEL_SHA256}
    fresh = _pipeline([(0.25, 0.5)])
    fresh.load_artifact(out)
    for name in names:
        assert torch.equal(dict(fresh.model.named_parameters())[name], dict(pipe.model.named_parameters())[name])


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path):
    pipe = _pipeline()
    pipe.adapter = {"trainable_layers": 2, "trainable_names": pipe._trainable_names(2), "history": []}
    out = pipe.save_artifact(tmp_path / "ok")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    def write(**changes):
        target = tmp_path / f"bad{len(list(tmp_path.iterdir()))}"
        target.mkdir()
        (target / "adapter.safetensors").write_bytes((out / "adapter.safetensors").read_bytes())
        (target / "manifest.json").write_text(json.dumps({**manifest, **changes}), encoding="utf-8")
        return target

    cases = [
        ({"format": "other"}, "format"),
        ({"format_version": "9.9"}, "format_version"),
        ({"base_model": {**manifest["base_model"], "weight_sha256": "0" * 64}}, "different base"),
        ({"base_model": {**manifest["base_model"], "weight_file": "model.bin"}}, "different base weight file"),
        ({"files": []}, "exactly one file"),
        ({"files": [{**manifest["files"][0], "path": "../x.safetensors"}]}, "must name exactly"),
        ({"adapter": {"trainable_layers": 9}}, "trainable_layers"),
        ({"files": [{**manifest["files"][0], "sha256": "0" * 64}]}, "digest or size"),
        ({"tensors": ["bbox_embed.0.weight"]}, "tensor list"),
    ]
    for changes, message in cases:
        with pytest.raises(ValueError, match=message):
            _pipeline().load_artifact(write(**changes))


def test_evaluate_scores_boxes_only_where_gold_boxes_exist():
    pipe = _pipeline([(0.25, 0.5)])
    with_boxes = {"id": "a", "image": _image(dots=[(40, 60)]), "label": "dot", "count": 1, "boxes": [[36, 56, 44, 64]]}
    points_only = {"id": "b", "image": _image(dots=[(40, 60)]), "label": "dot", "count": 1, "points": [[40, 60]]}
    result = pipe.evaluate([with_boxes, points_only])
    assert result["boxes"]["n"] == 1 and result["localisation"]["n"] == 2 and result["mae"] == 0.0
    assert result["per_image"][0]["boxes"] is not None and result["per_image"][1]["boxes"] is None
    assert Path(__file__).name == "test_adaptation.py"
