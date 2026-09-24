"""Offline tests for the image dataset contract, the pinned corpus readers, the reconstruction and
linear-probe metrics with their baselines, BYOD loaders, CSV export, artifact-manifest rejections and
adapt() argument validation. Nothing here loads the checkpoint; photos come from injected fetchers and
tiny PIL drawings."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import zipfile

import numpy as np
import pytest
from PIL import Image, ImageDraw

from test_pipeline import FakeModel, FakeProcessor
from vit_mae_pipeline import (
    ARTIFACT_FORMAT,
    CORPUS_BYTES,
    ENCODER_LAYERS,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    SAMPLE_RECORDS,
    SAMPLE_SPLIT,
    SPECIES,
    ViTMAEPipeline,
    blur_fill,
    build_sample_dataset,
    check_split_disjoint,
    class_names,
    classification_metrics,
    colour_neighbour_baseline,
    colour_signature,
    dataset_digest,
    fetch_corpus,
    knn_scores,
    load_byod_dataset,
    majority_baseline,
    masked_mse,
    mean_patch_fill,
    observer_overlap,
    psnr_from_mse,
    read_corpus,
    reconstruction_metrics,
    split_dataset,
    validate_dataset,
    write_dataset_csv,
)
from vit_mae_pipeline import pipeline as pl
from vit_mae_pipeline import samples as sm

COLOURS = ["red", "green", "blue", "yellow", "white", "black"]


def _picture(colour, size=(96, 64), mark=0):
    image = Image.new("RGB", size, "gray")
    ImageDraw.Draw(image).rectangle([16, 12, 80, 52], fill=colour)
    image.putpixel((mark % size[0], 0), (mark % 256, 0, 0))  # a per-record pixel so digests differ
    return image


def _records(n=12, prefix="r"):
    return [
        {
            "id": f"{prefix}{i:03d}",
            "image": _picture(COLOURS[i % 6], mark=i),
            "label": COLOURS[i % 6],
        }
        for i in range(n)
    ]


def _fake_pipeline():
    return ViTMAEPipeline(FakeModel(), FakeProcessor(), device="cpu")


# --- corpus reader ----------------------------------------------------------------------------------


def test_pinned_corpus_constants():
    assert len(SAMPLE_RECORDS) == 360 and len(SPECIES) == 6
    labels = {r[1] for r in SAMPLE_RECORDS}
    assert labels == set(SPECIES) and all(
        sum(1 for r in SAMPLE_RECORDS if r[1] == k) == 60 for k in SPECIES
    )
    assert sum(r[5] for r in SAMPLE_RECORDS) == CORPUS_BYTES
    assert all(len(r[6]) == 64 and r[7].lower() in ("jpg", "jpeg", "png") for r in SAMPLE_RECORDS)
    assert len({r[2] for r in SAMPLE_RECORDS}) == 360  # distinct photo ids
    assert len({(r[1], r[4]) for r in SAMPLE_RECORDS}) == 360  # one photo per observer per species
    assert SAMPLE_SPLIT == {"train": 36, "validation": 8, "test": 16}
    assert sm.photo_url(1, "JPG").endswith("/1/medium.JPG") and sm.photo_url(2, "png").endswith(
        "/2/medium.png"
    )
    with pytest.raises(ValueError, match="extension"):
        sm.photo_url(3, "gif")


def test_fetch_corpus_pins_every_file_and_caches(tmp_path, monkeypatch):
    payload = {}
    pins = []
    for i, colour in enumerate(COLOURS[:3]):
        buffer = _picture(colour)
        import io

        raw = io.BytesIO()
        buffer.save(raw, format="JPEG")
        data = raw.getvalue()
        payload[f"https://inaturalist-open-data.s3.amazonaws.com/photos/{100 + i}/medium.jpg"] = (
            data
        )
        pins.append(
            (
                f"{colour}-00",
                colour,
                100 + i,
                900 + i,
                f"user{i}",
                len(data),
                hashlib.sha256(data).hexdigest(),
                "jpg",
            )
        )
    monkeypatch.setattr(sm, "SAMPLE_RECORDS", tuple(pins))
    monkeypatch.setattr(sm, "SPECIES", {c: (c, c.title()) for c in COLOURS[:3]})
    calls = []

    def fetcher(url):
        calls.append(url)
        return payload[url]

    files = fetch_corpus(cache_dir=tmp_path, fetcher=fetcher)
    assert sorted(files) == sorted(p[0] for p in pins) and len(calls) == 3
    assert (
        fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == files and len(calls) == 3
    )  # cached, re-hashed
    (tmp_path / "101.jpg").write_bytes(b"drifted")
    fetch_corpus(cache_dir=tmp_path, fetcher=fetcher)
    assert len(calls) == 4 and (tmp_path / "101.jpg").read_bytes() == payload[calls[-1]]
    with pytest.raises(ValueError, match="pinned"):
        fetch_corpus(cache_dir=tmp_path / "bad", fetcher=lambda url: b"tampered")
    records = read_corpus(files)
    assert [r["label"] for r in records] == COLOURS[:3] and all(
        "inat_observation_url" in r for r in records
    )


def test_build_sample_dataset_is_stratified_seeded_and_disjoint(monkeypatch):
    records = [
        {"id": f"{c}-{i:02d}", "image": _picture(c), "label": c, "observer": f"u{i}"}
        for c in COLOURS[:3]
        for i in range(10)
    ]
    sizes = {"train": 6, "validation": 2, "test": 2}
    splits = build_sample_dataset(records, seed=1, sizes=sizes)
    assert {k: len(v) for k, v in splits.items()} == {"train": 18, "validation": 6, "test": 6}
    assert class_names(splits["train"]) == sorted(COLOURS[:3])
    for part in splits.values():
        assert all(sum(1 for r in part if r["label"] == c) == len(part) // 3 for c in COLOURS[:3])
    # images are drawn identically per colour, so the pixel-digest leakage check must be run on ids here
    assert splits["train"][0]["id"] == "train-000" and "source_id" in splits["train"][0]
    assert build_sample_dataset(records, seed=1, sizes=sizes) == splits
    assert build_sample_dataset(records, seed=2, sizes=sizes) != splits
    with pytest.raises(ValueError, match="only 10 records available"):
        build_sample_dataset(records, sizes={"train": 9, "validation": 1, "test": 1})
    assert observer_overlap(splits)["observers"] == 10


# --- dataset validation -------------------------------------------------------------------------------


def test_validate_dataset_reports_and_rejects(tmp_path):
    records = _records()
    report = validate_dataset(records)
    assert report["n_records"] == 12 and report["classes"] == sorted(COLOURS)
    assert report["label_counts"] == {c: 2 for c in COLOURS} and report["image_side"] == {
        "min": 96,
        "max": 96,
    }
    assert report["digest"] == dataset_digest(report["records"]) and report["model_id"] == MODEL_ID
    path = tmp_path / "photo.jpg"
    _picture("red").save(path)
    assert validate_dataset([{**r, "image": str(path)} for r in records])["n_records"] == 12
    good = records
    for bad, message in (
        (good[:7], "8..20000"),
        ([{**good[0], "id": "bad id"}, *good[1:]], "id must match"),
        ([{**good[0], "id": good[1]["id"]}, *good[1:]], "duplicate id"),
        ([{**good[0], "image": str(tmp_path / "missing.jpg")}, *good[1:]], "not found"),
        ([{**good[0], "image": Image.new("RGB", (5000, 10))}, *good[1:]], "MAX_IMAGE_SIDE"),
        ([{**good[0], "label": ""}, *good[1:]], "label must be"),
        ([{**r, "label": "same"} for r in good], "distinct labels"),
        ([{k: v for k, v in good[0].items() if k != "label"}, *good[1:]], "missing 'label'"),
        (["not a mapping", *good[1:]], "must be a mapping"),
        ({"a": 1}, "must be a list"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_dataset(bad)


def test_split_dataset_is_stratified_seeded_and_deduplicated():
    records = _records(n=30)
    records += [{**r, "id": r["id"] + "dup"} for r in records[:6]]  # pixel-identical duplicates
    splits = split_dataset(records, val_fraction=0.2, test_fraction=0.2, seed=3)
    assert sum(len(v) for v in splits.values()) == 30 and check_split_disjoint(splits)
    assert split_dataset(records, val_fraction=0.2, test_fraction=0.2, seed=3) == splits
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.5, test_fraction=0.6)
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint({"train": records[:1], "test": [{**records[0], "id": "x"}]})


# --- metrics and baselines ----------------------------------------------------------------------------


def test_classification_metrics_and_knn():
    scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    report = classification_metrics(scores, [0, 1, 1, 1], ["a", "b"])
    assert report["n"] == 4 and abs(report["accuracy"] - 0.75) < 1e-9
    assert report["per_class"]["a"]["support"] == 1 and 0 < report["macro_f1"] < 1
    with pytest.raises(ValueError, match="grid"):
        classification_metrics(scores[:, :1], [0, 1, 1, 1], ["a", "b"])
    with pytest.raises(ValueError, match="finite"):
        classification_metrics(np.array([[np.nan, 1.0]]), [0], ["a", "b"])
    train = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    train = train / np.linalg.norm(train, axis=1, keepdims=True)
    grid = knn_scores(train, [0, 1, 1], np.array([[1.0, 0.0], [0.0, 1.0]]), 2, k=1)
    assert grid.argmax(axis=1).tolist() == [0, 1]
    with pytest.raises(ValueError, match="feature matrices"):
        knn_scores(train, [0, 1], train, 2)


def test_reconstruction_metrics_and_fills_are_computable_by_hand():
    patches = np.zeros((4, 12))
    patches[0] = 1.0  # one white patch, three black; grid 2 x 2
    mask = np.array([1, 0, 0, 0])
    prediction = np.zeros((4, 12))
    assert masked_mse(prediction, patches, mask) == 1.0  # the white patch predicted black
    assert masked_mse(prediction, patches, 1 - mask) == 0.0
    with pytest.raises(ValueError, match="at least one patch"):
        masked_mse(prediction, patches, np.zeros(4))
    filled = mean_patch_fill(patches, mask)
    assert np.allclose(filled[0], 0.0) and np.allclose(filled[1:], patches[1:])  # visible mean is black
    blurred = blur_fill(patches, mask, grid=2)
    assert np.allclose(blurred[0], 0.0)
    rows = [{"id": "a", "masked_mse": 0.1, "visible_mse": 0.2}, {"id": "b", "masked_mse": 0.3, "visible_mse": 0.4}]
    report = reconstruction_metrics(rows, mask_ratio=0.75)
    assert report["n"] == 2 and abs(report["masked_mse"] - 0.2) < 1e-12 and abs(report["masked_mse_visible"] - 0.3) < 1e-12
    assert report["psnr_masked"] == psnr_from_mse(0.2) and report["mask_ratio"] == 0.75
    assert psnr_from_mse(0.01) > psnr_from_mse(0.1)
    with pytest.raises(ValueError, match="no images"):
        reconstruction_metrics([], mask_ratio=0.75)


def test_baselines_need_training_records_and_sit_near_chance():
    records = _records(n=24)
    classes = class_names(records)
    majority = majority_baseline(records, records, classes)
    assert abs(majority["accuracy"] - 1 / 6) < 1e-9 and majority["baseline"].startswith("majority")
    colour = colour_neighbour_baseline(records, records, classes)
    assert colour["accuracy"] > 0.9  # drawn colours are trivially separable by their own signature
    assert len(colour_signature(records[0]["image"])) == 27
    with pytest.raises(ValueError, match="training records"):
        majority_baseline([], records, classes)
    with pytest.raises(ValueError, match="training records"):
        colour_neighbour_baseline([], records, classes)


def test_byod_directory_zip_round_trip_and_rejections(tmp_path):
    records = _records(n=8)
    for r in records:
        r["image"].save(
            tmp_path / r["id"], format="JPEG"
        )  # `write_dataset_csv` names BYOD files by id
    write_dataset_csv(records, tmp_path / "labels.csv")
    loaded = load_byod_dataset(tmp_path)
    assert [r["label"] for r in loaded] == [r["label"] for r in records]
    with zipfile.ZipFile(tmp_path / "byod.zip", "w") as archive:
        archive.write(tmp_path / "labels.csv", "labels.csv")
        for r in records:
            archive.write(tmp_path / r["id"], r["id"])
    assert [r["id"] for r in load_byod_dataset(tmp_path / "byod.zip")] == [r["id"] for r in records]
    with zipfile.ZipFile(tmp_path / "nolabels.zip", "w") as archive:
        archive.writestr("x.txt", "x")
    with pytest.raises(ValueError, match="labels.csv"):
        load_byod_dataset(tmp_path / "nolabels.zip")
    with pytest.raises(ValueError, match="directory or a .zip"):
        load_byod_dataset(tmp_path / "labels.csv")


# --- adaptation and artifacts without a model ---------------------------------------------------------


def test_adapt_and_artifacts_validate_arguments_before_touching_the_model(tmp_path):
    pipe = _fake_pipeline()
    records = _records()
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(records, epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(records, lr=1.0)
    with pytest.raises(ValueError, match="batch_size"):
        pipe.adapt(records, batch_size=0)
    with pytest.raises(ValueError, match="trainable_blocks"):
        pipe.adapt(records, trainable_blocks=ENCODER_LAYERS + 1)
    with pytest.raises(ValueError, match="mask_ratio"):
        pipe.adapt(records, mask_ratio=1.5)
    with pytest.raises(ValueError, match="8..20000"):
        pipe.adapt(records[:3])
    with pytest.raises(ValueError, match="call adapt"):
        pipe.save_artifact(tmp_path)
    assert pipe.adapter is None and pipe.probe is None


def test_unlabelled_records_pass_the_reconstruction_contract_and_fail_the_probe():
    records = [{"id": f"u{i}", "image": _picture(COLOURS[i % 6], mark=i)} for i in range(8)]
    report = validate_dataset(records, require_labels=False)
    assert report["n_records"] == 8 and report["classes"] == [] and report["digest"] == dataset_digest(report["records"])
    with pytest.raises(ValueError, match="missing 'label'"):
        validate_dataset(records)
    with pytest.raises(ValueError, match="label must be"):
        validate_dataset([{**records[0], "label": ""}, *records[1:]], require_labels=False)


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path):
    pipe = _fake_pipeline()
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": MODEL_SHA256},
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": ["decoder.decoder_pred.weight"],
        "adapter": {"trainable_blocks": 1},
        "probe": None,
    }

    def write(m):
        (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(m))

    write({**manifest, "format": "other"})
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(tmp_path)
    write({**manifest, "format_version": "0.9"})
    with pytest.raises(ValueError, match="format_version"):
        pipe.load_artifact(tmp_path)
    write({**manifest, "base_model": {**manifest["base_model"], "weight_sha256": "0" * 64}})
    with pytest.raises(ValueError, match="different base model"):
        pipe.load_artifact(tmp_path)
    write({**manifest, "base_model": {**manifest["base_model"], "weight_file": "other.bin"}})
    with pytest.raises(ValueError, match="different base weight file"):
        pipe.load_artifact(tmp_path)
    write({**manifest, "files": manifest["files"] * 2})
    with pytest.raises(ValueError, match="exactly one file"):
        pipe.load_artifact(tmp_path)
    write({**manifest, "files": [{**manifest["files"][0], "path": "../" + pl.ARTIFACT_WEIGHTS_NAME}]})
    with pytest.raises(ValueError, match="must name exactly|inside the artifact directory"):
        pipe.load_artifact(tmp_path)
    write({**manifest, "adapter": {"trainable_blocks": ENCODER_LAYERS + 1}})
    with pytest.raises(ValueError, match="trainable_blocks"):
        pipe.load_artifact(tmp_path)
    write({**manifest, "probe": {"classes": []}})
    with pytest.raises(ValueError, match="probe record"):
        pipe.load_artifact(tmp_path)
    write(manifest)
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="digest or size mismatch"):
        pipe.load_artifact(tmp_path)
