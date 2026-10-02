"""Regression tests for the 2026-10-02 review fixes (CNT-M1..M5, CNT-m1..m7).

They cover the supplemental workshop notebook, which has no generator: the tests read its JSON and
execute its model-free cells (controls, scene generator, metric helpers, carried-source staging,
BYOD reader) in a temporary directory. No model is downloaded or run: detectors and CountGD are
replaced by inert stand-ins where a journey needs them. Tests that also need pandas (not in CI's
frozen environment) skip when it is absent.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageOps

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "tutorials"
    / "DIMER_Open_Vocabulary_Detection_and_Counting_Workshop.ipynb"
)
NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CELLS = {c["id"]: c for c in NB["cells"]}
CODE = [c for c in NB["cells"] if c["cell_type"] == "code"]
MARKDOWN = "\n".join("".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "markdown")


def source(cell_id: str) -> str:
    return "".join(CELLS[cell_id]["source"])


def functions(cell_id: str, *names: str) -> str:
    """Imports plus the named top-level function definitions of one cell."""
    tree = ast.parse(source(cell_id))
    keep = [
        node
        for node in tree.body
        if isinstance(node, ast.Import | ast.ImportFrom)
        or (isinstance(node, ast.FunctionDef) and node.name in names)
    ]
    assert {n.name for n in keep if isinstance(n, ast.FunctionDef)} == set(names), names
    return ast.unparse(ast.Module(body=keep, type_ignores=[]))


def run(ns: dict, *cell_ids: str) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        for cell_id in cell_ids:
            exec(compile(source(cell_id), cell_id, "exec"), ns)
    return ns


def base_namespace() -> dict:
    return {
        "np": np,
        "torch": torch,
        "Image": Image,
        "ImageDraw": ImageDraw,
        "ImageOps": ImageOps,
        "display": lambda *a, **k: None,
        "__name__": "__notebook__",
    }


@pytest.fixture
def scenes(tmp_path, monkeypatch):
    """Controls, generator, sample manifest and both metric cells, run in a temporary directory."""
    monkeypatch.chdir(tmp_path)
    return run(base_namespace(), "daa99d90", "d5aa425c", "843fb01b", "83fbdc46", "f5ee4325")


def git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload, usedforsecurity=False
    ).hexdigest()


def pins(name: str) -> dict:
    tree = ast.parse(source("083028a5"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return ast.literal_eval(node.value)
    raise AssertionError(name)


# -------------------- CNT-M1 carried source


def test_no_cell_downloads_source_at_run_time():
    for cell in CODE:
        text = "".join(cell["source"])
        assert "raw.githubusercontent.com/kurtvalcorza" not in text, cell[
            "id"
        ]  # no DIMER source by URL
        if not cell["id"].startswith(
            "carried-"
        ):  # carried samples.py modules fetch public sample data only
            assert "urlopen" not in text and "urllib" not in text, cell["id"]
            assert "raw.githubusercontent.com" not in text, cell["id"]


def test_every_carried_module_matches_its_pinned_git_blob():
    ns: dict = {}
    for cell in CODE:
        if cell["id"].startswith("carried-"):
            exec(compile("".join(cell["source"]), cell["id"], "exec"), ns)
    carried = ns["CARRIED_SOURCE"]
    expected = {f"countgd_pipeline/{k}": v for k, v in pins("COUNTGD_SOURCE").items()}
    expected |= {
        f"grounding_dino_detection_pipeline/{k}": v for k, v in pins("GROUNDING_SOURCE").items()
    }
    assert len(expected) == 13 and set(carried) == set(expected)
    for key, (blob, size) in expected.items():
        payload = carried[key].encode("utf-8")
        assert (git_blob_sha1(payload), len(payload)) == (blob, size), key


def test_staging_writes_verified_modules_and_refuses_edited_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns = run(base_namespace(), "daa99d90", "d5aa425c")
    ns["sys"] = __import__("sys")
    carried_ids = [c["id"] for c in CODE if c["id"].startswith("carried-countgd")]
    before = list(ns["sys"].path)
    try:
        run(ns, *carried_ids, "083028a5")
        staged = sorted(
            p.name for p in (tmp_path / "work" / "carrier_src" / "countgd_pipeline").glob("*.py")
        )
        assert staged == sorted(pins("COUNTGD_SOURCE"))
        ns["CARRIED_SOURCE"]["countgd_pipeline/config.py"] += "# edited\n"
        with pytest.raises(ValueError, match="integrity mismatch: countgd_pipeline/config.py"):
            ns["stage_carried_package"]("countgd_pipeline", pins("COUNTGD_SOURCE"))
        del ns["CARRIED_SOURCE"]["countgd_pipeline/config.py"]
        with pytest.raises(
            RuntimeError, match="carried source countgd_pipeline/config.py is missing"
        ):
            ns["stage_carried_package"]("countgd_pipeline", pins("COUNTGD_SOURCE"))
    finally:
        ns["sys"].path[:] = before


def test_full_tier_stages_grounding_source_from_the_carried_cells():
    full = source("229ad112")
    assert 'stage_carried_package("grounding_dino_detection_pipeline",GROUNDING_SOURCE)' in full


def test_metadata_records_standalone_carrier():
    dimer = NB["metadata"]["dimer"]
    assert dimer["standalone"] is True
    assert dimer["release_status"] == "candidate" and dimer["clean_runtime_evidence"] == "pending"
    assert set(dimer["carried_sources"]) == {
        "countgd_pipeline",
        "grounding_dino_detection_pipeline",
    }
    assert dimer["review_revisions"][0]["reviewed_commit"].startswith("413d558")


# -------------------- CNT-M2 BYOD


def _zip(path: Path, scenes_csv: str, boxes_csv: str, images: dict) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("scenes.csv", scenes_csv)
        archive.writestr("boxes.csv", boxes_csv)
        for name, image in images.items():
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            archive.writestr(name, buffer.getvalue())
    return path


def _images(n: int) -> dict:
    return {f"images/s{i}.png": Image.new("RGB", (96, 64), (10 * i, 120, 200)) for i in range(n)}


def _valid_tables(n: int = 6, split: bool = True) -> tuple[str, str]:
    names = ["train", "train", "validation", "validation", "test", "test"]
    scenes_csv = (
        "image_id,file,target_label"
        + (",split" if split else "")
        + "\n"
        + "".join(
            f"s{i},images/s{i}.png,Red Apple" + (f",{names[i % 6]}" if split else "") + "\n"
            for i in range(n)
        )
    )
    lines = []
    for i in range(n):
        lines += [
            f"s{i},red apple,{5 + 12 * k},5,{14 + 12 * k},14,{'true' if k == 0 else 'false'}\n"
            for k in range(2 + i % 2)
        ]
        lines.append(f"s{i},green leaf,60,30,80,50,false\n")
    return scenes_csv, "image_id,label,x0,y0,x1,y1,is_exemplar\n" + "".join(lines)


@pytest.fixture
def byod(scenes):
    return run(scenes, "cnt-byod-helpers")


def test_byod_controls_are_read_by_the_byod_cell():
    readers = [
        c["id"]
        for c in CODE
        if c["id"] != "daa99d90" and re.search(r"\bUSE_BYOD\b", "".join(c["source"]))
    ]
    assert readers == ["cnt-byod-run"]
    assert "BYOD_ZIP_PATH" in source("cnt-byod-run")


def test_valid_byod_archive_becomes_records_with_the_sample_fields(byod, tmp_path):
    path = _zip(tmp_path / "ok.zip", *_valid_tables(), _images(6))
    result = byod["read_byod_zip"](path)
    assert {k: len(v) for k, v in result["splits"].items()} == {
        "train": 2,
        "validation": 2,
        "test": 2,
    }
    record = result["splits"]["test"][0]
    for field in (
        "target_label",
        "target_boxes",
        "target_points",
        "target_count",
        "exemplars",
        "pixel_sha256",
    ):
        assert field in record
    assert record["target_label"] == "red apple" and record["prompts"] == [
        "red apple",
        "green leaf",
    ]
    assert record["target_count"] == len(record["target_boxes"]) and len(record["exemplars"]) == 1
    reference = byod["detection_reference"](record)
    assert reference["labels"].count("green leaf") == 1


def test_byod_without_split_column_uses_seeded_60_20_20(byod, tmp_path):
    path = _zip(tmp_path / "auto.zip", *_valid_tables(10, split=False), _images(10))
    first = byod["read_byod_zip"](path)
    second = byod["read_byod_zip"](path)
    assert {k: len(v) for k, v in first["splits"].items()} == {
        "train": 6,
        "validation": 2,
        "test": 2,
    }
    assert [r["id"] for r in first["splits"]["test"]] == [r["id"] for r in second["splits"]["test"]]
    assert "seed 42" in first["split_source"]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_column", r"scenes.csv: missing columns \['target_label'\]"),
        ("unknown_image", r"scenes.csv row 3: file 'images/zz.png' is not in the zip"),
        (
            "box_outside",
            r"boxes.csv row 2: box \[5, 5, 140, 14\] is empty or outside the 96x64 image",
        ),
        ("too_many_exemplars", r"scene 's0': 4 exemplar boxes; at most 3"),
        ("exemplar_not_target", r"boxes.csv row \d+: exemplar box labelled 'green leaf'"),
        ("bad_split", r"scenes.csv row 2: split 'holdout'"),
        ("mixed_split", r"fill the split column for every row"),
        ("label_separator", r"label 'red. apple' must be 1-48 characters"),
        ("test_label_unseen", r"test target labels \['pear'\] never occur in validation"),
        ("duplicate_image", r"contain the same image"),
    ],
)
def test_invalid_byod_archives_are_refused_with_the_offending_row(byod, tmp_path, case, message):
    scenes_csv, boxes_csv = _valid_tables()
    images = _images(6)
    if case == "missing_column":
        scenes_csv = scenes_csv.replace(
            "image_id,file,target_label,split", "image_id,file,label,split"
        )
    elif case == "unknown_image":
        scenes_csv = scenes_csv.replace("s1,images/s1.png", "s1,images/zz.png")
    elif case == "box_outside":
        boxes_csv = boxes_csv.replace("s0,red apple,5,5,14,14", "s0,red apple,5,5,140,14", 1)
    elif case == "too_many_exemplars":
        boxes_csv += "".join(
            f"s0,red apple,{30 + 6 * k},40,{35 + 6 * k},45,true\n" for k in range(3)
        )
    elif case == "exemplar_not_target":
        boxes_csv += "s0,green leaf,1,40,10,50,true\n"
    elif case == "bad_split":
        scenes_csv = scenes_csv.replace(
            "s0,images/s0.png,Red Apple,train", "s0,images/s0.png,Red Apple,holdout"
        )
    elif case == "mixed_split":
        scenes_csv = scenes_csv.replace(
            "s0,images/s0.png,Red Apple,train", "s0,images/s0.png,Red Apple,"
        )
    elif case == "label_separator":
        scenes_csv = scenes_csv.replace("s5,images/s5.png,Red Apple", "s5,images/s5.png,red. apple")
    elif case == "test_label_unseen":
        scenes_csv = scenes_csv.replace("s5,images/s5.png,Red Apple", "s5,images/s5.png,pear")
        boxes_csv = boxes_csv.replace("s5,red apple", "s5,pear")
    elif case == "duplicate_image":
        images["images/s1.png"] = images["images/s0.png"]
    path = _zip(tmp_path / f"{case}.zip", scenes_csv, boxes_csv, images)
    with pytest.raises(byod["BYODError"], match=message):
        byod["read_byod_zip"](path)


def test_byod_archive_safety_rules(byod, tmp_path):
    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("../scenes.csv", "x")
    with pytest.raises(byod["BYODError"], match=r"absolute paths and '\.\.' are not allowed"):
        byod["read_byod_zip"](traversal)
    link = tmp_path / "link.zip"
    with zipfile.ZipFile(link, "w") as archive:
        info = zipfile.ZipInfo("scenes.csv")
        info.external_attr = 0o120777 << 16
        archive.writestr(info, "target")
    with pytest.raises(byod["BYODError"], match="symbolic links are not allowed"):
        byod["read_byod_zip"](link)
    with pytest.raises(byod["BYODError"], match="not a URL"):
        byod["read_byod_zip"]("https://example.org/data.zip")
    with pytest.raises(byod["BYODError"], match="does not exist"):
        byod["read_byod_zip"](tmp_path / "missing.zip")


def test_default_path_shows_one_refusal_without_running_a_model(byod):
    example = byod["byod_refusal_example"]
    assert example["refused"] is True
    assert example["message"].startswith(
        "boxes.csv row 2: box [10, 10, 90, 20] is empty or outside the 64x48 image"
    )


def test_byod_disabled_touches_no_model(byod):
    byod.update({"USE_BYOD": False})
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(source("cnt-byod-run"), "cnt-byod-run", "exec"), byod)
    assert byod["byod_report"] is None and out.getvalue().startswith(
        "BYOD disabled (USE_BYOD=False)"
    )


class StandInDetector:
    """Every gold box whose label was prompted (score 0.9), plus a low-score duplicate."""

    def __init__(self, records, reference):
        self.records, self.reference = records, reference

    def predict(self, image, prompts, score_floor=0.05):
        record = next(r for r in self.records if r["image"] is image)
        ref = self.reference(record)
        dets = [
            {"box": b, "label": lab, "score": 0.9}
            for b, lab in zip(ref["boxes"], ref["labels"], strict=True)
            if lab in prompts
        ]
        return dets + [{**dets[0], "score": 0.07}]


class StandInCountGD:
    def evaluate(self, records, *, threshold=0.23, use_text=True, use_exemplars=True):
        rows = [
            {"id": r["id"], "gold": r["count"], "predicted": r["count"] + (0 if use_text else 2)}
            for r in records
        ]
        err = np.asarray([x["predicted"] - x["gold"] for x in rows], float)
        return {
            "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt((err**2).mean())),
            "nae": 0.0,
            "exact_fraction": float((err == 0).mean()),
            "localisation": {"f1": 1.0},
            "boxes": {"f1": 1.0},
            "per_image": rows,
            "adapted": False,
        }


def test_byod_branch_reaches_detection_counting_and_export_with_stand_ins(byod, tmp_path):
    pytest.importorskip("pandas")
    import pandas as pd

    path = _zip(tmp_path / "ok.zip", *_valid_tables(), _images(6))
    ns = byod
    exec(compile(functions("1160dede", "detector_count_predictions"), "1160dede", "exec"), ns)
    exec(compile(functions("ec46acb1", "as_countgd_record"), "ec46acb1", "exec"), ns)
    holder: dict = {}
    ns.update(
        {
            "pd": pd,
            "USE_BYOD": True,
            "BYOD_ZIP_PATH": str(path),
            "frozen_countgd": StandInCountGD(),
            "template_matching_baseline": lambda records: {
                "mae": 1.0,
                "rmse": 1.0,
                "nae": 0.1,
                "exact_fraction": 0.0,
            },
            "GroundingFrozen": lambda: StandInDetector(
                holder["records"], ns["detection_reference"]
            ),
            "OWLv2Frozen": lambda: StandInDetector(holder["records"], ns["detection_reference"]),
        }
    )
    real_reader = ns["read_byod_zip"]

    def reader(p):
        result = real_reader(p)
        holder["records"] = result["records"]
        return result

    ns["read_byod_zip"] = reader
    run(ns, "cnt-byod-run")
    report = ns["byod_report"]
    systems = [row["system"] for row in report["counting_test"]]
    assert systems == [
        "mean_count",
        "template_matcher",
        "grounding_dino_as_counter",
        "owlv2_as_counter",
        "countgd_frozen_text",
        "countgd_frozen_exemplar",
        "countgd_frozen_text+exemplar",
    ]
    assert report["detector_count_thresholds"] == {"grounding_dino": 0.4, "owlv2": 0.4}
    assert {row["model"] for row in report["detection_test"]} == {"grounding_dino", "owlv2"}
    out = Path("outputs") / "byod"
    assert sorted(p.name for p in out.iterdir()) == [
        "byod_manifest.json",
        "byod_report.json",
        "counting_test_comparison.csv",
        "detection_test_metrics.csv",
    ]


# -------------------- CNT-M3 reload parity


def test_reload_parity_compares_against_a_reference_captured_before_release():
    parity = source("06a56cd1").replace(" ", "")
    assert "a=fresh_gd.predict_boxes" not in parity and "b=fresh_gd.predict_boxes" not in parity
    assert "compare_detections(grounding_reference,reloaded)" in parity
    assert "raiseRuntimeError(" in parity
    full = source("229ad112")
    assert full.index("grounding_reference=gd_pipe.predict_boxes(") < full.index("del gd_pipe")


def test_compare_detections_passes_equal_outputs_and_fails_drift():
    ns = {"PARITY_SCORE_TOLERANCE": 1e-3, "PARITY_BOX_TOLERANCE_PX": 0.05}
    exec(compile(functions("06a56cd1", "compare_detections"), "06a56cd1", "exec"), ns)
    compare = ns["compare_detections"]
    ref = [(0.9, "red circle", [1.0, 2.0, 3.0, 4.0]), (0.5, "blue square", [5.0, 6.0, 7.0, 8.0])]
    assert compare(ref, [tuple(x) for x in ref])["passed"] is True
    assert compare(ref, [(0.9005, "red circle", [1.0, 2.0, 3.04, 4.0]), ref[1]])["passed"] is True
    assert compare(ref, [(0.8, "red circle", ref[0][2]), ref[1]])["passed"] is False
    assert compare(ref, [(0.9, "blue square", ref[0][2]), ref[1]])["passed"] is False
    assert compare(ref, [(0.9, "red circle", [1.0, 2.0, 3.5, 4.0]), ref[1]])["passed"] is False
    assert compare(ref, ref[:1])["passed"] is False


# -------------------- CNT-M4 diagnostics


def _detection_and_counting_state(ns):
    """Stand-in detector predictions and CountGD results over the real test scenes."""
    exec(compile(functions("1160dede", "detector_count_predictions"), "1160dede", "exec"), ns)
    records = ns["test_records"]
    perfect = [
        [{"box": b, "label": r["target_label"], "score": 0.9} for b in r["target_boxes"]]
        for r in records
    ]
    ns["test_detection"] = {"grounding_dino": perfect, "owlv2": perfect}
    ns["counter_thresholds"] = {"grounding_dino": 0.1, "owlv2": 0.3}
    ns["detector_counter_test"] = {"grounding_dino": {}, "owlv2": {}}

    def result(extra):
        return {
            "per_image": [
                {
                    "id": r["id"],
                    "gold": r["target_count"],
                    "predicted": r["target_count"] + extra(r),
                }
                for r in records
            ]
        }

    ns["countgd_test_results"] = {
        "frozen_text": result(lambda r: 0),
        "frozen_exemplar": result(lambda r: r["distractor_count"]),
        "frozen_text+exemplar": result(lambda r: 1),
    }
    return ns


def test_diagnostics_include_countgd_and_relate_errors_to_distractors(scenes):
    pd = pytest.importorskip("pandas")
    ns = _detection_and_counting_state(scenes)
    ns["pd"] = pd
    run(ns, "5fb1218c")
    systems = set(ns["density_df"]["system"])
    assert {
        "countgd_frozen_text",
        "countgd_frozen_exemplar",
        "countgd_frozen_text+exemplar",
        "grounding_dino_as_counter",
        "owlv2_as_counter",
    } == systems
    table = ns["distractor_table"].set_index("system")
    assert table.loc["countgd_frozen_exemplar", "share_closer_to_targets_plus_distractors"] == 1.0
    assert table.loc[
        "countgd_frozen_exemplar", "corr_signed_error_vs_distractors"
    ] == pytest.approx(1.0)
    assert table.loc["countgd_frozen_text", "share_closer_to_targets_plus_distractors"] == 0.0
    for name in ("density_metrics.csv", "density_summary.csv", "distractor_summary.csv"):
        assert (Path("outputs") / "counting" / "test" / name).is_file()


# -------------------- CNT-M5 activity


def test_activity_controls_exist_and_default_off():
    controls = source("daa99d90")
    assert re.search(r'^RUN_ACTIVITY = False\s+# @param \{type:"boolean"\}', controls, re.M)
    assert re.search(r"^ACTIVITY_COUNTGD_THRESHOLD = 0\.30\s+# @param", controls, re.M)
    assert (
        controls.count("COUNTGD_THRESHOLD_GRID") >= 3
    )  # defined, then read to validate the activity choice
    order = [c["id"] for c in NB["cells"]]
    assert order.index("918e4e83") < order.index("cnt-activity")  # runs after the freeze


def test_activity_threshold_must_come_from_the_grid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bad = source("daa99d90").replace(
        "ACTIVITY_COUNTGD_THRESHOLD = 0.30", "ACTIVITY_COUNTGD_THRESHOLD = 0.23"
    )
    with pytest.raises(ValueError, match="ACTIVITY_COUNTGD_THRESHOLD must be one of"):
        exec(compile(bad, "controls", "exec"), {})


def test_activity_uses_validation_only_and_leaves_the_freeze_untouched(scenes):
    pd = pytest.importorskip("pandas")
    ns = scenes
    exec(compile(functions("a044e35f", "sha256_file"), "a044e35f", "exec"), ns)
    exec(compile(functions("ec46acb1", "as_countgd_record"), "ec46acb1", "exec"), ns)
    ns["countgd_validation"] = [ns["as_countgd_record"](r) for r in ns["validation_records"]]
    seen = []

    class Recorder(StandInCountGD):
        def evaluate(self, records, **kwargs):
            seen.append({r["id"] for r in records})
            return super().evaluate(records, **kwargs)

    model = Recorder()
    ns.update({"pd": pd, "frozen_countgd": model, "RUN_ACTIVITY": True})
    ns["countgd_validation_results"] = {
        name: model.evaluate(ns["countgd_validation"], use_text=t, use_exemplars=e)
        for name, t, e in (
            ("text", True, False),
            ("exemplar", False, True),
            ("text+exemplar", True, True),
        )
    }
    seen.clear()
    ns["freeze_path"] = Path("outputs") / "frozen" / "frozen_experiment.json"
    ns["freeze_path"].write_text("{}", encoding="utf-8")
    run(ns, "cnt-activity")
    validation_ids = {r["id"] for r in ns["validation_records"]}
    assert seen and all(ids == validation_ids for ids in seen)
    assert {row["threshold"] for row in ns["activity_result"]["rows"]} == {0.23, 0.30}
    assert sorted(p.name for p in (Path("outputs") / "activity").iterdir()) == [
        "countgd_threshold_activity.csv",
        "countgd_threshold_activity.json",
    ]


# -------------------- CNT-m1, m2, m3 outputs


def test_export_carries_results_that_were_only_printed():
    export = source("43953ffd")
    for key in (
        '"grounding_dino_test":full_grounding_test',
        '"new_scenes":new_scene_report',
        '"activity":activity_result',
        '"byod":byod_report',
        '"distractors":distractor_table',
    ):
        assert key in export
    assert 'OUTPUT_ROOT/"new_data"/"new_scene_predictions.json"' in source("f2ca43e5")


def test_completion_summary_shows_results_bundle_and_branches():
    summary = source("491b58e6")
    for token in (
        '"test_ap50"',
        '"test_mae"',
        '"test_box_f1"',
        '"report_bundle":bundle',
        '"optional_branches"',
    ):
        assert token in summary
    assert "workshop complete" not in summary


def test_bundle_leaves_out_files_from_an_earlier_run(tmp_path):
    ns: dict = {"Path": Path}
    exec(compile(functions("43953ffd", "write_run_bundle"), "43953ffd", "exec"), ns)
    out = tmp_path / "outputs"
    (out / "sub").mkdir(parents=True)
    stale = out / "sub" / "old.json"
    stale.write_text("old")
    os.utime(stale, (time.time() - 3600, time.time() - 3600))
    started = time.time()
    (out / "new.json").write_text("new")
    bundle = tmp_path / "bundle.zip"
    included, excluded = ns["write_run_bundle"](out, started, bundle)
    assert included == ["new.json"] and excluded == ["sub/old.json"]
    assert zipfile.ZipFile(bundle).namelist() == ["new.json"]


def test_run_start_survives_rerunning_the_controls_cell(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns = run({}, "daa99d90")
    first = ns["RUN_STARTED"]
    time.sleep(0.01)
    run(ns, "daa99d90")
    assert ns["RUN_STARTED"] == first


# -------------------- CNT-m4 pixel digest


def test_pixel_digest_is_independent_of_png_encoding(scenes):
    record = scenes["test_records"][0]
    image = record["image"]
    assert (
        record["pixel_sha256"]
        == hashlib.sha256(
            f"RGB:{image.size[0]}x{image.size[1]}:".encode() + image.convert("RGB").tobytes()
        ).hexdigest()
    )
    for level in (0, 9):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", compress_level=level)
        assert (
            scenes["pixel_digest"](Image.open(io.BytesIO(buffer.getvalue())))
            == record["pixel_sha256"]
        )
    sizes = {
        s: (sum(r["target_count"] for r in rs), sum(r["distractor_count"] for r in rs))
        for s, rs in scenes["splits"].items()
    }
    assert sizes == {
        "train": (541, 303),
        "validation": (232, 118),
        "test": (278, 143),
    }  # the hosted run's scenes


# -------------------- CNT-m5, m6, m7 learner layer


def test_infrastructure_cells_are_titled_and_collapsed():
    infra = [c for c in CODE if "".join(c["source"]).startswith("# @title Infrastructure")]
    assert len(infra) >= 20
    assert all(c["metadata"].get("cellView") == "form" for c in infra)
    for cell_id in (
        "ef509873",
        "327a749f",
        "a044e35f",
        "083028a5",
        "b8cd8f11",
        "43953ffd",
        "cnt-byod-helpers",
    ):
        assert source(cell_id).startswith("# @title Infrastructure"), cell_id


def test_terminology_structure_and_objectives():
    assert "Try it yourselfs" not in MARKDOWN
    assert len(re.findall(r"^#+ .*Troubleshooting", MARKDOWN, re.M)) == 1
    assert "guided-04" not in CELLS
    assert "# @title Workshop controls" not in source("daa99d90")
    goals = source("62e90abd")
    assert "understand" not in goals


def test_principal_results_have_predictions_notes_and_worked_answers():
    for cell_id in ("d7e8b2a1", "20e0436c", "1dfd2c62", "0fbbd3fb", "3e058b9e"):
        assert "**Before you run it:**" in source(cell_id), cell_id
    for cell_id in (
        "cnt-notice-09",
        "cnt-notice-10",
        "cnt-notice-12",
        "cnt-notice-14",
        "cnt-notice-19",
    ):
        assert source(cell_id).startswith("> **What to notice.**") and "<details>" in source(
            cell_id
        ), cell_id
    exercises = source("d976fb4f")
    assert exercises.count("<details>") == 6
    for cell_id in ("747c9153", "a147fce8"):
        assert 'drop(columns=["per_phrase_ap50"])' in source(cell_id)


def test_opening_states_downloads_and_unmeasured_duration():
    opening = source("4fb28422")
    assert "2.6 GB" in opening and "has not been measured" in opening
    assert "Release hardening gate" not in opening and "stages the exact" not in opening


def test_cell_ids_are_unique_and_code_compiles():
    ids = [c["id"] for c in NB["cells"]]
    assert len(ids) == len(set(ids))
    for cell in CODE:
        compile("".join(cell["source"]), cell["id"], "exec")
