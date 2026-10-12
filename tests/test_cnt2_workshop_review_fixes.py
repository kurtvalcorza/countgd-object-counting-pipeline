"""Regression tests for the 2026-10-12 round review fixes of the counting workshop (CNT2-*).

CNT2-M1 and CNT2-M2 are owner decisions; this revision implements option B provisionally
(disclosure only), so their tests check the wording, not a new computation. CNT2-m1 is checked by
executing the BYOD cells in this interpreter and through the carried cell runner (no model runs: the
refusal happens before any model call).
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from test_cnt_workshop_review_fixes import (
    CELLS,
    CODE,
    NB,
    _images,
    _valid_tables,
    _zip,
    base_namespace,
    run,
    source,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "counting_workshop_env_worker.py"


@pytest.fixture
def byod(tmp_path, monkeypatch):
    """Controls, generator, metrics and the BYOD helpers, run in a temporary directory."""
    monkeypatch.chdir(tmp_path)
    ns = run(base_namespace(), "daa99d90", "d5aa425c", "843fb01b", "83fbdc46", "f5ee4325")
    return run(ns, "cnt-byod-helpers")


def section(heading: str) -> str:
    """Markdown and code of the section starting at `heading`, up to the next '# N.' heading."""
    out, on = [], False
    for cell in NB["cells"]:
        text = "".join(cell["source"])
        if cell["cell_type"] == "markdown" and text.lstrip().startswith(heading):
            on = True
        elif on and cell["cell_type"] == "markdown" and re.match(r"^# \d+\.", text.lstrip()):
            break
        if on:
            out.append(text)
    assert out, heading
    return "\n".join(out)


def _bad_zip(path: Path) -> Path:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48)).save(buffer, format="PNG")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "scenes.csv", "image_id,file,target_label,split\ns1,images/a.png,red circle,test\n"
        )
        archive.writestr(
            "boxes.csv", "image_id,label,x0,y0,x1,y1,is_exemplar\ns1,red circle,10,10,90,20,false\n"
        )
        archive.writestr("images/a.png", buffer.getvalue())
    return path


# -------------------- CNT2-m1 a refused BYOD archive is one line, not a traceback


def _run_byod_cell(ns: dict) -> list[str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        exec(compile(source("cnt-byod-run"), "cnt-byod-run", "exec"), ns)
    assert err.getvalue() == ""
    return out.getvalue().splitlines()


def test_refused_archive_prints_one_line_runs_no_model_and_clears_stale_results(byod, tmp_path):
    stale = Path("outputs") / "byod" / "byod_report.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("{}", encoding="utf-8")  # left by an earlier, valid archive in this session

    def no_model(_byod):
        raise AssertionError("a refused archive must not reach a model")

    byod.update(
        {
            "USE_BYOD": True,
            "BYOD_ZIP_PATH": str(_bad_zip(tmp_path / "bad.zip")),
            "run_byod": no_model,
        }
    )
    lines = _run_byod_cell(byod)
    assert lines == [
        "BYOD archive refused: boxes.csv row 2: box [10, 10, 90, 20] is empty or outside the"
        " 64x48 image images/a.png. No model ran; fix this and run this cell again."
    ]
    assert byod["byod_report"] is None
    refusal = byod["byod_refusal"]
    assert refusal["refused"] is True and refusal["message"].startswith("boxes.csv row 2: box")
    assert not stale.exists()


def test_other_byod_failures_still_raise(byod, tmp_path):
    real_reader = byod["read_byod_zip"]

    def broken_reader(_path):
        raise OSError("disk full")

    byod.update({"USE_BYOD": True, "BYOD_ZIP_PATH": "x.zip", "read_byod_zip": broken_reader})
    with pytest.raises(OSError, match="disk full"):
        run(byod, "cnt-byod-run")

    def broken_model(_byod):
        raise RuntimeError("CUDA out of memory")

    path = _zip(tmp_path / "ok.zip", *_valid_tables(), _images(6))
    byod.update(
        {"BYOD_ZIP_PATH": str(path), "read_byod_zip": real_reader, "run_byod": broken_model}
    )
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        run(byod, "cnt-byod-run")


def test_only_the_archive_contract_error_is_caught():
    tree = ast.parse(source("cnt-byod-run"))
    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
    assert [ast.unparse(h.type) for h in handlers] == ["BYODError"]
    try_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Try))
    assert "run_byod" not in ast.unparse(ast.Module(body=try_node.body, type_ignores=[]))
    helpers = ast.parse(source("cnt-byod-helpers"))
    byod_error = next(
        n for n in helpers.body if isinstance(n, ast.ClassDef) and n.name == "BYODError"
    )
    assert [ast.unparse(b) for b in byod_error.bases] == ["ValueError"]


def test_refusal_through_the_carried_runner_has_no_traceback(tmp_path):
    process = subprocess.Popen(
        [sys.executable, "-u", str(RUNNER)],
        cwd=tmp_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    def request(cell_source: str) -> tuple[list[dict], dict]:
        process.stdin.write(
            json.dumps({"op": "exec", "name": "test", "source": cell_source}) + "\n"
        )
        process.stdin.flush()
        events = []
        while True:
            event = json.loads(process.stdout.readline())
            if event["t"] == "done":
                return events, event
            events.append(event)

    try:
        assert json.loads(process.stdout.readline())["t"] == "ready"
        prelude = (
            "import hashlib\nimport numpy as np\nfrom pathlib import Path\n"
            "WORK_ROOT=Path('.')\nOUTPUT_ROOT=Path('outputs')\n"
            "def pixel_digest(im):\n    return hashlib.sha256(im.tobytes()).hexdigest()\n"
        )
        assert request(prelude)[1]["ok"]
        assert request(source("cnt-byod-helpers"))[1]["ok"]
        bad = _bad_zip(tmp_path / "bad.zip")
        events, done = request(
            f"USE_BYOD=True\nBYOD_ZIP_PATH={str(bad)!r}\n" + source("cnt-byod-run")
        )
    finally:
        process.stdin.close()
        process.wait(timeout=30)
    assert done["ok"] and done["error"] is None
    text = "".join(e.get("text", "") for e in events if e["t"] == "out")
    assert text.count("BYOD archive refused: boxes.csv row 2") == 1 and "Traceback" not in text


# -------------------- CNT2-M1 / CNT2-M2 provisional option B: the protocol is disclosed


PROTOCOL_CELLS = (
    "4fb28422",
    "1569a70e",
    "0fbbd3fb",
    "0f5fa7bd",
    "cnt-notice-19",
    "305eacac",
    "d976fb4f",
)


def test_section_14_describes_the_two_phrase_query():
    s14 = section("# 14.")
    assert "target phrase only" not in s14
    assert "[target_label, distractor_label]" in s14 and "best phrase is the target" in s14
    # the query the diagram describes is the one the code sends
    for call_cell in ("747c9153", "a147fce8"):
        assert source(call_cell).count('[r["target_label"],r["distractor_label"]]') == 2


@pytest.mark.parametrize("cell_id", PROTOCOL_CELLS)
def test_distractor_name_asymmetry_is_disclosed_where_the_comparison_is_read(cell_id):
    text = source(cell_id).lower()
    assert "distractor's name" in text or "every labelled phrase" in text, cell_id
    assert "target name only" in text or "never receive" in text or "never given" in text, cell_id


@pytest.mark.parametrize("cell_id", ("4fb28422", "0fbbd3fb", "0f5fa7bd", "305eacac", "d976fb4f"))
def test_tuned_versus_fixed_threshold_is_disclosed(cell_id):
    text = source(cell_id)
    assert "fixed upstream threshold of 0.23" in text, cell_id
    assert "selected on validation" in text or "selected on your validation" in text, cell_id


def test_wording_exercise_c_makes_sufficiency_conditional():
    """Wording test: Exercise C ties "sufficient" to a tolerance and the shared query ceiling."""
    answer = source("d976fb4f")
    exercise_c = answer[answer.index("### Exercise C") : answer.index("### Exercise D")]
    assert "as in these flat synthetic scenes" not in exercise_c
    assert "tolerance" in exercise_c and "point F1" in exercise_c
    assert "CountGD shares the roughly 900-query" in exercise_c
    assert "hypotheses" in exercise_c


def test_wording_exercise_b_requires_positive_and_negative_scenes():
    """Wording test: a presence threshold needs both scene kinds, a rule and both error rates."""
    answer = source("d976fb4f")
    exercise_b = answer[answer.index("### Exercise B") : answer.index("### Exercise C")]
    for phrase in (
        "positive scenes",
        "negative scenes",
        "image-level rule",
        "missed presence",
        "false presence",
    ):
        assert phrase in exercise_b, phrase


def test_wording_conclusions_are_worded_under_this_protocol():
    """Wording test (protocol disclosures of CNT2-M1/M2)."""
    assert "under this protocol" in source("cnt-notice-19")
    assert "only keep boxes labelled with the target phrase, so" not in source("cnt-notice-19")
    assert "protocol differences of section 14" in source("guided-03")


def test_frozen_record_states_both_protocol_choices():
    freeze = source("918e4e83")
    assert '"threshold_selection":"fixed upstream value; not selected on validation"' in freeze
    assert '"text_prompt":"target name only"' in freeze
    assert '"detector_as_counter_protocol":{' in freeze
    compile(freeze, "918e4e83", "exec")


def test_owner_decisions_are_recorded_as_provisional():
    entry = next(
        r
        for r in NB["metadata"]["dimer"]["review_revisions"]
        if "findings_disclosed_provisionally" in r
    )
    assert entry["date"] == "2026-10-13" and entry["computation_changed"] is False
    assert set(entry["findings_disclosed_provisionally"]) == {"CNT2-M1", "CNT2-M2"}
    assert all(
        "owner decision pending" in v for v in entry["findings_disclosed_provisionally"].values()
    )


# -------------------- CNT2-m2 .. m6


def test_section_6_names_which_scorer_scores_which_system():
    s6 = source("82efebb0")
    assert "common counting evaluator" not in s6
    assert "`countgd_pipeline.metrics`" in s6 and "template matcher" in s6
    assert "point matchers differ" in s6


def test_countgd_box_f1_is_marked_not_comparable():
    for cell_id in ("cnt-notice-12", "0f5fa7bd"):
        text = source(cell_id)
        assert "CountGD's box F1 is not comparable" in text and "2 x 2 px boxes" in text, cell_id
    assert "comparable between the two detectors only" in source("491b58e6")


def test_section_18_promises_a_countgd_picture_only_if_one_is_drawn():
    """Invariant: s18 asks for CountGD qualitative output only when a figure cell draws CountGD."""
    s18 = source("0f5fa7bd")
    figure_cells = [c["id"] for c in CODE if "savefig" in "".join(c["source"])]
    draws_countgd = any("countgd" in "".join(CELLS[i]["source"]).lower() for i in figure_cells)
    if not draws_countgd:
        assert "qualitative scene output" not in s18
        assert "No cell draws CountGD's predicted points" in s18 and "unverified" in s18


# -------------------- Codex follow-up (CX1-CX10)


def _generator(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return run(base_namespace(), "daa99d90", "d5aa425c")


def test_generator_limits_stated_in_section_4_hold_for_every_scene(tmp_path, monkeypatch):
    ns = _generator(tmp_path, monkeypatch)
    records = ns["train_records"] + ns["validation_records"] + ns["test_records"]
    for r in records:
        t_colour, t_shape = r["target_label"].split()
        d_colour, d_shape = r["distractor_label"].split()
        assert t_colour != d_colour and t_shape != d_shape, r["id"]  # differ in both attributes
        assert r["target_count"] <= 40 and r["distractor_count"] <= 20  # far below ~900 queries
        boxes = r["target_boxes"] + r["distractor_boxes"]
        # objects are separated: centres more than the two half-sizes plus 3 px apart
        shapes = [
            ((b[0] + b[2] - 1) / 2, (b[1] + b[3] - 1) / 2, (b[2] - b[0] - 1) / 2) for b in boxes
        ]
        for i, (ax, ay, a_size) in enumerate(shapes):
            for bx, by, b_size in shapes[i + 1 :]:
                assert (ax - bx) ** 2 + (ay - by) ** 2 > (a_size + b_size + 3) ** 2, r["id"]
    text = source("fc40989e")
    assert "**both** colour and shape" in text and "heavy overlap is not tested" in text


def test_splits_are_image_disjoint_but_not_category_disjoint(tmp_path, monkeypatch):
    ns = _generator(tmp_path, monkeypatch)
    splits = ("train", "validation", "test")
    pixels = {s: {r["pixel_sha256"] for r in ns[f"{s}_records"]} for s in splits}
    assert not pixels["train"] & pixels["test"] and not pixels["validation"] & pixels["test"]
    labels = {s: {r["target_label"] for r in ns[f"{s}_records"]} for s in ("train", "test")}
    assert labels["test"] & labels["train"]  # shared concepts: not category-disjoint
    assert "image-disjoint, not category-disjoint" in source("62e90abd")
    assert "image-disjoint, not category-disjoint" in source("4fb28422")


def test_exemplar_counting_is_disclosed_as_assisted(tmp_path, monkeypatch):
    ns = _generator(tmp_path, monkeypatch)
    for r in ns["test_records"]:  # the exemplars are scored targets of the same scene
        assert all(e in r["target_boxes"] for e in r["exemplars"])
    assert "assisted counting" in source("1dfd2c62")


def test_worked_example_numbers_follow_the_notebook_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns = run(base_namespace(), "daa99d90", "d5aa425c", "843fb01b", "83fbdc46", "f5ee4325")
    before_nms = ns["prf"]([{"tp": 2, "fp": 2, "fn": 1}])
    after_nms = ns["prf"]([{"tp": 2, "fp": 1, "fn": 1}])
    assert round(before_nms["precision"], 2) == 0.50 and round(before_nms["recall"], 2) == 0.67
    assert round(before_nms["f1"], 2) == 0.57
    assert round(after_nms["precision"], 2) == round(after_nms["recall"], 2) == 0.67
    example = source("62e90abd")
    assert "0.50" in example and "≈ 0.67" in example and "≈ 0.57" in example and "**+1**" in example
    assert "have seen basic object detection" not in source("guided-00")


def test_counting_tables_add_signed_error_and_point_precision_recall_without_dropping_columns():
    """Invariant: existing columns stay; new display columns come from values already computed."""
    for cell_id in ("ec46acb1", "92c772c5", "b9b5ef65"):
        code = source(cell_id)
        for existing in ('"mae"', '"rmse"', '"nae"', '"point_f1"', '"box_f1"'):
            assert existing in code, (cell_id, existing)
        for added in ('"mean_signed_error"', '"point_precision"', '"point_recall"'):
            assert added in code, (cell_id, added)
        assert ".evaluate(" not in code.split('"point_recall"')[-1]  # no new model call after them
    summary = source("491b58e6")
    assert '"test_point_f1":row.get("point_f1")' in summary
    assert '"test_box_f1":row.get("box_f1")' in summary


def test_export_records_dependency_and_evaluator_identity():
    export = source("43953ffd")
    for key in (
        '"notebook_identity"',
        '"dependency_lock_sha256"',
        '"cell_runner_sha256"',
        '"evaluation_identity"',
        '"metrics_git_blob":COUNTGD_SOURCE["metrics.py"][0]',
        '"point_radius"',
        '"point_assignment"',
        '"box_assignment"',
    ):
        assert key in export, key
    # the lock the export hashes is the one section 2 installs from
    assert 'LOCK_PATH = RUNNER_ROOT / "requirements.lock.txt"' in source("ef509873")


def test_section_11_names_the_carrier_commit_the_staging_cell_pins():
    tree = ast.parse(source("083028a5"))
    commit = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and getattr(n.targets[0], "id", None) == "COUNTGD_CARRIER_COMMIT"
    )
    assert f"@{commit[:7]}`" in source("005fe167")
    assert NB["metadata"]["dimer"]["carried_sources"]["countgd_pipeline"]["commit"] == commit


def test_wording_density_and_distractor_tables_are_associations():
    """Wording test: section 19 reads as hypothesis-generating, not causal."""
    notice = source("cnt-notice-19")
    assert "is limited by crowding" not in notice
    assert "together mean the system is counting" not in notice
    assert "**associated**" in notice and "**consistent with**" in notice
    assert "unverified" in notice and "colour and shape" in notice
    assert "not diagnoses" in source("3e058b9e")
    assert "never learned" not in source("cnt-notice-12") + source("0f5fa7bd")
    assert "is adding extra objects" not in source("cnt-notice-12")
    assert "compensating category/localization error" not in source("ebafe76d")
    assert "no bound" in source("82efebb0")
    assert "small point-F1 differences" not in source("82efebb0")


def test_export_names_the_current_workshop_revision():
    export = source("43953ffd")
    exported = re.search(r'"workshop_revision":"([^"]+)"', export).group(1)
    assert exported == NB["metadata"]["workshop_revision"] == "0.3.0-candidate"
    latest = [
        r["workshop_revision"]
        for r in NB["metadata"]["dimer"]["review_revisions"]
        if "workshop_revision" in r
    ][-1]
    assert exported == latest
    assert '"byod_refusal":globals().get("byod_refusal")' in export


def test_use_notice_precedes_the_first_code_cell():
    first_code = next(i for i, c in enumerate(NB["cells"]) if c["cell_type"] == "code")
    notices = [
        i
        for i, c in enumerate(NB["cells"][:first_code])
        if c["cell_type"] == "markdown"
        and re.search(r"^> \*\*⚠️ Notice\.\*\*", "".join(c["source"]), re.M)
        and 'provided "as is", without warranty' in "".join(c["source"])
    ]
    assert notices == [0]
    assert "[!WARNING]" not in "".join(NB["cells"][0]["source"])
