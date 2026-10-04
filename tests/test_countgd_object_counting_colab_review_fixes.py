"""Regression tests for the row-16 review of tutorials/countgd_object_counting_colab.ipynb (CGD-M1..M4, CGD-m1..m3).

The learner cells are executed verbatim from the generated notebook JSON, in one namespace built by executing the
notebook's own carried-module cells, with the offline stand-in network of tests/test_pipeline.py in place of the
pinned checkpoint. These are interface/journey tests, not pretrained-inference evidence.
"""
# ruff: noqa: E501  -- notebook source lines and messages are kept whole
from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path
from typing import Any

import pytest

# Windows DLL load-order trap (FIX_PACKET addendum): import torch, if present, before any NumPy linear algebra.
with contextlib.suppress(ImportError):
    import torch  # noqa: F401

from test_pipeline import FakeCriterion, FakeModel, FakeTokenizer

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "countgd_object_counting_colab.ipynb"


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = _load_tool("build_notebook")
TEMPLATE = _load_tool("notebook_template").TEMPLATE


def _src(cell: dict) -> str:
    value = cell["source"]
    return "".join(value) if isinstance(value, list) else value


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(nb: dict) -> list[dict]:
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _markdown(nb: dict) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")


def _cell(nb: dict, marker: str) -> str:
    hits = [_src(c) for c in _code_cells(nb) if marker in _src(c)]
    assert len(hits) == 1, (marker, len(hits))
    return hits[0]


def _field(source: str, name: str, value: str) -> str:
    new, n = re.subn(rf"^{name} = .*$", lambda _m: f"{name} = {value}", source, count=1, flags=re.M)
    assert n == 1, name
    return new


# --- CGD-M1: isolated runtime, no in-kernel install --------------------------------------------------------------


def test_isolated_runtime_uses_the_fleet_uv_mechanism(nb: dict) -> None:
    code = _code_cells(nb)
    kernel = [i for i, c in enumerate(code) if "# dimer: kernel cell" in _src(c)]
    assert kernel == [0, 1], "only the install and router cells may run in the notebook kernel"
    install, router = _src(code[0]), _src(code[1])
    assert '"venv", "--quiet", "--managed-python", "--python", MANAGED_PYTHON' in install
    assert f"MANAGED_PYTHON = {TEMPLATE['managed_python']!r}" in install
    assert '"--require-hashes", "--only-binary", ":all:"' in install
    assert f"UV_SHA256 = {TEMPLATE['uv']['sha256']!r}" in install and f"UV_BYTES = {TEMPLATE['uv']['bytes']}" in install
    assert 'platform.machine() != "x86_64"' in install
    assert "sys.executable" not in install.split("SKIP_INSTALL = ", 1)[1], "nothing may be pip-installed into the kernel"
    assert "_ip.input_transformers_cleanup.append(_route_to_isolated_runtime)" in router
    assert 'DIMER_NOTEBOOK_CI_PREINSTALLED="1"' in router
    record = _src(code[2])
    assert record.startswith("# @title Infrastructure: record the runtime")
    assert "SKIP_INSTALL = os.environ.get('DIMER_NOTEBOOK_CI_PREINSTALLED') == '1'" in record
    assert nb["metadata"]["dimer"]["notebook_spec"] == "2.2" and nb["metadata"]["dimer"]["generated_from"]["generator"] == "build_notebook.py/2.1"


def _lock_entries(text: str) -> dict[str, tuple[str, set[str]]]:
    out = {}
    for block in re.split(r"\n(?=[a-z0-9])", text):
        match = re.match(r"([a-z0-9][a-z0-9._-]*)==(\S+)", block)
        if match:
            out[match.group(1)] = (match.group(2), set(re.findall(r"sha256:([0-9a-f]{64})", block)))
    return out


def test_carried_lock_equals_the_repository_lock_the_pins_and_the_workshop_files(nb: dict) -> None:
    install = _src(_code_cells(nb)[0])
    lock_text = (ROOT / TEMPLATE["lock"]).read_text(encoding="utf-8")
    assert f"LOCK_TEXT = r'''{lock_text}'''" in install
    assert f"LOCK_SHA256 = {hashlib.sha256(lock_text.encode('utf-8')).hexdigest()!r}" in install
    BUILD.check_lock(BUILD._pins(ROOT, TEMPLATE), lock_text)  # raises SystemExit on drift
    assert "--generate-hashes" in lock_text.splitlines()[1] and "x86_64-manylinux_2_28" in lock_text.splitlines()[1]
    # Every locked file is one the workshop's lock (already run on a Colab T4) also installs.
    ours = _lock_entries(lock_text)
    workshop = _lock_entries((ROOT / "tools" / "counting-workshop-requirements.lock").read_text(encoding="utf-8"))
    assert len(ours) == 48
    for name, (version, hashes) in ours.items():
        assert workshop[name][0] == version and hashes and hashes <= workshop[name][1], name


def test_router_needs_no_ipython_when_an_executor_runs_every_cell(nb: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setitem(sys.modules, "IPython", None)  # `import IPython` now raises ImportError
    namespace: dict[str, Any] = {"SKIP_INSTALL": True, "__name__": "__main__"}
    exec(compile(_src(_code_cells(nb)[1]), "router", "exec"), namespace)
    assert "Routing disabled" in capsys.readouterr().out


@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(nb: dict, monkeypatch: pytest.MonkeyPatch, real_google: bool) -> None:
    """chronos-2 Colab run: accelerate's find_spec("google.colab") raised on a spec-less worker stub."""
    namespace: dict[str, Any] = {"SKIP_INSTALL": True, "__name__": "__main__"}
    exec(compile(_src(_code_cells(nb)[1]), "router", "exec"), namespace)
    worker = namespace["_WORKER_SOURCE"]
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start : worker.index('_main = types.ModuleType("__main__")', start)]
    names = ("google", "google.colab", "google.colab.files")
    saved = {name: sys.modules[name] for name in names if name in sys.modules}
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    try:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules["google"] = fake_google if real_google else None
        monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
        exec(compile(shim, "worker-colab-shim", "exec"), {"os": __import__("os"), "sys": sys, "types": types, "_send": None, "_recv": None})
        for name in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(name)
            assert spec is not None and spec.name == name
        assert sys.modules["google.colab"].__path__ == [] and callable(sys.modules["google.colab.files"].upload)
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def test_release_records_no_longer_call_a_restart_dependent_run_a_pass() -> None:
    text = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "restarted_after_install_cell: true" in text and "not a one-pass `Run all`" in text
    assert "after the install cell, as designed" not in text and "is expected where the runtime's preinstalled" not in text
    for name in ("STATUS.md", "README.md", "tutorials/README.md"):
        body = (ROOT / name).read_text(encoding="utf-8")
        assert "Candidate" in body, name
    assert "Current status: **Candidate**" in (ROOT / "STATUS.md").read_text(encoding="utf-8")


# --- CGD-M4: guided layer and Infrastructure labels ----------------------------------------------------------------


def test_infrastructure_cells_are_titled_and_collapsed(nb: dict) -> None:
    infra = [c for c in _code_cells(nb) if "# dimer: kernel cell" in _src(c) or c.get("metadata", {}).get("dimer", {}).get("embedded_module") or _src(c).startswith("# @title Infrastructure: record")]
    assert len(infra) == 3 + len(TEMPLATE["modules"])
    for cell in infra:
        assert _src(cell).startswith("# @title Infrastructure:"), _src(cell)[:80]
        assert cell["metadata"].get("cellView") == "form"


def test_guided_layer_is_present_and_stale_text_is_gone(nb: dict) -> None:
    md = _markdown(nb)
    for marker in (
        "### Who this is for", "### How to use this notebook", "### Roadmap", "### Input → Model → Output",
        "## Before Section 4: glossary", "## Troubleshooting", "## Conclusion template",
        "#### What to notice (Section 5)", "#### What to notice (Section 6)", "#### What to notice (Section 8)",
        "## 10. Activity: change how much of the decoder trains", "**Predict:**", "**Change one thing:**", "**Observe:**", "**Explain:**",
    ):
        assert marker in md, marker
    for section in ("## 5. Count one scene", "## 6. Baselines", "## 8. Held-out evaluation"):
        block = md.split(section, 1)[1].split("#### What to notice", 1)[0]
        assert "**Predict first.**" in block, section
    assert md.count("<details><summary>") >= 4
    for stale in ("at least eight images", "to train the box head alone", "the build record measured MAE 7.42 →", "categories CountGD never saw", "the cell stops with a restart instruction"):
        assert stale not in md, stale
    # CGD-m2: every adapted-model headline names its runtime.
    assert "Build workstation CPU: kept epoch 3" in md and "Kaggle T4: kept epoch 2" in md


# --- the notebook's own cells on the stand-in network ------------------------------------------------------------


@pytest.fixture()
def ns(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The notebook namespace after Sections 1-3, with the stand-in network instead of the pinned checkpoint."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DIMER_NOTEBOOK_CI_PREINSTALLED", "1")
    monkeypatch.setitem(sys.modules, "google.colab", None)  # not Colab: `from google.colab import files` raises
    namespace: dict[str, Any] = {"__name__": "__main__"}
    code = _code_cells(nb)
    for cell in code[2:]:
        if cell.get("metadata", {}).get("dimer", {}).get("embedded_module") or _src(cell).startswith("# @title Infrastructure: record"):
            exec(compile(_src(cell), "setup", "exec"), namespace)
    base = namespace["CountGDPipeline"]

    class StandInPipeline(base):  # type: ignore[misc, valid-type]
        loads = 0

        @classmethod
        def from_pretrained(cls, **kwargs):
            cls.loads += 1
            return cls(FakeModel([(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)]), FakeCriterion(), FakeTokenizer(), device="cpu")

    namespace["CountGDPipeline"] = StandInPipeline
    namespace["pipe"] = StandInPipeline.from_pretrained()
    namespace.update(WEIGHTS_DIR=tmp_path / "weights" / "countgd", TOKENIZER_WEIGHTS_DIR=tmp_path / "weights" / "bert", snapshot={"files": []}, fetched=[])
    return namespace


def _run(ns: dict[str, Any], source: str, name: str = "cell") -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(source, name, "exec"), ns)
    return out.getvalue()


def _scenes(ns: dict[str, Any], n: int, label: str = "blue circle") -> list[dict]:
    out, seed = [], 5000
    while len(out) < n:
        scene = ns["synthetic_scene"](seed)
        seed += 1
        if scene["label"] == label:
            out.append(scene)
    return out


def _zip(path: Path, records: list[dict], *, kind: str = "boxes", folder: str = "images/", drop: str | None = None, extra: dict[str, bytes] | None = None) -> Path:
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=["id", "file", "label", "count", "exemplars", "points", "boxes"])
    writer.writeheader()
    with zipfile.ZipFile(path, "w") as archive:
        for i, record in enumerate(records):
            name = f"{folder}scene_{i:02d}.png"
            buffer = io.BytesIO()
            record["image"].save(buffer, format="PNG")
            if name != drop:
                archive.writestr(name, buffer.getvalue())
            writer.writerow({
                "id": f"img{i:02d}", "file": name, "label": record["label"], "count": record["count"],
                "exemplars": json.dumps(record["exemplars"]),
                "points": json.dumps(record["points"]) if kind == "points" else "",
                "boxes": json.dumps(record["boxes"]) if kind == "boxes" else "",
            })
        for member, payload in (extra or {}).items():
            archive.writestr(member, payload)
        archive.writestr(f"{folder}labels.csv", table.getvalue())
    return path


def _section4_byod(nb: dict, zip_path: Path | str) -> str:
    source = _cell(nb, "USE_BYOD = False  # @param")
    source = _field(source, "USE_BYOD", 'True  # @param {type:"boolean"}')
    return _field(source, "BYOD_ZIP_PATH", f"{str(zip_path)!r}")


def test_split_dataset_minimum_is_18_for_one_category(nb: dict, ns: dict[str, Any]) -> None:
    """CGD-M2: the stated minimum is the real one (default fractions, MIN_RECORDS images in every split)."""
    smallest = {}
    for n in range(12, 25):
        parts = ns["split_dataset"]([{**r, "id": f"r{i}"} for i, r in enumerate(_scenes(ns, n))], seed=42)
        smallest[n] = min(len(p) for p in parts.values())
    passing = [n for n, size in smallest.items() if size >= ns["MIN_RECORDS"]]
    assert passing == list(range(18, 25))
    assert "BYOD_MIN_IMAGES_ONE_CATEGORY = 18  #" in _cell(nb, "USE_BYOD = False  # @param")


def test_section4_names_the_short_split_and_the_fix(nb: dict, ns: dict[str, Any], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"validation gets 3\. Every split .* needs at least 4 images.*at least 18 images"):
        _run(ns, _section4_byod(nb, _zip(tmp_path / "17.zip", _scenes(ns, 17))))
    out = _run(ns, _section4_byod(nb, _zip(tmp_path / "18.zip", _scenes(ns, 18))))
    assert "'splits': {'test': 5, 'validation': 4, 'train': 9}" in out
    assert out.count("'rejected'") == 4 and "'accepted'" not in out


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("count_only", r"row 'img00' \(line 2\) has a count but neither points nor boxes.*add a points column"),
        ("missing", r"row 'img03' \(line 5\): file 'images/scene_03.png' is not in the zip"),
        ("broken", r"row 'img03' \(line 5\): 'images/scene_03.png' is not an image Pillow can decode"),
        ("dotdot", r"zip member '\.\./evil\.txt' has an absolute or parent-directory"),
        ("not_a_zip", r"BYOD needs a \.zip file"),
    ],
)
def test_section4_refuses_bad_byod_zips_with_the_row_and_the_fix(nb: dict, ns: dict[str, Any], tmp_path: Path, case: str, message: str) -> None:
    """CGD-M2 (count-only) and CGD-m1 (missing, undecodable, '..', wrong file type): ValueError naming row/file and fix."""
    scenes = _scenes(ns, 18)
    if case == "count_only":
        path = _zip(tmp_path / "c.zip", scenes, kind="count")
    elif case == "missing":
        path = _zip(tmp_path / "m.zip", scenes, drop="images/scene_03.png")
    elif case == "broken":
        path = _zip(tmp_path / "b.zip", scenes, drop="images/scene_03.png", extra={"images/scene_03.png": b"not an image"})
    elif case == "dotdot":
        path = _zip(tmp_path / "d.zip", scenes, extra={"../evil.txt": b"x"})
    else:
        path = tmp_path / "labels.csv"
        path.write_text("id,file,label,count\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _run(ns, _section4_byod(nb, path))


def test_section4_without_a_path_off_colab_says_to_set_the_path(nb: dict, ns: dict[str, Any]) -> None:
    source = _field(_cell(nb, "USE_BYOD = False  # @param"), "USE_BYOD", 'True  # @param {type:"boolean"}')
    with pytest.raises(ValueError, match="BYOD_ZIP_PATH is empty, and this runtime has no upload dialog"):
        _run(ns, source)


def _sections_5_to_9(nb: dict, ns: dict[str, Any], *, epochs: int = 2, trainable_layers: int | None = None, only_7_to_9: bool = False) -> dict[str, str]:
    outs = {}
    if not only_7_to_9:
        outs["5"] = _run(ns, _cell(nb, "demo = synthetic_scene(DEMO_SEED)"), "S5")
        outs["6"] = _run(ns, _cell(nb, "frozen_syn = pipe.evaluate(test_records)"), "S6")
    s7 = _field(_cell(nb, "adapt_result = pipe.adapt("), "EPOCHS", f"{epochs}  # @param")
    if trainable_layers is not None:
        s7 = _field(s7, "TRAINABLE_LAYERS", f"{trainable_layers}  # @param")
    outs["7"] = _run(ns, s7, "S7")
    outs["8"] = _run(ns, _cell(nb, "adapted_syn = pipe.evaluate(test_records)"), "S8")
    outs["9"] = _run(ns, _cell(nb, "pipe.save_artifact(artifact_dir"), "S9")
    return outs


def test_points_only_byod_runs_sections_4_to_9_and_writes_per_image_predictions(nb: dict, ns: dict[str, Any], tmp_path: Path) -> None:
    """CGD-M2 (points-only) and CGD-m3 (per-image predictions) on the stand-in network."""
    _run(ns, _section4_byod(nb, _zip(tmp_path / "p.zip", _scenes(ns, 18), kind="points")))
    assert all("boxes" not in r and r["points"] for r in ns["train_records"])
    _sections_5_to_9(nb, ns)
    assert ns["frozen_syn"]["boxes"]["n"] == 0 and ns["frozen_syn"]["localisation"]["n"] == len(ns["test_records"])
    result = json.loads(Path("outputs/countgd_object_counting_result.json").read_text(encoding="utf-8"))
    assert result["byod_predictions"] == "outputs/countgd_object_counting_byod_predictions.json"
    predictions = json.loads(Path(result["byod_predictions"]).read_text(encoding="utf-8"))
    assert [p["id"] for p in predictions["images"]] == [r["id"] for r in ns["test_records"]]
    assert all(set(p) == {"id", "label", "gold", "count", "boxes", "points", "scores"} and len(p["boxes"]) == p["count"] for p in predictions["images"])
    assert result["reload_parity"]["identical_counts"] == result["reload_parity"]["of"] == 4


def test_frozen_pipeline_keeps_an_unadapted_model_and_reloads_an_adapted_one(ns: dict[str, Any], nb: dict) -> None:
    _run(ns, _cell(nb, "USE_BYOD = False  # @param").split("os.makedirs('outputs', exist_ok=True)", 1)[0])  # helper definitions only
    pipe = ns["pipe"]
    loads = ns["CountGDPipeline"].loads
    assert ns["frozen_pipeline"](pipe) is pipe and ns["CountGDPipeline"].loads == loads
    pipe.adapter = {"best_epoch": 1, "trainable_layers": 2}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fresh = ns["frozen_pipeline"](pipe)
    assert fresh is not pipe and fresh.adapter is None and ns["CountGDPipeline"].loads == loads + 1
    assert "'reloaded_pinned_base': True" in out.getvalue()


def test_rerunning_section7_fine_tunes_the_pinned_base_and_the_export_reloads(nb: dict, ns: dict[str, Any], tmp_path: Path) -> None:
    """CGD-M3: the suggested TRAINABLE_LAYERS = 0 re-run starts from the frozen model and keeps reload parity."""
    _run(ns, _section4_byod(nb, _zip(tmp_path / "b.zip", _scenes(ns, 18))))
    _sections_5_to_9(nb, ns, epochs=2)
    first_epoch0 = ns["val_history"][0]
    assert ns["pipe"].adapter is not None
    loads = ns["CountGDPipeline"].loads
    outs = _sections_5_to_9(nb, ns, epochs=1, trainable_layers=0, only_7_to_9=True)
    assert "'reloaded_pinned_base': True" in outs["7"] and ns["CountGDPipeline"].loads >= loads + 1
    assert ns["val_history"][0] == first_epoch0
    assert ns["adapt_result"]["trainable_layers"] == 0
    manifest = json.loads(Path("outputs/countgd_object_counting_adapter/manifest.json").read_text(encoding="utf-8"))
    assert manifest["adapter"]["trainable_layers"] == 0 and ns["parity"]["identical_counts"] == ns["parity"]["of"]


def test_byod_rerun_from_section4_scores_the_frozen_model(nb: dict, ns: dict[str, Any], tmp_path: Path) -> None:
    """CGD-M3 (BYOD): Sections 5 and 6 on an adapted kernel reload the base, so 'frozen' means the published model."""
    _run(ns, _section4_byod(nb, _zip(tmp_path / "b.zip", _scenes(ns, 18))))
    _run(ns, _cell(nb, "frozen_syn = pipe.evaluate(test_records)"))
    fresh = {k: ns["frozen_syn"][k] for k in ("mae", "rmse")}
    ns["pipe"].adapter = {"best_epoch": 1, "trainable_layers": 2}  # as after Section 7
    out = _run(ns, _cell(nb, "frozen_syn = pipe.evaluate(test_records)"))
    assert "'reloaded_pinned_base': True" in out and ns["pipe"].adapter is None
    assert {k: ns["frozen_syn"][k] for k in ("mae", "rmse")} == fresh


def test_the_trained_set_does_not_contain_the_real_box_head() -> None:
    """The real network registers the shared box head first under transformer.decoder.bbox_embed, so the package's
    `bbox_embed.0.` prefix never matches it: 0 layers train only the decoder norm (512 parameters, 2 tensors)."""
    import torch

    from countgd_pipeline import CountGDPipeline

    model = FakeModel()
    model.transformer.decoder.bbox_embed = model.bbox_embed  # registered inside the transformer, as in GroundingDINO
    shared = torch.nn.Module()
    shared.transformer = model.transformer
    shared.bbox_embed = model.bbox_embed  # registered after the transformer: named_parameters keeps the first name
    pipe = CountGDPipeline(shared, FakeCriterion(), FakeTokenizer(), device="cpu")
    names = pipe._trainable_names(0)
    assert names == ["transformer.decoder.norm.weight", "transformer.decoder.norm.bias"]
    assert sum(p.numel() for n, p in shared.named_parameters() if n in names) == 512
