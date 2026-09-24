# ruff: noqa: E501  -- markers are copied verbatim from the notebook
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "countgd_object_counting_colab.ipynb"
REGISTRY = ROOT / "tutorials" / "README.md"
LOCKFILE = ROOT / "requirements.lock.txt"

EXPECTED_OUTPUTS = {
    "countgd_object_counting_train.csv",
    "countgd_object_counting_input_manifest.json",
    "countgd_object_counting_evaluation_report.json",
    "countgd_object_counting_demo.json",
    "countgd_object_counting_adapter",
    "countgd_object_counting_result.json",
    "provenance.json",
}


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source_text(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def _normalized_source(notebook: dict) -> str:
    return " ".join(_source_text(notebook).split())


def test_release_notebook_declares_e2e_profile() -> None:
    notebook = _load_notebook()
    dimer = notebook["metadata"]["dimer"]
    assert dimer["notebook_profile"] == "E2E"
    assert dimer["notebook_spec"] == "2.0"
    assert dimer["standalone"] is True  # NOTEBOOK_SPEC 2.0 §4; parity in test_notebook_parity.py

    registry = REGISTRY.read_text(encoding="utf-8")
    assert "countgd_object_counting_colab.ipynb" in registry
    assert "`E2E`" in registry


def test_release_notebook_has_required_learning_contract_markers() -> None:
    source = _normalized_source(_load_notebook())
    required = (
        "bounded counting fine-tune",
        "count, one box and one point per counted object",
        "object detection or segmentation as a product",
        "synthetic counting scenes with known object boxes",
        "count error",
        "point localisation",
        "box IoU",
        "mean-count baseline",
        "template matcher",
        "audits it **statically**",
        "no dispersion estimate",
        "## Interpretation and limits",
    )
    for marker in required:
        assert marker in source, marker


def test_release_notebook_has_gated_byod_path() -> None:
    source = _source_text(_load_notebook())
    assert "USE_BYOD = False" in source
    assert "files.upload()" in source
    assert "records = load_byod_dataset(byod_zip)" in source
    assert "splits = split_dataset(records, seed=SPLIT_SEED)" in source


def test_release_notebook_exercises_the_counting_and_adaptation_contracts() -> None:
    source = _source_text(_load_notebook())
    for marker in (
        "splits = build_synthetic_dataset(SYNTHETIC_SPLIT)",
        "corpus_files = fetch_corpus(cache_dir='weights/fsc147-subset')",
        "fsc_splits = build_sample_dataset(corpus, seed=SPLIT_SEED)",
        "disjoint = check_split_disjoint(splits)",
        "demo = synthetic_scene(DEMO_SEED)",
        "demo_frozen = {mode: pipe.count(demo['image'], **kwargs)['results'][0] for mode, kwargs in PROMPTS.items()}",
        "frozen_syn = pipe.evaluate(test_records)",
        "frozen_fsc = pipe.evaluate(fsc_test) if fsc_test else None",
        "'mean_count': mean_count_baseline(train_records, test_records)",
        "'template_matching': template_matching_baseline(test_records)",
        "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE,",
        "assert val_history[adapt_result['best_epoch']] <= val_history[0]",
        "adapted_syn = pipe.evaluate(test_records)",
        "pipe.save_artifact(artifact_dir,",
        "reloaded = CountGDPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, tokenizer_dir=TOKENIZER_WEIGHTS_DIR, device=pipe.device)",  # noqa: E501
        "assert parity['identical_counts'] == parity['of']",
    ):
        assert marker in source, marker


def test_release_notebook_exports_every_demonstrated_capability() -> None:
    source = _source_text(_load_notebook())
    for filename in EXPECTED_OUTPUTS:
        assert f"outputs/{filename}" in source, filename
    assert "write_provenance('outputs/provenance.json', pipeline=pipe)" in source


def test_release_notebook_prints_runtime_and_immutable_identity() -> None:
    source = _source_text(_load_notebook())
    for marker in (
        "platform.python_version()",
        "torch.__version__",
        "transformers.__version__",
        "'device': str(pipe.device)",
        "'model_id': MODEL_ID",
        "'model_revision': MODEL_REVISION",
        "'weight_sha256': MODEL_SHA256",
        "'source_sha256': SOURCE_CKPT_SHA256",
        "'pickle_audit_sha256': PICKLE_AUDIT_SHA256",
    ):
        assert marker in source, marker


def test_release_notebook_loads_on_the_available_device() -> None:
    source = _source_text(_load_notebook())
    registry = REGISTRY.read_text(encoding="utf-8")
    lockfile = LOCKFILE.read_text(encoding="utf-8")

    assert "torch==2.14.0+cpu" in lockfile  # the CI reference environment stays CPU-only
    # The standalone carrier lets the pipeline pick CUDA when it is visible; CPU is the fallback.
    assert "pipe = CountGDPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, tokenizer_dir=TOKENIZER_WEIGHTS_DIR)" in source  # noqa: E501
    assert 'from_pretrained(device="cpu"' not in source
    assert "CUDA used automatically when present" in registry


def test_release_notebook_source_is_clean() -> None:
    notebook = _load_notebook()
    source = _source_text(notebook)

    for marker in ("TODO", "TBD", "FIXME", "@P:"):
        assert marker not in source, marker

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []
