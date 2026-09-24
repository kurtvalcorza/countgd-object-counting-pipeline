# ruff: noqa: E501  -- assertions are kept on one line
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from countgd_pipeline import (
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    PICKLE_AUDIT_SHA256,
    SOURCE_CKPT_SHA256,
    build_provenance,
    write_provenance,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "countgd_object_counting_colab.ipynb"


def test_lock_parity_script_passes() -> None:
    subprocess.run([sys.executable, "scripts/check_lock.py"], cwd=ROOT, check=True)


def test_provenance_identity_and_semantics() -> None:
    record = build_provenance(include_runtime=False)
    model = record["model"]
    assert (model["id"], model["repo_type"], model["revision"], model["license"]) == (MODEL_ID, "space", MODEL_REVISION, "MIT")
    assert model["weight_sha256"] == MODEL_SHA256 and model["weight_file"] == "countgd.safetensors"
    assert model["derived_from"]["sha256"] == SOURCE_CKPT_SHA256 and model["derived_from"]["pickle_audit_sha256"] == PICKLE_AUDIT_SHA256
    assert record["inference"]["confidence_threshold"] == 0.23 and "grid_sample" in record["inference"]["deformable_attention"]
    assert "object-sized" in record["inference"]["training_box"]
    assert record["code"]["commit"] == "b6f362b3f5cd20db4a171faa410dfed8f2f466d8" and record["adapter"] is None


def test_provenance_records_actual_pipeline_metadata(tmp_path: Path) -> None:
    fake_pipeline = SimpleNamespace(checkpoint_path=tmp_path / "w", checkpoint_source="explicit_path", manifest_verified=True, weight_sha256="fake", weight_size_bytes=1234567, device="cuda:0")
    out_file = tmp_path / "provenance.json"
    write_provenance(out_file, pipeline=fake_pipeline)
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert (data["model"]["checkpoint_source"], data["model"]["manifest_verified"], data["model"]["weight_size_bytes"]) == ("explicit_path", True, 1234567)
    assert data["model"]["checkpoint_path"] == str(tmp_path / "w") and data["inference"]["device"] == "cuda:0"


def test_colab_notebook_is_json_and_python_cells_compile() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for index, cell in enumerate(code_cells):
        source = "".join(cell.get("source", []))
        if source.lstrip().startswith(("%", "!")):
            continue
        compile(source, f"{NOTEBOOK}#cell-{index}", "exec")
