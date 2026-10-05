"""Regression test for the 2026-10-05 fleet-sweep check of countgd_object_counting_colab (SWP-A).

The sweep flagged one `assert` whose line names a metric. It is a contract-integrity check, kept as a hard check;
this test pins that no `assert` in the generated notebook compares a model-quality metric with a baseline or a
frozen model, and that the contract check is still present.
"""
# ruff: noqa: E501  -- the test quotes a notebook source line in full

from __future__ import annotations

import json
import re
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "tutorials" / "countgd_object_counting_colab.ipynb"
CONTRACT_ASSERT = "assert abs(adapted_val['mae'] - val_history[adapt_result['best_epoch']]) < 1e-6  # the kept epoch re-scores to the number that selected it"
QUALITY = re.compile(r"\b(accuracy|macro_f1|auroc|f1|mae|rmse|mse|iou|map|precision|recall|log_loss)\b")


def _code() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_swp_a_no_quality_assert_remains():
    code = _code()
    assert code.count(CONTRACT_ASSERT) == 1  # reload parity / selection consistency stays a hard check
    asserts = [line.strip() for line in code.splitlines() if line.strip().startswith("assert ")]
    comparisons = [line for line in asserts if re.search(r"[<>]", line) and QUALITY.search(line)]
    # Every metric-named comparison is the contract check, never `metric > baseline` or `metric > frozen`.
    assert all(line == CONTRACT_ASSERT for line in comparisons), comparisons
    assert not [line for line in asserts if re.search(r"(baseline|floor|frozen)\w*\[['\"]\w+['\"]\]\s*[<>]|[<>]\s*(baseline|floor|frozen)\w*\[", line)]
