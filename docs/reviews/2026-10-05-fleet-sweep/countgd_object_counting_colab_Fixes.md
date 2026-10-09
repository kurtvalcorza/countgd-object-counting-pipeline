# CountGD object-counting notebook — fleet-sweep fixes

**Source:** the 2026-10-05 static fleet sweep of `main` flagged this notebook for one `assert` line that compares a
metric-named value with a threshold. This is a targeted check; **no full Notebook Review Framework v1 review has been
done** in this cycle.
**Base:** `main` at `48d2384`.
**Notebook:** `tutorials/countgd_object_counting_colab.ipynb` (unchanged; `tools/build_notebook.py --check` is clean).
**Readiness:** **Verification pending.** Nothing in the notebook changed; a hosted one-pass Run all is still the gate.
**Status labels:** unchanged (`STATUS.md`, release status and `clean_runtime_evidence` are left for the maintainer).

The notebook already runs its model stages in the isolated `uv` environment (no in-kernel install, no restart guard)
and carries the guided layer, so SWP-R and SWP-G do not apply. The diff is limited to a regression test and this record.

## Findings

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| SWP-A | Did not reproduce | The flagged line is `assert abs(adapted_val['mae'] - val_history[adapt_result['best_epoch']]) < 1e-6  # the kept epoch re-scores to the number that selected it`. It checks that re-evaluating the kept epoch on the validation split reproduces the MAE that selected it (selection consistency); it does not compare a metric with a baseline or the frozen model. It is a contract-integrity check, not a model-quality comparison against a baseline or a frozen model, so it is kept as a hard check per the brief. No other `assert` in the notebook compares a quality metric. | none (a static regression test is added) | `test_swp_a_no_quality_assert_remains` |
| SWP-R | Not applicable | The model stages already run in the isolated environment; no `pip install` into the kernel and no restart request. | — | sweep: `ok(isolated)` |
| SWP-G | Not applicable | Guided layer present. | — | — |
| SWP-F / SWP-B | Not applicable | Not in the flagged cell; not examined (targeted fix). | — | — |

## User-visible changes

None. The notebook is byte-identical to `main`.

## Verification (offline; not clean-runtime evidence)

- No model stage ran here; no pretrained-inference evidence is claimed.
- `python tools/build_notebook.py --check`: OK. `python tools/validate_release_assets.py`: PASS. `ruff check`: clean.
- `pytest` (torch-free subset, -m 'not integration'; CI installs CPU torch from requirements.lock.txt): 38 passed, 1 skipped before → 39 passed, 1 skipped after.

## Remaining gates

- A hosted **Run all in one pass** in a fresh Colab runtime, then the REL12 BYOD run.
- A full Notebook Review Framework v1 review has not been done in this cycle.
