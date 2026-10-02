# DIMER Open-Vocabulary Detection and Counting Notebook — Review

**Verdict: Needs revision**
**Review date:** 2 October 2026
**Repository:** `kurtvalcorza/countgd-object-counting-pipeline`
**Notebook:** `tutorials/DIMER_Open_Vocabulary_Detection_and_Counting_Workshop.ipynb`
**Reviewed commit:** `413d558301f0c6e70a9c42870cee2497dfc97bfb` (`main`, confirmed against the GitHub API at review time)
**Notebook Git blob:** `b0ddeb6e8abe4be3c76166205147704112d46f4b` (file sha256 `9abdd8998cdc…d45824`)
**Finding prefix:** `CNT`
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (`ml-worker` `origin/main` `b1cfe13`); the notebook declares 2.1.

## Executive assessment

The comparative design is sound and carefully controlled: one deterministic 24/8/12 synthetic scene set with exact
target and distractor boxes, a notebook-owned detection evaluator (AP@[.50:.95], AP50/75, recall, duplicates) applied
identically to Grounding DINO Tiny and OWLv2, detector-as-counter thresholds selected on validation only, an explicit
freeze before the test split, CountGD read three ways (count error, point F1, box F1), two non-neural baselines, and
honest wording about score semantics and synthetic evidence. A maintainer Colab T4 `Run all` of the STANDARD tier on
2026-09-26 covers this revision's code cells exactly. No Blocker was found.

Five Majors remain. The default path downloads the CountGD package source from `raw.githubusercontent.com` at run
time (and `FULL` the Grounding DINO carrier), which NOTEBOOK_SPEC ST4 forbids; the notebook itself names inlining
these modules as its release gate (CNT-M1). The `USE_BYOD` and `BYOD_ZIP_PATH` form fields are read by no cell:
there is no BYOD branch at all, although DAT10 requires one and the notebook describes its contract (CNT-M2). The
`FULL` Grounding DINO "reload parity" compares the reloaded adapter with itself, and neither parity check fails on
a mismatch (CNT-M3). Section 19 promises density and distractor diagnostics "especially for exemplar-only CountGD",
but computes them only for the two detectors and never reads the distractor ratio (CNT-M4). The "one controlled
change" activity names no control and cannot be done without editing infrastructure code; the CountGD threshold grid
it could use is defined and never read (CNT-M5). Seven Minors cover unexported results, a summary without results,
a bundle that can pick up earlier runs' files, a platform-dependent dataset digest, and guided-layer gaps.

## 1. Review contract and evidence

| Item | Recorded value |
|---|---|
| Revision | `413d558` (`main` = GitHub API `commits/main` at review time); notebook blob `b0ddeb6e` |
| Spec version | NOTEBOOK_SPEC 2.1 declared (metadata, opening cell, validator); reviewed against 2.2, whose only additions are the GDL1–GDL15 `SHOULD`s |
| Profile / mode | `MULTI-CAPABILITY` / `WORKSHOP` |
| Audience | "Learners who can run Python cells in Colab/Jupyter and have seen basic object detection" (guided-00) |
| Prerequisites | Colab or Jupyter; a GPU runtime ("NVIDIA Tesla T4 or equivalent") |
| Supported runtime | Colab T4; tiers `STANDARD` (default: frozen comparison) and `FULL` (adds bounded Grounding DINO and CountGD adaptation, FSC-147 regression check, adapter reload) |
| Promised outcomes | twelve learning goals (md-02): open-vocabulary meaning, Grounding DINO vs OWLv2 under one contract, prompt sensitivity, duplicates and negative prompts, AP/recall, detection vs counting, MAE/RMSE/NAE, compensating errors via localisation, CountGD prompt modes, detector-as-counter, validation-owned selection and freeze, adaptation trade-offs; BYOD (controls cell, section 24) |
| Status | `Candidate`; `metadata.dimer.standalone: false`, `candidate_gate`: inline pinned carrier modules |

Scope: all 69 cells (38 markdown, 31 code), the CountGD carrier modules the notebook stages (identical to this
repository's `src/countgd_pipeline/` at the reviewed commit: all 9 pinned Git blobs equal `HEAD`), the Grounding DINO
carrier at `grounding-dino-detection-pipeline@216c43e` (read for API compatibility only), exported files, failure
messages, `docs/open-vocabulary-detection-and-counting-workshop-spec.md` §76–§80 (BYOD contract).

### Evidence actually obtained

- **Documented execution evidence:** `docs/release-verification.md` §"Supplemental … workshop" and
  `docs/execution-evidence/2026-09-26/…ipynb` (sha256 `d7cb37ee…cd79`, re-hashed here). Its 31 code cells equal this
  revision's code cells apart from Colab `# @title` lines (probe P13; 0 differences). It covers the `STANDARD`
  default path only (Colab T4, Python 3.13.15, torch 2.14.0+cu130). `FULL` and BYOD were never executed. Its saved
  outputs were inspected; execution was not repeated.
- **Direct execution (this review, Windows, CPU, anaconda Python 3.13, Pillow 11.3.0, no GPU, no model weights):**
  the controls, generator, sample-manifest and both metric cells (probes D01–D04): the 44 scenes reproduce the hosted
  per-split target/distractor totals exactly (train 541/303, validation 232/118, test 278/143), perfect detections
  score AP 1.0, and a count-correct, wrong-objects prediction gives MAE 0 with box F1 0.083. Repository checks:
  `pytest -m 'not integration'` 96 passed / 1 skipped (converted checkpoint absent), `ruff`, `validate_release_assets.py`,
  `build_notebook.py --check`, `check_lock.py` all pass.
- **Source inspection:** every cell; the carried CountGD `pipeline.py`, `samples.py`, `metrics.py`; the Grounding DINO
  carrier's `pipeline.py`.
- **Not verified:** any model behaviour on this host (no weights, no GPU); `FULL` (never executed anywhere); BYOD
  (does not exist).

### Journeys

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection | Clear capability framing, glossary, interpretation boundaries, troubleshooting. Gaps: what-to-notice notes at 2 of 7 principal results, exercises without sample answers, infrastructure not collapsed (m5, m6, m7); the activity is not doable as written (M5) |
| Clean default (`STANDARD`) | Documented execution evidence (Colab T4, identical code cells) + direct CPU execution of the model-free cells | Completes; but it fetches package source at run time (M1); summary has no results (m2); the dataset digest is not portable (m4) |
| Active learning | Source inspection | No runnable controlled change exists (M5); `FULL` is the only alternative configuration and has never run; its reload check is tautological (M3) |
| Reuse and recovery | Source inspection | BYOD absent despite its form fields (M2); no invalid-input path to examine; bundle may include an earlier run's files (m3) |

## 2. Separate judgments

**Technical correctness.** The shared evaluator, NMS, matching and metric code are correct on the cases probed
(D02, D03). Threshold selection, freeze and test ownership are implemented as described. Defects: run-time source
fetch (M1), a reload check that cannot fail (M3), dead controls (M2, M5), results computed but not exported (m1),
a non-portable dataset digest (m4), a bundle not scoped to the run (m3).

**Promise fulfilment.** Capabilities A, B and C execute as promised in `STANDARD`. Undelivered: BYOD (M2), the
CountGD half of the density/distractor diagnostic (M4), a controlled-change activity (M5), and `FULL`'s "fresh
adapter reload verification" for Grounding DINO (M3).

**Learner experience.** Strong conceptual material (sections 25–26, glossary, troubleshooting) sits after long
unlabelled infrastructure; principal results mostly appear without a prediction or a note on what to read, and the
exercises have no worked guidance. The hosted run shows learner-relevant patterns the notebook never comments on
(detector-as-counter MAE 1.25–1.5 vs CountGD text 4.67; combined prompting worse than text alone; Grounding DINO
returns 20.75 boxes per image for the absent "purple star").

**Spec conformance (separate from severity).** Unmet `MUST`s: ST4/RUN4/SRC12 (M1), DAT10–DAT19 and REL12 (M2),
VER1–VER5 in `FULL` (M3), RUN8 (m2), REL1 for `FULL`/BYOD (not executed). Unmet `SHOULD`s: EXE2 (M2), GDL5, GDL7–GDL11,
GDL15, UX12 (m5–m7).

## 3. Findings

### CNT-M1 — Major (ST4, RUN4, SRC12 MUST): the default path downloads DIMER package source at run time

**Cell/section:** code `083028a5` (section 11, default path) and `229ad112` (section 15, `FULL`).
**Observed issue:** both cells `urllib.request.urlopen` nine `countgd_pipeline` files (and in `FULL` four
`grounding_dino_detection_pipeline` files) from `raw.githubusercontent.com`, verify each Git blob, write them to
`work/carrier_src` and import them. The opening cell and `metadata.dimer.candidate_gate` name inlining them as the
release gate.
**Consequence:** the default path depends on GitHub being reachable and is nonconformant with ST4 ("MUST NOT fetch
DIMER repository scripts/modules from raw URLs"); the notebook cannot be promoted. The blob check makes it safe, not
conformant.
**Evidence:** source inspection; probe P01 lists both cells. Probe `carrier_pins`: the nine pinned CountGD blobs equal
this repository's `src/countgd_pipeline/` at `413d558`; the four Grounding DINO blobs exist at
`grounding-dino-detection-pipeline@216c43e`.
**Recommended correction:** carry the 13 modules verbatim in notebook cells (collapsed, labelled Infrastructure),
keep the existing Git-blob pins, verify each carried source against its pin before writing it, and remove `urllib`.
No model, data or evaluation contract changes.
**Acceptance check:** no code cell contains `raw.githubusercontent.com` or `urlopen`; for each of the 13 pinned
files, the carried source's Git blob SHA-1 and byte length equal the pin; `metadata.dimer.standalone` is `true`.

### CNT-M2 — Major (DAT10–DAT19, REL12 MUST; EXE2): the BYOD controls are read by no cell; there is no BYOD branch

**Cell/section:** controls `daa99d90` (`USE_BYOD`, `BYOD_ZIP_PATH`); section 24 (`305eacac`).
**Observed issue:** no code cell other than the controls cell references either field (probe P02). Section 24
describes a `scenes.csv` + `boxes.csv` + `images/` contract and privacy guidance, but nothing reads, validates or
scores such an archive. guided-00 tells the learner they may "optionally continue with BYOD".
**Consequence:** a learner who ticks `USE_BYOD` gets the sample run with no message; the capability's user-data path
does not exist (DAT10, REL12), and DAT12/DAT19 validation cannot be exercised.
**Evidence:** source inspection, probe P02.
**Recommended correction:** implement the documented contract (workshop spec §76–§80): read the zip member by member
(reject absolute paths, `..`, symlinks, oversized members, URLs), validate `scenes.csv`/`boxes.csv` with errors that
name the file, row and field, honour explicit splits or assign 60/20/20 with seed 42, then run the same frozen
stages (detection evaluator, validation-selected detector-as-counter, CountGD prompt modes, baselines) and export to
`outputs/byod/`. Upload dialog only when the path is empty. Demonstrate one rejection on the default path.
**Acceptance check:** with `USE_BYOD=True` and a valid zip, the branch reaches detection, detector-as-counter and
CountGD evaluation using the same helper functions as the sample path and writes `outputs/byod/`; each of ≥5 invalid
archives (missing column, unknown image, box outside image, >3 exemplars, traversal member) is refused before any
model call with a message naming the offending file and row; with `USE_BYOD=False` no BYOD code touches a model.

### CNT-M3 — Major (VER1–VER5 MUST, `FULL`): the Grounding DINO reload parity compares the reloaded adapter with itself

**Cell/section:** `06a56cd1` (section 22).
**Observed issue:** `a` and `b` are both `fresh_gd.predict_boxes(...)` on the same reloaded pipeline, so
`same_length` is always true and `max_score_diff` always 0. The in-memory adapted model (`gd_pipe`) was deleted in
`229ad112` without recording a reference output. The CountGD parity booleans are printed but never asserted.
**Consequence:** `FULL` reports reload verification for Grounding DINO that cannot detect a broken artifact, and a
CountGD mismatch would not stop the run. VER5 (distinguish loading from reproducing) is not met.
**Evidence:** source inspection, probe P03.
**Recommended correction:** record the adapted in-memory predictions on a fixed scene before `del gd_pipe`; compare
the fresh reload against them with an explicit tolerance; raise on any count/label/box/score mismatch for both models.
**Acceptance check:** the parity cell compares against a reference captured before the in-memory model is released,
uses a stated tolerance, and raises `RuntimeError` naming the model when parity fails; no comparison of an object
with itself remains.

### CNT-M4 — Major (promise; EVAL15): the density/distractor diagnostic omits CountGD and never reads the distractor ratio

**Cell/section:** section 19 (`3e058b9e`, `5fb1218c`); Exercise D (`d976fb4f`); learning goal 9.
**Observed issue:** the prose promises test MAE by density group and "error against the distractor/target ratio —
especially for exemplar-only CountGD". The code builds rows only for `detector_counter_test` (the two detectors),
stores `distractor_ratio` and never displays or analyses it.
**Consequence:** the notebook's explanation for exemplar-only over-counting (visual exemplars pick up distractors) is
asserted, never shown, although the run already holds the data (`countgd_test_results[...]["per_image"]`). In the
hosted run exemplar-only MAE is 11.67 against 11.9 distractors per test scene, the pattern the section should make
visible.
**Evidence:** source inspection, probe P04; hosted outputs of `92c772c5`/`5fb1218c`.
**Recommended correction:** add the CountGD modes to the per-scene rows from their `per_image` predictions; display
mean signed error by density and the relation between over-count and distractor count per system (for example, the
share of scenes whose prediction is closer to targets + distractors than to targets).
**Acceptance check:** the diagnostic table contains every CountGD prompt mode next to the detector counters, and a
second displayed table reports, per system, signed error and its relation to the scene's distractor count; both are
written to `outputs/counting/test/`.

### CNT-M5 — Major (GDL10, UX6): the "one controlled change" activity has no control and cannot be done as written

**Cell/section:** guided-01; controls `daa99d90` (`COUNTGD_THRESHOLD_GRID`).
**Observed issue:** the activity says "change one condition, rerun the relevant analysis" with no form field, cell
or rerun instruction. Adding distractors or changing the prompt mode requires editing generator or evaluation code.
`COUNTGD_THRESHOLD_GRID` is defined and never read (probe P05).
**Consequence:** the active-learning journey cannot be followed; a learner who edits a shared cell and reruns part of
the notebook can leave frozen outputs inconsistent with the test record.
**Evidence:** source inspection, probe P05.
**Recommended correction:** add an optional, off-by-default activity cell with one form-field choice, run on
validation scenes only, written to its own output directory, with a prediction prompt before it and a worked
interpretation after it; it must not modify frozen choices or test outputs.
**Acceptance check:** a `RUN_ACTIVITY` toggle (default `False`) and one choice field exist; with the toggle on, the
activity uses only validation records, writes only under `outputs/activity/`, and leaves `frozen_experiment.json`
unchanged (asserted by digest); `COUNTGD_THRESHOLD_GRID` is read.

### CNT-m1 — Minor (OUT1, OUT4): some computed results never reach the export

`FULL`'s adapted Grounding DINO test (`full_grounding_test`, `21534efa`) is printed but absent from
`experiment_manifest.json` and `workshop_summary.json`; the new-scene result (`new_countgd_result`, `f2ca43e5`) is
printed only, and `outputs/new_data/` stays empty. The new-scene MAE (6.67 in the hosted run, on three scenes) has no
note on how to read it. *Evidence:* source inspection, probe P06. *Correction:* export both (new-scene per-scene
predictions to `outputs/new_data/`) and add a note. *Acceptance check:* the manifest has `full_adaptation.grounding_dino_test`
and `new_scenes`; `outputs/new_data/new_scene_predictions.json` is written on the default path.

### CNT-m2 — Minor (RUN8 MUST): the completion summary contains no results

`491b58e6` shows tier, digests, thresholds and paths but no metric, no bundle path, and no indication of which
optional branches ran. *Evidence:* hosted output; probe P07. *Correction:* add headline test results (best detector
AP50, each counter's MAE and box F1), the bundle path and branch status. *Acceptance check:* the summary displays test
AP50 per detector, MAE and box F1 per counting system, the bundle path, and `byod`/`activity`/`full` status.

### CNT-m3 — Minor (OUT, provenance): the report bundle zips whatever is under `outputs/`

`shutil.make_archive` archives the whole output directory, so files left by an earlier run (a previous `FULL` run's
adapters, an earlier BYOD) enter a later `STANDARD` bundle beside its manifest. *Evidence:* source inspection, probe
P08. *Correction:* record a run start time in the controls cell and bundle only files written by this run, listing
anything excluded. *Acceptance check:* a stale file older than the run start is excluded from the bundle and named.

### CNT-m4 — Minor (ENV9, OUT9): the dataset digest depends on PNG encoder bytes

`pixel_sha256` hashes the PNG-encoded bytes, which vary with the zlib build. On this host the 44 scenes reproduce the
hosted per-split counts exactly, yet `dataset_sha256` differs from the hosted `53b17b4c…` (probe D01; same Pillow
11.3.0, different platform). A learner comparing digests would conclude the data changed. *Correction:* hash the
decoded RGB pixels with the image size. *Acceptance check:* `pixel_sha256` equals `sha256(size + mode + tobytes())`
and is identical whether or not the image round-trips through PNG.

### CNT-m5 — Minor (GDL11, GDL15, GDL5): infrastructure, terminology and structure

No code cell is collapsed (`cellView: form`) or titled Infrastructure; section 11 (carrier bootstrap) is not marked
Infrastructure; the controls cell is titled "Workshop controls" and the final line says "workshop complete"
(GDL15); "Try it yourselfs" (typo); two Troubleshooting sections (`ebafe76d`, `guided-04`); goals 3 and 12 use
"understand" (GDL5). *Evidence:* probe P09. *Acceptance check:* every setup/manifest/carrier/export cell is
collapsed and starts `# @title Infrastructure:`; one Troubleshooting section; no "workshop" for the artifact; no
"understand" in the goals; typo gone.

### CNT-m6 — Minor (GDL7–GDL9, UX4): principal results lack predictions, what-to-notice notes and worked answers

Only sections 17 and 18 carry "What to notice"; the validation detector comparison, prompt sensitivity, CountGD
prompt modes and detector-as-counter thresholds appear without a prediction or a reading guide. Exercises A, C, D
and E have no sample answer; the single `<details>` block is generic. The validation tables carry a raw
`per_phrase_ap50` dict column that renders truncated. *Evidence:* probe P10; hosted outputs. *Acceptance check:* each
of the five principal results has a prediction prompt before it and a What-to-notice note after it; each exercise has
a collapsible worked answer; the dict column is shown as its own per-phrase table.

### CNT-m7 — Minor (GDL1, UX12): no download size or duration expectation

The opening states a T4 but not that the default path downloads ≈2.6 GB of weights (689 MB + 620 MB + 1.25 GB) and
converts the CountGD checkpoint once (≈0.94 GB written), nor that no measured duration exists. *Evidence:* probe P11.
*Acceptance check:* the opening states the download and conversion sizes and labels any duration as unmeasured or
as measured with its environment.

### Suggestions (optional, not release requirements)

- **CNT-S1:** report dispersion (e.g. a bootstrap interval over the 12 test scenes) beside the point metrics.
- **CNT-S2:** record peak VRAM and per-stage wall time in the manifest (already listed as a candidate gate).
- **CNT-S3:** add a BYOD `FULL` (adaptation on user data); the multi-capability profile only requires inference (DAT15).
- **CNT-S4:** a short note explaining why text+exemplar prompting scored worse than text alone in the hosted run.

## 4. Positive findings and non-findings

- Split ownership is clean: seeds 1000/2000/3000 ranges, pixel-level leak check, thresholds chosen on validation only,
  explicit freeze record before any test call, new scenes 4000–4002 outside every range.
- The evaluator is shared and model-agnostic; the primary comparison uses raw outputs and NMS only as a labelled
  diagnostic, so no model gets its own post-processing advantage.
- Score semantics are stated correctly (section 25: scores are not probabilities, not comparable across families).
- Integrity: every model file is size- and SHA-256-checked; the CountGD pickle is audited and converted once; no
  `trust_remote_code`; no credentials.
- Not a finding: OWLv2 staying frozen in `FULL` is stated and explained.

## 5. Promise-to-evidence matrix

| Promise | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| Compare Grounding DINO and OWLv2 under one contract | `e866f42c`, `747c9153`, `a147fce8` | AP/recall/duplicates tables (hosted) | §17 note | Delivered |
| Prompt wording as input | `69733ae2` | photo vs bare table (hosted) | Exercise A, no answer | Delivered; guidance gap (m6) |
| Negative prompt false positives | `69733ae2` | absent boxes/image | Exercise B | Delivered |
| CountGD three prompt modes | `ec46acb1`, `92c772c5` | MAE, point/box F1 | §18 note | Delivered |
| Compensating errors via localisation | `f5ee4325` | point/box F1 beside MAE | Exercise E, no answer | Delivered; guidance gap (m6) |
| Detector-as-counter | `1160dede`, `b9b5ef65` | threshold tables, comparison | Exercise C | Delivered |
| Density and distractor diagnostics for CountGD | `5fb1218c` | detectors only | — | **Not delivered (M4)** |
| Controlled change | guided-01 | none | — | **Not delivered (M5)** |
| Adaptation trade-off (`FULL`) | `229ad112`, `d0dff168`, `51ba74ee` | never executed | Exercise F | Not verified; reload check tautological (M3) |
| BYOD | controls only | none | — | **Not delivered (M2)** |
| Standalone | `083028a5` fetches source | — | — | **Nonconformant (M1)** |

| Objective | Learner activity | Evidence exercised |
|---|---|---|
| Goals 1–5, 10 | read tables, Exercises A–C | tables present; no worked guidance |
| Goals 6–9 | read tables, Exercises D–E | D lacks its evidence (M4) |
| Goal 11 | read freeze record | `frozen_experiment.json` |
| Goal 12 | Exercise F | `FULL` only; never run |

## 6. Readiness

**Needs revision.** Five open Majors (M1–M5). After they are fixed the status becomes **Verification pending**
until a hosted T4 `Run all` of the fixed head records: the `STANDARD` default path, BYOD with one valid and one
rejected archive (REL12), the activity, and `FULL` with both reload parities. Status stays **Candidate**;
`clean_runtime_evidence` stays `pending`.

## 7. Verified versus inferred

- **Verified (direct execution, CPU):** the scene generator reproduces the hosted per-split counts; the metric
  functions behave correctly on the probed cases; the dataset digest differs from the hosted value.
- **Verified (source inspection):** M1–M5 and m1–m3, m5–m7 by reading the cells (probes P01–P11 encode the checks).
- **Verified (documented evidence):** the 2026-09-26 Colab run covers this revision's code cells (P13).
- **Inferred:** the digest difference comes from the platform's zlib PNG encoding (scene contents match by counts;
  decoded pixels were not compared with the hosted run, whose images were not exported).
- **Not verified:** any model output on this host; `FULL`; Colab behaviour of an upload dialog.
- **Finding most likely to be wrong:** CNT-M5's severity. One could argue the reflection exercises A–F and the
  prompt-sensitivity table already give a guided comparison, making the missing runnable change a Minor.

Probe ZIP: `DIMER_Open_Vocabulary_Detection_and_Counting_Workshop_Review_Probes.zip` (`run_probes.py`,
`results.json`, `source_manifest.json`).
