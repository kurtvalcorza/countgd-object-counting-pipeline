# CountGD Open-World Object Counting E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 2 October 2026  
**Repository:** `kurtvalcorza/countgd-object-counting-pipeline`  
**Notebook:** `tutorials/countgd_object_counting_colab.ipynb`  
**Reviewed commit:** `413d558301f0c6e70a9c42870cee2497dfc97bfb` (`main`, confirmed with `gh api repos/kurtvalcorza/countgd-object-counting-pipeline/commits/main` before and after the probes)  
**Notebook Git blob:** `c619a762f8c53dd52f48bdc55e02e507fd9f92df`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-24 (commit `8d61b94`, `fetched_blob_verified: true`); the notebook has not changed since `7800b58`, and `git diff --stat 8d61b94 413d558 -- src tools tutorials/countgd_object_counting_colab.ipynb` touches only `tools/validate_release_assets.py`. `tools/build_notebook.py --check` and `tools/validate_release_assets.py` both exit 0 at the reviewed commit.  
**Finding prefix:** `CGD` (`CNT` is already used by the workshop notebook's review in PR #5)  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main` (`b1cfe13`).

## Executive assessment

This is a well-built notebook. It carries the eight package modules verbatim (cell metadata digests equal the module digests, and the generator check passes), pins the authors' Space at an immutable revision, audits the 1.25 GB pickle checkpoint statically before a restricted load, converts it once into a digest-pinned safetensors file, fetches 80 digest-pinned FSC-147 photographs, draws seeded synthetic scenes with known object boxes, validates and splits both, counts a demo scene in three prompt modes with sanity checks, reads the model three ways (count error, point localisation, box IoU) against a mean-count baseline and a template matcher, runs a bounded fine-tune with validation-MAE epoch selection, scores both held-out sets again, and reloads the exported adapter with a parity assertion. The prose explains the score semantics and the limits of the evidence carefully.

The direct CPU execution in this review reproduced the recorded values where it reached:

| Measure | This review (CPU, pinned venv, defaults) | Kaggle T4 record (blob `c619a762`) | CPU build record |
|---|---|---|---|
| Cells 3–25 at defaults | ok (install skipped); snapshot and tokenizer re-hashed; 24 / 8 / 12 scenes, 80 photographs split 40 / 16 / 24; four refusal probes rejected | ok on pass 2 | ok |
| Demo scene, text / exemplars / both | **35 / 51 / 35**, every sanity check `True` | 35 / 51 / 35 | 35 / 51 / 35 |
| Frozen model, 12 synthetic test scenes | MAE 7.417, RMSE 9.971, point F1 0.862, box F1 0.453 | 7.417 / 9.971 / 0.862 / 0.453 | identical |
| Baselines (synthetic), mean count / template matcher | MAE 10.000 / 14.500 | 10.000 / 14.500 | identical |
| Fine-tune, kept epoch, adapted test MAE | not run at defaults (time cap) | kept epoch **2**, MAE 1.583, FSC-147 3.542 | kept epoch **3**, MAE 0.917, FSC-147 3.500 |

Four problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (CGD-M1).** The recorded T4 run stopped at the install cell's stale-module guard (`cuda-bindings` 12.9.4 → 13.4.3, `numpy` 2.0.2 → 2.5.3) and passed only after a restart; the record calls this "as designed" and `PASSED`.
2. **The documented BYOD inputs do not reach the pipeline (CGD-M2).** The stated minimum ("at least eight images") fails Section 4: a one-category set needs 18 images, and 8–17 are rejected with `N records; 4..20000 are required`, which does not say which split or how many images to add. Points-only and count-only records, both offered in the opening, fail in Section 4 with a raw `KeyError: 'boxes'` from the refusal-probe code; a box-annotated set of 18 images passes.
3. **Re-running a stage continues from the adapted model (CGD-M3).** Nothing resets the pipeline. Re-running Section 7 with `TRAINABLE_LAYERS = 0`, as the Optional experiments suggest, trains on top of the adapted model, labels its epoch 0 "frozen model", and the export cell then fails its reload-parity assertion (box difference 296 px). BYOD is entered by re-running from Section 4 after the default run, so its "frozen" baseline and epoch 0 are the synthetic-adapted model.
4. **Guided layer largely absent (CGD-M4).** Declared `GUIDED`, but there is no audience statement, how-to-use, roadmap, glossary, prediction prompt, interpretation checkpoint, troubleshooting section or conclusion template, and the 6,294 lines of carried modules (3,923 in `modeling.py`) are not labelled as infrastructure.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`); `BYOD_ZIP_PATH` cites 2.1 EXE2 |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. Knowledge prerequisites: basic Python and PIL; bounding boxes and IoU; MAE and RMSE; threshold decisions vs probabilities; what fine-tuning the last layers changes |
| Supported runtime | "Google Colab, Kaggle or Jupyter, Python 3.12"; CUDA used when present, CPU also runs (float32); ~3 GB disk |
| Promised outcomes | Pinned install; carried package; pinned Space snapshot staged and digest-verified, pickle audited and converted; seeded synthetic scenes and 80 digest-pinned FSC-147 photographs validated and split without leakage; demo scene counted by text, exemplars and both with an input manifest and rejection probe; frozen model vs mean-count baseline and template matcher on count error, point localisation and box IoU; bounded fine-tune with validation-MAE epoch selection; held-out evaluation on scenes and photographs; demo re-count; adapter export and reload parity; seven output files; BYOD through "the same validation, seeded split, baselines, fine-tune, held-out evaluation, artifact export and reload-parity cells" |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; recorded generating revision `6ec56df` |

### Evidence actually obtained

- **Source inspection.** All 35 cells (16 code; cells 5–19 carry the eight modules). Also read: `pipeline.py` (`count`, `evaluate`, `adapt`, `save_artifact`, `load_artifact`), `samples.py` (`validate_dataset`, `split_dataset`, `load_byod_dataset`), `metrics.py` (`box_metrics`, `localisation_metrics`, `template_matching_baseline`), the generator, `README.md`, `STATUS.md`, `tutorials/README.md`, `docs/release-verification.md`. The repository has no `AGENTS.md`; `docs/execution-evidence/` holds only the workshop notebook's Colab run.
- **Documented execution evidence.** `docs/release-verification.md` plus the archived executor output for Kaggle kernel `dimer-nb2-countgd-object-counting` v1 (`run_summary.json`). Kaggle Tesla T4, 2026-09-24, **the reviewed blob**, clean HF cache, image torch 2.10.0+cu128 / numpy 2.0.2. Pass 1 failed in cell 3 with the restart `RuntimeError` (162.5 s); pass 2 ran 16/16 (306.1 s). The CPU build-record run of the same blob was a local pre-flight with pins pre-installed. No Colab run, no BYOD run and no optional-experiment run is recorded.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py`, Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=-1`), 24 threads, build venv `dimer-next16` (Python 3.12.10, torch 2.14.0+cu130, torchvision 0.29.0, transformers 4.57.6, safetensors 0.8.0, numpy 2.5.3, scipy 1.18.1, huggingface-hub 0.36.2, pillow 11.3.0 — the notebook's `PINS`). Nothing was installed.
  - **Install skipped:** cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`.
  - **Not a clean runtime:** the source checkpoint, the converted `countgd.safetensors`, the six tokenizer files and the FSC-147 cache were hard-linked into a scratch working directory; cell 21 wrote both manifests itself, fetched nothing, and `verify_snapshot` / `from_pretrained` re-hashed the files. The pickle audit and conversion were therefore **not** re-run.
  - **Executed at defaults:** cells 3–25 and cell 27 (P7, P10–P13). Cells 29, 31 and 33 were **not** run at defaults: the full CPU path is ~21 min on this host (build record 1,281.8 s), above the review's probe cap. The FSC-147 frozen evaluation and baselines were not run.
  - **Executed off the default path:** BYOD loader with 3 accepted and 11 refused inputs (P8); BYOD with an empty path outside Colab (P9); BYOD size sweep 8–18 images (P16); BYOD 18-image zips through cells 23 and 27 (P18); the re-run hazard through cells 29 and 33 on reduced sets (P14, P20, P21). The model-backed probes ran in two invocations under `timeout` (PIDs 14051 and 14344), both exited on their own; about 15 minutes in total.
- **Learner observation:** none. No claim here is about measured learning effectiveness.

## 2. Separate judgments

- **Technical correctness:** strong on the default path as far as it was executed (cells 3–27 reproduce the record exactly; supply-chain checks enforced). Defects: the install pattern forces a restart (CGD-M1); BYOD breaks on documented inputs (CGD-M2); stage re-runs leave the pipeline in a state the comparison labels and the export cannot represent (CGD-M3).
- **Promise fulfilment:** the default-path promises are met in the recorded T4 run, apart from one-pass `Run all`. The BYOD promise ("at least eight images"; points-only records "scored by point localisation") is not met. The Optional experiments are promised as not affecting the default path, but one of them breaks the export cell when followed.
- **Scientific validity:** sound. Splits are disjoint by decoded-pixel digest; validation selects the epoch and the test sets are used only for evaluation; baselines are scored by the same code on the same records; scores are described as uncalibrated similarities; the evidence is labelled one seeded draw with no dispersion estimate. The prose quotes the CPU build record's fine-tune result (kept epoch 3, MAE 0.92) as the expected result, while the supported GPU runtime kept epoch 2 (MAE 1.58) (CGD-m2).
- **Learner experience:** the prose is clear and explains why the three prompt modes count differently, which is the notebook's best teaching moment. It reads as a dense reference: no orientation, prediction, checkpoint or conclusion scaffolding (CGD-M4), and no guidance on which cells to re-run (CGD-M3).
- **Spec conformance:** unresolved applicable `MUST`s — RUN1, RUN10, ENV6 and REL2 (CGD-M1); DAT12, DAT14, DAT19 and REL12 (CGD-M2); UX7 and VER5 for the suggested experiment (CGD-M3). SHOULD gaps: GDL1–GDL4, GDL6, GDL7, GDL9, GDL11–GDL14 (CGD-M4); UX10 and EXE2 (CGD-m1).

## 3. Promise and objective tracing

| Claim | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| Run all completes with no intervention | cell 3 pip install + stale-module guard | T4 pass 1 `RuntimeError`, pass 2 ok | Learner must restart and run again | **Not met** (CGD-M1) |
| Snapshot pinned, verified, pickle audited and converted | cell 21, `model.py` | T4: converted digest matched; review: re-hash ok, nothing fetched | Prose explains the trust boundary | Met (audit not re-run in review) |
| Synthetic scenes + FSC-147 validated and split without leakage, four refusals | cell 23 | P11: 24/8/12, 40/16/24, all four probes rejected with the rule named | Clear | Met |
| Demo counted three ways; exemplar mode counts distractors | cell 25 | P12: 35 / 51 / 35, checks `True` | Explained before and after | Met |
| Frozen model vs mean count and template matcher on count, points, boxes | cell 27 | P13: MAE 7.417 vs 10.0 / 14.5; box F1 0.453 | Explained; definitions printed | Met (synthetic); FSC-147 by record only |
| Bounded fine-tune with validation-MAE selection | cell 29 | T4: kept epoch 2; CPU record: epoch 3 | Prose expects epoch 3 | Met; prose fixed to CPU values (CGD-m2) |
| Held-out evaluation after fine-tune; FSC-147 not broken | cell 31 | T4: 7.42 → 1.58, FSC 4.54 → 3.54 | Prose quotes 0.92 / 3.50 | Met; quoted values are CPU-only (CGD-m2) |
| Adapter exported and reloaded with parity | cell 33 | T4: 4/4 identical, 0.0 difference | Clear | Met at defaults; fails after a suggested experiment (CGD-M3) |
| BYOD: one zip, ≥ 8 images, boxes or points, same cells through reload | cells 23–33 | P16: < 18 one-label images rejected; P18: points-only / count-only `KeyError: 'boxes'`; 18 box-annotated images pass cells 23 and 27 | Rejections do not say what to change | **Not met** (CGD-M2) |
| Optional experiments do not affect the default path | Interpretation cell | P21: re-running cell 29 with `TRAINABLE_LAYERS = 0` → cell 33 `AssertionError` | No re-run guidance | **Not met** (CGD-M3) |

| Objective (opening) | Learner activity | Evidence it was exercised |
|---|---|---|
| install the pinned runtime; read what the package guarantees | run cell 3; read cells 4–19 | Run only; the package is 6,294 lines with no reading guide |
| stage and verify the snapshot; see a pickle audited and converted | run cell 21 | Printed identity, fetched list, verified files; the audit itself prints nothing the learner reads |
| draw scenes, fetch photographs, validate and split without leakage | run cell 23 | Split sizes, digests and four refusal messages printed |
| count one scene three ways and read what each mode counts | run cell 25; read Section 5 prose | Counts printed and PNGs saved (not displayed); no prediction asked |
| read counts, points and boxes correctly | Section 5–6 prose | Explanations only; no checkpoint |
| measure the frozen model beside two baselines | run cell 27 | Table printed; no interpretation prompt |
| run a bounded fine-tune with explicit hyperparameters | run cell 29 (form fields) | Epoch rows printed; changing a field without guidance breaks state (CGD-M3) |
| evaluate on held-out scenes and photographs | run cell 31 | Comparison printed and written |
| export an adapter that reloads with verified parity | run cell 33 | Parity printed and asserted |

Most objectives are exercised only by running a cell and reading output; none asks the learner to predict, explain or decide (CGD-M4).

## 4. Findings

### CGD-M1 Major — `Run all` needs a manual restart after the in-kernel pip install

**Cell/section:** Section 1, cell 3 (install + stale-module guard); `docs/release-verification.md` step 4 and the evidence row; `STATUS.md`; `tutorials/README.md`.

**Observed issue:** Cell 3 pip-installs the pins into the running kernel and raises `RuntimeError('... Restart the runtime, then rerun from the top.')` when a loaded distribution changed. In the recorded Kaggle T4 run of this blob, pass 1 stopped there (`cuda-bindings: loaded=12.9.4, installed=13.4.3; numpy: loaded=2.0.2, installed=2.5.3`, 162.5 s) and pass 2 ran 16/16 after a restart (306.1 s). The release record calls the restart "as designed" and the run `PASSED`; step 4 of the procedure says a restart "is expected where the runtime's preinstalled torch or numpy differ from the pins", which is the normal case on Colab and Kaggle.

**Consequence:** A learner who chooses Run all sees an error at the first code cell and must restart and run again. The notebook is not `Run all` conformant, and the record overstates what was verified.

**Evidence:** Documented execution evidence (`run_summary.json`: `passes[0].ok = false`, `restarted_after_install_cell: true`, `committed_blob = c619a762…`, `fetched_blob_verified = true`); source inspection (cell 3; `docs/release-verification.md` lines on step 4 and the evidence table).

**Recommended correction:** Adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass: the setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart can be required. Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`. Do not add another in-kernel install guard or loosen pins to dodge the restart. Implement it in the repository's notebook generator (`tools/build_notebook.py` / `tools/notebook_template.py`), regenerate, re-qualify with a one-pass hosted Run all, and correct the release record so a restart-dependent run is not reported as a `Run all` PASS.

**Acceptance check:** A fresh Colab and a fresh Kaggle runtime each complete the regenerated notebook with one `Run all`, no restart and no error, recorded with the blob id; `docs/release-verification.md`, `STATUS.md` and `tutorials/README.md` no longer describe a restart as expected or a restart-dependent run as `PASSED`.

**Spec:** RUN1, RUN10, ENV6, REL2.

### CGD-M2 Major — the documented BYOD inputs fail in Section 4

**Cell/section:** Opening "Bring Your Own Data" paragraph, Prerequisites "Data contract", Section 4 cell 23; `samples.py` `split_dataset` / `validate_dataset`; generator template for cell 23.

**Observed issue:**
1. **Minimum size.** The opening says "at least eight images". Cell 23 validates each split with the default `min_records=4`, but `split_dataset` takes 30 % / 20 % per label with at least one test record per label. For one category, 8–17 images are all rejected with `ValueError: N records; 4..20000 are required` (P16: 8 → 2, 9–11 → 3, 12 → 2, 13–17 → 3); 18 is the first size that passes. Eight images over six categories fail earlier with `split leaves 2 training records`. Neither message names the split or the number of images needed.
2. **Points-only and count-only records.** The opening says "records with points only train on upstream's point boxes and are scored by point localisation", and the Prerequisites make `points` and `boxes` optional. With 18 one-category images, a points-only zip and a count-only zip both pass `load_byod_dataset`, `split_dataset` and `validate_dataset`, then cell 23 fails at line 50 with `KeyError: 'boxes'`: the "box outside its image" refusal probe builds `example['boxes'][1:]` from the user's first training record (P18). By source inspection the next failure in line is cell 27's `assert frozen_syn['boxes']['n'] == len(test_records)` (`box_metrics([])` returns `n = 0`), and count-only records would fail in `adapt` ("adaptation needs gold points") only after Section 6.
3. A box-annotated 18-image zip passes cells 23 and 27 (P18: split 9 / 4 / 5, frozen MAE 0.0 on 5 test scenes), so the branch works for one data shape at about twice the stated minimum.

**Consequence:** A learner following the stated contract with a small set, or with point annotations (FSC-147's own format), cannot reach the fine-tune. The errors give no corrective action. DAT14 requires BYOD to reach validate → split → fine-tune → evaluate → export.

**Evidence:** Direct execution (P8, P16, P18; CPU, pinned venv, notebook cell 23 and 27 code verbatim with `USE_BYOD = True` and `BYOD_ZIP_PATH` set); source inspection (cell 23 lines 36 and 50, cell 27 line 29, `metrics.box_metrics`, `pipeline.adapt`).

**Recommended correction:** In the generator template, (a) derive and state the real minimum (or validate the BYOD splits with a size rule that matches `split_dataset`, e.g. `min_records=1` for validation/test and a stated per-category minimum), and make `split_dataset` / cell 23 fail with a message that names the split, its size, and how many images per category are needed; (b) build the refusal probes from the default synthetic example (or skip the box probe when the record has no `boxes`), and make cell 27's assertion and printing conditional on the annotations present; (c) reject count-only BYOD records in Section 4 with a message that adaptation needs points or boxes, or document an evaluation-only route. Add a regression test that loads 8-, 18-image and points-only zips through the cell 23/27 code.

**Acceptance check:** With `USE_BYOD = True` and `BYOD_ZIP_PATH` set, (1) a zip of the stated minimum size with boxes, and (2) the same with points only, each run cells 23–33 to completion and write the adapter and result JSON; (3) a smaller zip and a count-only zip are rejected in Section 4 with a message naming the failed rule and the fix.

**Spec:** DAT12, DAT14, DAT19, REL12, UX10.

### CGD-M3 Major — re-running a stage continues from the adapted model; a suggested experiment breaks the export

**Cell/section:** Section 7 cell 29, Section 9 cell 33, "Optional experiments" paragraph, BYOD instruction "re-run from that cell" (opening); `pipeline.adapt`.

**Observed issue:** `pipe` is created once (cell 21). `adapt` trains whatever weights are loaded, records them as epoch 0 with the note "frozen model", and has no reset to the base. No cell says which cells to re-run.
1. **Suggested experiment.** After an adaptation that kept a trained epoch (both recorded runs did: epoch 3 on CPU, epoch 2 on T4), re-running cell 29 with `TRAINABLE_LAYERS = 0` (as the Optional experiments suggest) trains 512 parameters on top of the adapted model, prints epoch 0 as "frozen model" although decoder layer 5 still carries the first run's change (max |Δ| 4.0e-4, unchanged after the re-run), and exports an adapter with 2 tensors. Cell 33's reload builds the base plus those 2 tensors, so the parity check fails with a bare `AssertionError` (max box difference 296.2 px, max score difference 0.035) (P20, P21).
2. **BYOD.** The learner is told to enable BYOD "after the tutorial workflow completes" and re-run from Section 4. Section 6's "frozen" row, Section 7's epoch 0 "frozen model" and the comparison's `frozen` column are then the synthetic-adapted model, and the BYOD adapter continues from it. The comparison the notebook teaches ("baselines first", frozen vs adapted) is mislabelled on the learner's own data.

**Consequence:** The learner's first experiment either crashes the export or produces a comparison against the wrong reference, with no indication why. The notebook's central lesson (read the adapted model against the frozen one) is silently invalidated.

**Evidence:** Direct execution, P20/P21 (CPU; reduced sets of 2 train / 2 validation / 2 test scenes and 1 epoch; the first adaptation used `pipe.adapt(..., val=None)` as a stand-in for a completed Section 7 that kept a trained epoch; cells 29 and 33 then ran verbatim apart from `EPOCHS = 1` and `TRAINABLE_LAYERS = 0`). P14, the same sequence with a validation set, kept epoch 0 and so started from the frozen weights, which is why P20 forces a trained epoch. BYOD consequence by source inspection (`adapt`, cells 21–31).

**Recommended correction:** In the generator template, reload or restore the frozen base at the start of Section 6 and Section 7 (e.g. keep a copy of the trainable tensors after cell 21 and overlay it before scoring the frozen model and before `adapt`, or re-run `from_pretrained`), and make `adapt` refuse or warn when `self.adapter` is already set. State in each experiment and in the BYOD instruction exactly which cells to re-run, and turn cell 33's parity assertion into a message that names the likely cause.

**Acceptance check:** In one kernel, run all cells at defaults, then re-run cell 29 with `TRAINABLE_LAYERS = 0` and cells 31–33: epoch 0 equals the frozen validation MAE from the first run, and the reload parity passes. Then enable BYOD with a valid zip and re-run from cell 23: Section 6's frozen numbers equal those of a fresh kernel on the same zip.

**Spec:** UX7, GDL10, VER5, DAT13, UX10.

### CGD-M4 Major — declared `GUIDED`, but the guided layer is largely absent

**Cell/section:** Opening and Prerequisites, Sections 2–9, Interpretation.

**Observed issue:** No intended-learner statement (prerequisites list knowledge only), no "How to use this notebook" (Run all, form fields, which cells are infrastructure), no roadmap, no glossary although the notebook introduces exemplar tokens, queries, Hungarian matching, NAE, point localisation radius, box IoU matching and focal loss, no prediction before any comparison (P3: no "predict" prompt), no interpretation checkpoints with sample answers, no troubleshooting section (1.25 GB download, ~3 GB disk, CPU duration, BYOD errors), and no conclusion template. The eight carried module cells hold 6,294 lines (`modeling.py` alone 3,923) and are introduced as "the repository's package" but not labelled as infrastructure the learner may run without reading, and no cell is collapsed (`cellView: form` absent). The learning objectives use "install", "read", "see", "draw", "run" rather than observable actions.

**Consequence:** A self-paced learner gets a dense, well-written reference but no structure telling them what to attend to, what to predict, how to check their understanding, or what to do when a cell fails. The 6,294-line carrier dominates the notebook's length.

**Evidence:** Source inspection of all 35 cells; P1 (embedded line counts, no `cellView`/collapsed metadata), P3 (keyword sweep).

**Recommended correction:** Add the guided layer in `tools/notebook_template.py`: audience and how-to-use in the opening; a roadmap; an Input → Model → Output contract ("image + prompt (text / ≤ 3 boxes) → CountGD → count, boxes, points, scores"); a short glossary; a prediction before Section 5 ("what will three exemplar boxes count on this scene?") and before Section 8; one checkpoint with a collapsible sample answer after Sections 5, 6 and 8; label the module cells **Infrastructure** and give them `cellView: form`; a troubleshooting section; a conclusion template that asks for task, principal result, baseline, failure mode and limits.

**Acceptance check:** The regenerated notebook contains each GDL1–GDL4, GDL6, GDL7, GDL9, GDL11–GDL14 element, identifiable by heading or callout; every module cell is labelled infrastructure and collapsed in Colab; the generator check still passes.

**Spec:** GDL1–GDL4, GDL6, GDL7, GDL9, GDL11–GDL14, GDL5 (objectives wording).

### CGD-m1 Minor — BYOD error messages that name no rule or fix

**Cell/section:** Section 4 cell 23; `samples.load_byod_dataset`.

**Observed issue:** `file` values with a folder (`images/x.png`) fail with `KeyError: 'images/x.png'` (members are indexed by basename, lookups by the CSV value); a file listed but missing from the zip gives `KeyError: 'img3.png'`; an undecodable image gives `UnidentifiedImageError: cannot identify image file <_io.BytesIO …>` without the file name; `USE_BYOD = True` with an empty `BYOD_ZIP_PATH` outside Colab gives `ModuleNotFoundError: No module named 'google.colab'`. A `..` member path is also read by basename (not extracted, so no traversal), and fails as a missing file. Missing `labels.csv`, missing columns, a non-integer count and a count that disagrees with its boxes are refused with the rule named.

**Consequence:** The learner cannot tell what to change. The folder-path case is likely, because zipping a folder of images produces it.

**Evidence:** Direct execution (P8, P9).

**Recommended correction:** In `load_byod_dataset`, accept relative member paths (normalised, with `..` and absolute paths refused), and wrap file lookup and decoding in errors that name the row id, the file and the rule; in cell 23, refuse an empty path outside Colab with a message that says to set `BYOD_ZIP_PATH`.

**Acceptance check:** Each of the five inputs above produces a `ValueError` naming the row or file and the fix; a zip whose `labels.csv` references `images/<name>` loads.

**Spec:** DAT19, UX10, EXE2, §20 (path handling).

### CGD-m2 Minor — expected results are the CPU build record, not the supported GPU runtime's

**Cell/section:** Section 7 and 8 prose, Section 9 prose, Interpretation.

**Observed issue:** The prose says the selector "kept epoch 3", expects the test MAE to move "7.42 → 0.92", box F1 "0.453 → 0.550" and FSC-147 "4.54 → 3.50", and the Interpretation states 0.92 / 3.50 as "the claim". These are the CPU pre-flight values; the recorded T4 run of the same blob kept epoch 2 with 1.58 / 0.546 / 3.54. The notebook does not say CUDA results differ (the README does), and the Prerequisites recommend a GPU runtime.

**Consequence:** A learner on Colab or Kaggle GPU sees different numbers than the text promises and cannot tell whether that is normal.

**Evidence:** Source inspection (P4); documented execution evidence (release record, both rows).

**Recommended correction:** In the template, describe the expected shape (validation MAE falls, then may turn; the kept epoch is the lowest) and quote both recorded runs with their runtime, stating that GPU kernels are not bit-deterministic so the kept epoch and the adapted numbers vary.

**Acceptance check:** No adapted-model number appears in the notebook without its runtime; the Interpretation's claim holds for both recorded runs.

**Spec:** GDL8, ENV8, UX12.

### CGD-m3 Minor — BYOD has no new-data inference step and "never saw" is stated without the pretraining caveat

**Cell/section:** Section 9; Section 6 and opening prose.

**Observed issue:** In BYOD mode, Section 9 re-counts the synthetic demo scene, and the only per-image outputs on the learner's data are the counts inside the evaluation report (no boxes or points exported per image). Separately, the FSC-147 test photographs are described as "categories CountGD never saw in training"; this holds for FSC-147's training split, but the notebook does not state that the GroundingDINO / BERT pretraining data may contain those categories.

**Consequence:** BYOD ends without a prediction on the learner's own images that they can inspect; the "never saw" wording slightly overstates the out-of-distribution check.

**Evidence:** Source inspection (cells 31, 33; P5).

**Recommended correction:** In BYOD mode, count the test split's images and write their boxes, points and scores per image (CSV or JSON); qualify "never saw" as "categories absent from FSC-147's training split" and add a one-line pretraining-overlap note.

**Acceptance check:** A BYOD run writes a per-image predictions file for the user's test images; the opening and Section 6 name the training split and the overlap limitation.

**Spec:** DAT14, INF2, OUT1, OUT4, DAT9.

### CGD-S1 Suggestion — declare NOTEBOOK_SPEC 2.2

The notebook, metadata and registry declare 2.0 (with a 2.1 field). Regenerate against 2.2 once the findings above are addressed.

### CGD-S2 Suggestion — display the demo PNGs

Section 5 and Section 9 save the frozen and adapted demo images but do not show them. Displaying the exemplar-only pair side by side would make the distractor lesson visible in the notebook.

### CGD-S3 Suggestion — show variability of the fine-tune

Run the default fine-tune at two or three seeds (or quote the CPU and T4 records together) to show that the kept epoch and adapted MAE vary.

### CGD-S4 Suggestion — per-stage wall times in the result JSON

Record the staging, conversion, evaluation and adaptation times in `countgd_object_counting_result.json` so a release record can quote measured runtimes per environment.

## 5. Journeys

| Journey | Basis | Result |
|---|---|---|
| First-time learner | Source inspection | Clear, careful prose; strong explanation of prompt modes and score semantics; no orientation, prediction, checkpoint, troubleshooting or conclusion scaffolding (CGD-M4); expected numbers fixed to CPU (CGD-m2) |
| Clean default | Documented execution (Kaggle T4, reviewed blob) + direct execution (CPU, cells 3–27) | T4: pass 1 stopped at the install guard, pass 2 16/16 (CGD-M1). CPU: cells 3–27 reproduce the record exactly (35/51/35; MAE 7.417, box F1 0.453; baselines 10.0 / 14.5). Sections 7–9 and FSC-147 not run directly. No Colab run |
| Active learning | Direct execution (reduced sets) | Re-running cell 29 with `TRAINABLE_LAYERS = 0` after a trained adaptation → cell 33 `AssertionError` (CGD-M3); threshold and FSC-147 fine-tune experiments not verified |
| Reuse and recovery | Direct execution | BYOD: 8–17 one-label images rejected; points-only / count-only → `KeyError: 'boxes'` (CGD-M2); 18 box-annotated images pass cells 23 and 27; adaptation, export and reload on BYOD not verified; refusals as in CGD-m1; Colab upload dialog not exercised |

## 6. Readiness

**Needs revision.** Four Major findings are open, and RUN1/RUN10/ENV6/REL2, DAT12/DAT14/DAT19/REL12 and UX7/VER5 are unresolved applicable `MUST`s. Remaining gates after the fixes: a one-pass hosted `Run all` of the regenerated blob (Colab and Kaggle), and a recorded BYOD run through export and reload (REL12).

## 7. What was verified and what was inferred

- **Verified by direct execution:** cells 3–27 at defaults (CPU); the demo counts and frozen synthetic metrics equal the record; the BYOD size threshold (18 one-label images); the points-only/count-only `KeyError`; the re-run → parity failure (reduced sets, stand-in first adaptation); the BYOD refusal messages.
- **Verified from documented evidence:** the T4 restart and the 16/16 second pass on the reviewed blob.
- **Inferred from source:** the BYOD "frozen" mislabel (CGD-M3 part 2); the cell 27 assertion failure for points-only records once cell 23 is fixed; the late count-only failure in `adapt`.
- **Not verified:** any Colab run; Sections 7–9 at defaults on this host; the pickle audit and conversion in this review; BYOD through fine-tune, export and reload.
- **Most likely to be wrong:** CGD-M3's severity. Its direct demonstration used a stand-in first adaptation on two scenes rather than the full default Section 7; a maintainer could argue the experiment is optional and rate it Minor. It is rated Major because the notebook itself suggests the experiment and the BYOD route uses the same re-run pattern, which silently changes what "frozen" means.

Probe files: `countgd_object_counting_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).
