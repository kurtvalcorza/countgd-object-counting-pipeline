# Release verification

`tutorials/countgd_object_counting_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`config.py`, `metrics.py`, `modeling.py`, `synthetic.py`, `model.py`,
  `provenance.py`, `samples.py`, `pipeline.py`, in dependency order), each equal to its source after the
  generator's documented rewrites (the two `__file__` uses in `model.py` made working-directory-relative, and the
  removal of package-relative imports); the inline `MANIFEST` and `TOKENIZER_MANIFEST` equal to the committed
  snapshot manifests and the inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical
  (on LF) to `tools/build_notebook.py` output for its recorded revision; the uv isolated-environment
  install (generator `/2.1`: a pinned `uv` wheel checked by size and SHA-256, managed CPython 3.12.12, the carried
  hash lock `tutorials/requirements-colab.lock.txt` installed with `--require-hashes --only-binary :all:`, a Linux
  x86_64 check) and exactly two kernel cells, every carried module and setup cell titled `Infrastructure:` and
  collapsed; the guarded pip path survives only in the runtime-record cell, for executors that pre-install the pins;
  `NOTEBOOK_SOURCE` recorded in exports; the guided-layer markers and the stale-text list of the 2026-10-04 review
  fixes;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cells (and repeated in the inline manifest, which
  the notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md` and `MODEL_CARD.md` with no stray revisions;
- the profile-specific public-API calls (the synthetic splits, the FSC-147 fetch and split, the refusal probes,
  the demo scene counted in three prompt modes with its input manifest and sanity checks, `pipe.evaluate` on the
  frozen model with both baselines, `pipe.adapt` with its explicit hyperparameters and the selector assertion,
  `pipe.evaluate` after adaptation with the re-score assertion, the demo re-count, `pipe.save_artifact`,
  `CountGDPipeline.from_artifact` and the reload-parity assertion, `write_provenance`, and the result fields
  including the source and pickle-audit digests), the seven expected `outputs/` paths, the learner-facing
  statements, and the gated-off BYOD default; forbidden patterns outside the carried module cells (direct
  `huggingface_hub` / `transformers` / `safetensors` / `urllib.request` use, `build_groundingdino(`, `torch.load(`,
  `pickle`, `torch.optim`, `.backward(`, `requires_grad`, `pipe.model.`, `extractall(`, `trust_remote_code=True`,
  a mutable `revision='main'`, any `git clone` / repository import);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.2"`), single H1, the 19 required headings in order, and the
  checkpoint-invariants section.

CI also installs the frozen CPU reference environment (`requirements.lock.txt`), runs `ruff`,
`scripts/check_lock.py`, `tools/build_notebook.py --check`, and the offline unit suite (`tests/`; no weights,
injected downloaders and photo fetcher — `tests/test_model_backed.py` is skipped without the converted checkpoint).
These are source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU or GPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Repository CI integration job (`tools/run_notebook.py`, manual `workflow_dispatch` or push to `main`) | GitHub-hosted Ubuntu runner, the frozen CPU reference environment with `DIMER_NOTEBOOK_CI_PREINSTALLED=1` | Executes the standalone notebook's code cells against the real pinned weights; a **pre-flight** on the locked stack, not a fresh-boundary run of the inline `PINS` and not promotion evidence on its own |
| Local harness (pre-flight only) | Build workstation, fresh Jupyter kernel, CPU only, pre-installed pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact commit under review and confirm static CI is green (or, where hosted CI cannot run, that the
   same checks pass locally and say so);
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above)
   with **no repository checkout**, an empty Hugging Face cache, and nothing pre-staged under `weights/countgd/`,
   `weights/bert-base-uncased/` or `weights/fsc147-subset/` (the standalone path writes the manifests itself,
   stages the missing files from the Hub and fetches the pinned photographs, so none of them may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `BYOD_ZIP_PATH = ''`, `SPLIT_SEED = 42`, `EPOCHS = 4`, `LEARNING_RATE = 2e-4`,
   `TRAINABLE_LAYERS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `transformers==4.57.6`, `safetensors==0.8.0`,
   `numpy==2.5.3`, `pillow==11.3.0`, `scipy==1.18.1`, `huggingface-hub==0.36.2`, installed into the isolated
   environment (Python 3.12.12) and not into the kernel. The run must complete in **one pass with no restart**
   (`restarted: false`); a run that needs a manual restart is recorded but is not a one-pass `Run all` and not
   promotion evidence;
5. verify every default-path stage completes:
   - Section 3: the inline manifests asserted against the module's constants; `stage_missing_files` fetching the
     Space `README.md` and the 1,250,122,522-byte `checkpoint_best_regular.pth` from `nikigoli/countgd` at
     `6e82e595…`; `verify_snapshot` re-hashing both (source SHA-256 `c1bab864…`); the six tokenizer files staged
     and verified; `CountGDPipeline.from_pretrained(...)` auditing the pickle, converting it into
     `countgd.safetensors` (937,560,480 bytes, SHA-256 `8e44867b…` — the load raises on any other digest) and
     loading it strictly;
   - Section 4: 24 / 8 / 12 synthetic scenes with `check_split_disjoint` reporting no shared image; 80 FSC-147
     photographs fetched with every byte count and SHA-256 matching, split 40 / 16 / 24 per category;
     `outputs/…_train.csv` written; the four refusal probes each rejected;
   - Section 5: the ceilings printed; the input manifest written (verdict `accepted`, one entry per prompt mode,
     one recorded rejection finding from the remote-URL probe); the demo scene (35 blue circles, 16 green
     triangles) counted by text, exemplars and both (35 / 51 / 35 on the build workstation's CPU) with every
     sanity check `True`; three PNGs written; the per-call report verdict `sample-sanity`;
   - Section 6: the frozen model on the 12 synthetic test scenes and the 24 FSC-147 photographs beside the
     mean-count baseline and the template matcher (build-record values in the table below);
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 3,752,196 trainable of 233,362,816 parameters (64 tensors, the shared box head included),
     four epochs with the validation MAE selecting the epoch, and the cell's assertion that the kept epoch is not
     worse than epoch 0;
   - Section 8: both held-out sets scored again, the comparison printed and `outputs/…_evaluation_report.json`
     written, with the cell's assertion that the kept epoch re-scores to the number that selected it;
   - Section 9: the demo scene re-counted in three modes, three adapted PNGs and `outputs/…_demo.json` written;
     `pipe.save_artifact` writing `outputs/…_adapter/{adapter.safetensors,manifest.json}`;
     `CountGDPipeline.from_artifact` reloading it with 4/4 identical counts on four test scenes (the cell asserts
     it); `outputs/provenance.json` and `outputs/…_result.json` written;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the caches were clean, the outcome, the produced outputs, the
   observed metrics (as observations, not a benchmark) and any warning in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `countgd_object_counting_colab.ipynb` (`E2E`) | `8d61b94` / `c619a762` | 2026-09-24 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-countgd-object-counting` v1; image `torch 2.10.0+cu128` before the pinned install, `torch 2.14.0+cu130` after; the notebook fetched by commit from GitHub and its Git blob verified before execution; no repository checkout; empty model, weights and photograph caches) | **Passed only after a manual restart** — pass 1 stopped at the install cell's stale-module guard (`cuda-bindings` 12.9.4 → 13.4.3, `numpy` 2.0.2 → 2.5.3; 162.5 s), pass 2 ran 16/16 code cells (306.1 s), 468.7 s in total (`restarted_after_install_cell: true`): not a one-pass `Run all`, not promotion evidence |

## Recorded executions

Each row is one execution: what ran, where, and what was observed. Static checks are not listed here.

| Date (UTC) | Subject | Runtime | Procedure | Observed result | Caveats |
|---|---|---|---|---|---|
| 2026-10-04 | box-head fix (CGD-x1): `_trainable_names` now also matches `transformer.decoder.bbox_embed.0.`, the name the real network reports for the shared box head | Build workstation CPU (`CUDA_VISIBLE_DEVICES=-1`), Python 3.12.10, `torch 2.14.0`; local harness as in the row below | the regenerated notebook's Section 4 (default synthetic data) and Section 7 verbatim except `EPOCHS = 1` (**reduced** from 4, CPU time), then `pipe.evaluate` on the 12 test scenes | trained set 64 tensors / 3,752,196 parameters, the six `transformer.decoder.bbox_embed.0.*` tensors among them (was 58 / 3,619,584); `TRAINABLE_LAYERS = 0`: 8 tensors / 133,124; validation MAE 9.375 → 7.625 after one epoch (the old trained set's first epoch in the recipe sweep: 6.88), kept epoch 1; synthetic test MAE 7.417 → 5.583, RMSE 9.971 → 8.874, point F1 0.862 → 0.892, box F1 0.453 → 0.478; 406.9 s | one epoch only, so not comparable to the 4-epoch records below, which were made with the old trained set; adapters saved before this fix (58 tensors) are refused by `load_artifact` |
| 2026-10-04 | `tutorials/countgd_object_counting_colab.ipynb` regenerated for the row-16 review fixes (uv isolated environment; this revision's blob) | Build workstation CPU (`CUDA_VISIBLE_DEVICES=-1`, Windows), Python 3.12.10, `torch 2.14.0`, `transformers 4.57.6`, the pins pre-installed; local harness, **not** the isolated environment (its two kernel cells were skipped and `DIMER_NOTEBOOK_CI_PREINSTALLED=1` set); the pinned checkpoint, its conversion and the FSC-147 cache hard-linked in | the learner cells executed verbatim from the notebook JSON in one namespace, in five processes: (1) default path Sections 4–6 plus the Section 4 refusal checks; (2) an 18-image one-category BYOD zip with `boxes` in a sub-folder through Sections 4–9 (`EPOCHS = 1`); (3) 12 points-only BYOD images (four categories, split 4 / 4 / 4) through Sections 4–6, a one-epoch `pipe.adapt` without validation as a stand-in for a completed Section 7, then Section 7 with `TRAINABLE_LAYERS = 0`, `EPOCHS = 1` and Sections 8–9; (4) the same points zip, Section 6, the stand-in adaptation, then Section 4 and Section 6 again | (1) demo 35 / 51 / 35; synthetic test frozen MAE 7.417, RMSE 9.971, point F1 0.862, box F1 0.453; baselines 10.000 / 14.500; FSC-147 frozen MAE 4.500, RMSE 8.670; refusals: 17 images → "validation gets 3 … one category needs at least 18 images", count-only, missing file, undecodable image, `..` member and an empty path off Colab each refused naming the row or file and the fix; 18 images (flat) accepted, split 9 / 4 / 5; (2) all cells ok, adapter and result JSON written, `…_byod_predictions.json` with 5 images, reload parity 4 / 4 exact (kept epoch 0: the frozen model already counted these scenes exactly); (3) Section 7 printed `reloaded_pinned_base`, epoch 0 validation MAE 11.75 equal to the frozen model's 11.75, 512 trainable parameters in 2 tensors, reload parity 4 / 4 exact; points-only scoring: box `n` 0, point `n` 4; (4) Section 6 after the stand-in adaptation equal to the fresh run (MAE 11.5, RMSE 12.53) | local pre-flight, not clean-runtime evidence; reduced epochs and stand-in adaptations, labelled; the FSC-147 frozen MAE differs from the 2026-09-24 CPU record (4.542) by one object on one photograph (different thread count); the uv download, environment build and hash-locked install were not executed (Linux only) |
| 2026-09-24 | `tutorials/countgd_object_counting_colab.ipynb` at `8d61b94`, blob `c619a762` | Kaggle Tesla T4 (15,360 MiB), Python 3.12.13, `torch 2.14.0+cu130`, `transformers 4.57.6`, `numpy 2.5.3`, pins installed by the notebook's own install cell | clean container, `Run all` in a fresh interpreter, all form fields at their defaults; the notebook staged the Space README and the 1,250,122,522-byte checkpoint, the tokenizer and the 80 photographs itself, and converted the checkpoint on the runtime | `countgd.safetensors` passed its pinned digest; demo scene 35 / 51 / 35 → 35 / 35 / 35 (text / exemplars / both); synthetic test MAE 7.417 → 1.583, RMSE 9.971 → 4.425, point F1 0.862 → 0.967, box F1 0.453 → 0.546; validation MAE 9.375 → 0.875, kept epoch 2; FSC-147 MAE 4.542 → 3.542, RMSE 8.720 → 8.468; baselines as on the CPU (mean count 10.000 / 16.208, template matcher 14.500 / 34.000); reload parity 4 / 4 identical counts, 0.0 box and score difference; adapter 14,485,144 bytes, 58 tensors; 101 files, 2,191 MB staged | the fine-tune's trajectory differs from the CPU runs (kept epoch 2, not 3) because CUDA kernels are not bit-deterministic; the frozen and baseline numbers match the CPU to the printed precision except the FSC-147 frozen RMSE (8.720 against 8.691); one observation, not an evaluation |
| 2026-09-24 | the same blob `c619a762` | Build workstation CPU (`CUDA_VISIBLE_DEVICES=-1`), Python 3.12.10, `torch 2.14.0`, `transformers 4.57.6`; local harness, pins pre-installed | fresh Jupyter kernel, all cells in order, the source checkpoint pre-staged by a hard link, everything else staged by the notebook | 1,281.8 s, 0 errors; the values quoted in the notebook, README and card: synthetic test MAE 7.417 → 0.917, RMSE 9.971 → 3.175, point F1 0.862 → 0.981, box F1 0.453 → 0.550; validation MAE 9.375 → 0.750, kept epoch 3 (epoch 4: 2.750); FSC-147 MAE 4.542 → 3.500, RMSE 8.691 → 8.337; demo 35 / 51 / 35 → 35 / 35 / 35; reload parity 4 / 4 exact; adapter SHA-256 `58414f2e…` | a pre-flight, not promotion evidence; bit-identical to the earlier local run below in every reported value |
| 2026-09-24 | the notebook generated before the template-matcher correction (blob `3efffd1e`, not committed) | Build workstation CPU, as above | as above | 1,210.9 s, 0 errors; model numbers as in the row above; the template matcher reported MAE 26,904.667 on the synthetic scenes and 623.750 on FSC-147 | this run exposed the flat-window defect in the template matcher's correlation, corrected before the committed blob; its baseline values are superseded |
| 2026-09-24 | package API, not the notebook (the source that generated the first notebook) | Build workstation CPU (`CUDA_VISIBLE_DEVICES=-1`), Python 3.12, torch 2.14.0, transformers 4.57.6 | the pinned checkpoint downloaded from the Space and from Google Drive and hashed; the static audit; the restricted load; the conversion run in two separate processes; the tensor-by-tensor comparison with the authors' Hub export | both downloads 1,250,122,522 bytes with SHA-256 `c1bab864…`; five globals, audit digest `4606eaf3…`; `countgd.safetensors` 937,560,480 bytes with SHA-256 `8e44867b…` in both processes; 1,042 / 1,042 tensors and 66 / 66 aliases equal to the export, tensor data section byte-identical | the Google Drive copy was deleted after hashing; the export is a cross-check, not a pin |
| 2026-09-24 | package API: the three prompt modes on scenes 0–15 | Build workstation CPU, as above | `count` with text, three exemplars, and both, at the default threshold | exemplars alone counted targets plus distractors on most scenes (scene 1: 35 gold, 16 distractors → 35 / 51 / 35); text and both exact on 7 of 16 scenes | one draw of synthetic scenes; not an evaluation |

### Recipe sweep behind the default fine-tune (CPU, 2026-09-24)

Package API on the synthetic scenes; test MAE and box F1 on the held-out test scenes of each arm, frozen → kept
epoch. Validation MAE per epoch (epoch 0 is the frozen model).

| Arm | train / val / test | epochs | lr | validation MAE per epoch | kept | test MAE | test box F1 | test point F1 |
|---|---|---|---|---|---|---|---|---|
| A | 6 / 3 / 6 | 1 | `1e-5` | 0.67 → 0.67 | 0 (frozen) | 9.33 → 9.33 | 0.288 → 0.288 | 0.763 → 0.763 |
| B | 12 / 4 / 8 | 3 | `1e-4` | 4.50 → 4.25 → 3.50 → 3.25 | 3 | 7.25 → 5.62 | 0.350 → 0.381 | 0.836 → 0.868 |
| C (default) | 24 / 8 / 12 | 4 | `2e-4` | 9.38 → 6.88 → 1.50 → 0.75 → 2.75 | 3 | 7.42 → 0.92 | 0.453 → 0.550 | 0.862 → 0.981 |

Arm C's validation MAE turned upward at epoch 4, and the selector kept epoch 3. The arms use different test sets
(their sizes differ), so their rows are not comparable to each other.

## Current status

**Candidate.** The notebook was regenerated on 2026-10-04 for the row-16 review (CGD-M1..M4, CGD-m1..m3: uv
isolated environment, BYOD rules, frozen-base reload before Sections 5–7, guided layer); no hosted run of the new
blob is recorded. The earlier blob `c619a762` (at `8d61b94`, generated from `6ec56df`) ran in a Kaggle Tesla T4
runtime on 2026-09-24 only after a manual restart following the in-kernel install, so it was never a one-pass
`Run all`. Promotion needs a one-pass clean-runtime `Run all` of the current blob (default path, the Section 10
activity, and BYOD with one valid and one refused zip).

## Supplemental open-vocabulary detection and counting workshop — `tutorials/DIMER_Open_Vocabulary_Detection_and_Counting_Workshop.ipynb`

This entry applies only to the supplemental workshop notebook, not the primary tutorial executions above.

### Maintainer-supplied successful Colab run — 2026-09-26

The maintainer supplied the [executed notebook](execution-evidence/2026-09-26/DIMER_Open_Vocabulary_Detection_and_Counting_Workshop.ipynb) and authorized merging PR #3 (merge commit `24496d5`). The file is archived byte-for-byte, SHA-256 `d7cb37ee744ca2ef6f17e1a3854eb2514ec5d776592afeb2bfa1991cf1dccd79`. All 31 code cells have execution counts, 65 saved outputs and zero saved errors. Code-cell sources match commit `2519e0453274c9e1b1565bbb2639d022102d899b`, tutorial blob `6323cbdfac4f437126964e26f5baa1dbcd5d0286`, apart from Colab-inserted `# @title` lines. Later commits on `main` that touch the notebook (`b6b8b11` (AI Use Disclosure)) change only markdown cells; its code cells are identical to the executed revision. This evidence commit does not change tutorial code.

Scope: STANDARD tier: synthetic counting scenes split 24 train / 8 validation / 12 test, CountGD with text, exemplar and text+exemplar prompts, and Grounding DINO Tiny and OWLv2 Base/16 as detector counters. FULL and BYOD were not exercised.

Saved runtime: Python 3.13.15, torch 2.14.0+cu130, torchvision 0.29.0+cu130, Transformers 4.57.6, CUDA Tesla T4. Execution reaches the final completion summary. The separate exported files were not supplied, so their bytes/digests were not independently inspected. Saved counts run sequentially from 1 to 31; runtime freshness and absence of manual restarts/reruns are not independently established by the artifact.

Results (sample-sanity measures on the built-in data, not general model rankings): Detector count thresholds selected on validation: Grounding DINO 0.1 (validation MAE 0.75) and OWLv2 0.3 (validation MAE 0.125); CountGD text+exemplar MAE 6.67 on the three unseen demonstration scenes. The report bundle and completion summary were produced; the notebook itself labels all built-in results synthetic sample-sanity evidence.

Status remains **Candidate**. Merge approval and this successful default-path run do not close the optional-path (FULL/BYOD) or REL12 qualification gates, and `metadata.dimer.clean_runtime_evidence` in the notebook stays `pending` as authored (editing it would change the verified blob).

### Review fixes of 2026-10-02 — not yet executed on a hosted runtime

The 2026-10-02 notebook review (`reviews/2026-10-02-notebook-review/`, reviewed commit `413d558`, notebook blob `b0ddeb6e`) found five Majors (CNT-M1..M5) and seven Minors. Their fixes change code cells: the CountGD and Grounding DINO carrier modules are now carried inline instead of downloaded, the BYOD branch, the controlled-change activity and the checked `FULL` reload parity are new, and `pixel_sha256` now hashes decoded pixels, so `dataset_sha256` differs from the value printed by the 2026-09-26 run. **The 2026-09-26 Colab run above covers the previous blob only.** The fixed revision has CPU evidence only: the notebook's model-free cells and every optional branch executed with stand-in models on a Windows workstation (not clean-runtime evidence). Before promotion, a fresh Colab T4 `Run all` of the fixed revision must record: the `STANDARD` default path, the activity (`RUN_ACTIVITY`), BYOD with one valid and one rejected archive (REL12), and `FULL` with both reload parities.

### 2026-10-04 carried CountGD `pipeline.py` re-pinned (box-head fix) — not yet executed on a hosted runtime

The `countgd_object_counting_colab.ipynb` review (PR #6, CGD-x1) found that the CountGD fine-tune never trained the shared box head: the network reports it as `transformer.decoder.bbox_embed.0.*` and the trained-set prefix was `bbox_embed.0.`. `src/countgd_pipeline/pipeline.py` now matches both (blob `ab839aad` → `557b40b8`, 35,418 → 35,886 bytes, commit `bd38240`), and this workshop's carried copy, its `COUNTGD_SOURCE` pin and `COUNTGD_CARRIER_COMMIT` were updated to it; the other eight carried modules are unchanged. Notebook blob → `4c86f59d`. Effect: the `FULL` tier's CountGD adaptation now trains 64 tensors / 3,752,196 parameters (was 58 / 3,619,584), as the workshop's own text already said, so its `FULL` CountGD numbers and adapter differ from earlier runs; `STANDARD` is unaffected (frozen inference only). **This blob needs its own hosted re-run**; status stays Candidate.

### 2026-10-03 uv isolated environment — not yet executed on a hosted runtime

Notebook blob `6ebf9a2a` → `026c2da7` (workshop revision 0.2.0-candidate → 0.3.0-candidate). The in-kernel `pip install` and its "restart the session" guard are gone. Section 2 verifies a pinned `uv` 0.12.15 wheel, builds a CPython 3.12.12 virtual environment in `work/env`, installs the carried hash lock (`tools/counting-workshop-requirements.lock`, `uv pip compile --generate-hashes` for `x86_64-manylinux_2_28`) with `--require-hashes --only-binary :all:`, and starts one long-lived process with the environment's interpreter (the carried `tools/counting_workshop_env_worker.py`). An IPython input transformer then forwards every later titled code cell to that process, so cell code, data, seeds, models and metrics are unchanged; the kernel only shows the printed text, tables and figures. Pins are the former in-kernel pins (torch 2.14.0, torchvision 0.29.0, torchaudio 2.11.0, transformers 4.57.6, huggingface-hub 0.36.2, safetensors 0.8.0, numpy 2.5.3, pillow 11.3.0, scipy 1.18.1) with the former ranges fixed at matplotlib 3.10.8 and pandas 2.2.3. User-visible: Linux x86_64 only (Colab, Kaggle, Linux Jupyter). Status stays **Candidate**; a hosted re-run of `026c2da7` passed on 2026-10-03 (see the Colab CLI execution record below); the earlier Colab runs recorded above do not cover it. Local CPU check of this blob (Windows workstation, not clean-runtime evidence; stand-ins: platform check disabled, the host interpreter instead of the hash-locked venv, CUDA hidden): `STANDARD` Run all through a real Jupyter kernel, the input transformer and the cell runner completed 49/49 code cells, and all 15 result tables equal those of the pre-uv notebook (`796c2f6`) executed in-process with the same interpreter on the same CPU (`dataset_sha256` `2eae4c91...`, unchanged). A second Run all with `FULL` (1 epoch each instead of 6 and 4, for CPU time), the activity and a valid BYOD archive also completed 49/49, with both reload parities passing. The uv download, venv build and hash-locked install themselves have not run yet (they have since run on Colab; see the record below). Found on the way: in a local Jupyter kernel (IPython 9.17.1) the `796c2f6` notebook stops at the carried-source staging cell with an integrity mismatch for `countgd_pipeline/modeling.py`, because IPython's own input clean-up rewrites whitespace-only lines inside the carried string; the Colab kernel was not affected (its run passed). This revision forwards each cell before those clean-up steps, and a test checks that every forwarded cell arrives byte-for-byte.

### Colab CLI execution of revision 0.3.0-candidate (`a4f8383`, blob `026c2da7`) — 2026-10-03

Executed notebook: [`execution-evidence/2026-10-03/DIMER_Open_Vocabulary_Detection_and_Counting_Workshop_a4f8383_colab-cli-t4.ipynb`](execution-evidence/2026-10-03/DIMER_Open_Vocabulary_Detection_and_Counting_Workshop_a4f8383_colab-cli-t4.ipynb), archived byte-for-byte, SHA-256 `97067f69eb6012e205e94b48755f02641f3bdba13ef5cba443f1366a76338521`.

- **Subject:** `tutorials/DIMER_Open_Vocabulary_Detection_and_Counting_Workshop.ipynb` at PR #5 head `a4f8383686070e64f6b869f91daacdda27f9f217`, blob `026c2da7d8d6645e49875890825633a8a37beffa` (workshop revision 0.3.0-candidate), downloaded from GitHub at the PR head and blob-verified before the session; the input file was unchanged after the run.
- **Executor:** Google Colab CLI 0.7.4 on a fresh Colab Tesla T4 session via the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`; the session was stopped afterwards). Code cells ran in order in one kernel; this is not a browser Run all, and the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log (cells 1–49, sequential and complete).
- **Path exercised:** default `STANDARD` tier with every control at its default. `FULL` adaptation, the activity (`RUN_ACTIVITY`), BYOD (`USE_BYOD`) and the FSC-147 check were not run; their cells printed their skip messages. The BYOD validator's built-in refusal example (no model) ran and refused its demo archive as designed.
- **Outcome:** PASSED, 49/49 code cells, no errors; 270.1 s wall for `colab exec`. 32 code cells saved output (54 outputs, 4 figures, 0 error outputs).
- **Isolated environment (first hosted run of this design):** cell 3 created the CPython 3.12.12 environment and installed the hash lock (`lock_sha256` `045ccc32...`); cell 4, run inside it, reported `/content/work/env/bin/python`, torch 2.14.0+cu130, torchvision 0.29.0+cu130, transformers 4.57.6, numpy 2.5.3, `cuda:0` on a Tesla T4. Cells 1–3 ran in the kernel as marked (`# dimer-runs-in: kernel` / `kernel+env`); the other 46 titled cells were forwarded. Every library warning in the saved outputs comes from `/content/work/env/lib/python3.12/...` and none from the kernel's `/usr/local/lib/python3.13` packages; the kernel-side `HF_TOKEN` Colab-secrets warning and the download progress widgets of earlier runs are absent, consistent with downloads happening in the worker process. No cell's output shows signs of running in the kernel instead of the environment. Remaining warnings are upstream deprecation notices (slow image processor, `torch.meshgrid` indexing, Transformers `device` argument, `torch.cuda.amp.autocast`). The plain-text form of wide DataFrames is column-truncated (`[4 rows x 11 columns]`); the HTML form of each table is complete.
- **Results:** `dataset_sha256` `2eae4c91...` (24/8/12 scenes); CountGD weight SHA-256 `8e44867b...`; carrier commit `8263e1e2`, 9 files verified; Grounding DINO validation AP50 0.990674 raw / 0.990705 common NMS, OWLv2 1.000000; detector count thresholds selected on validation Grounding DINO 0.1 (MAE 0.75) and OWLv2 0.3 (MAE 0.125); test AP50 Grounding DINO 0.992283 raw / 0.994871 common NMS, OWLv2 1.000000; test count MAE mean count 10.000, template matcher 14.500, Grounding DINO as counter 1.250, OWLv2 as counter 1.500, CountGD frozen text / exemplar / text+exemplar 4.667 / 11.667 / 7.417 (box F1 0.500 / 0.489 / 0.453); CountGD text+exemplar MAE 6.667 on the three new scenes (gold 39 / 13 / 34, predicted 48 / 20 / 38); report bundle 17 files, SHA-256 `8aa9baa5...`.
- **Comparison with the 2026-09-26 Colab run (blob `6323cbdf`):** every metric both runs print is equal to the printed precision — validation and test detection tables, prompt-sensitivity table, CountGD validation and test tables, count baselines, both threshold sweeps and selections, the detector-as-counter rows, the density errors for the two detectors, and the new-scene MAE 6.667. Different by design: `dataset_sha256` (`53b17b4c...` → `2eae4c91...`, the 2026-10-02 decoded-pixel hash fix), the bundle digest (new contents and provenance), Python 3.13.15 (kernel) → 3.12.12 (isolated environment), and 31 → 49 code cells. The 2026-09-26 run recorded no wall time.
- **Boundary:** the saved outputs were inspected; the exported bundle files were not separately downloaded. Journeys not exercised here (activity, BYOD with valid and rejected archives (REL12), `FULL` with both reload parities) remain open, as listed under the 2026-10-02 review fixes above.

Status stays **Candidate**. This run does not change the release status, and `metadata.dimer.clean_runtime_evidence` in the notebook stays `pending` as authored.
