# Release verification

`tutorials/vit_mae_pretraining_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`config.py`, `metrics.py`, `model.py`, `provenance.py`, `samples.py`,
  `pipeline.py`, in dependency order), each equal to its source after the generator's documented rewrites (the
  `DEFAULT_WEIGHTS_DIR` rule, the `resolve_weights_path` checkout-convenience line, and the removal of
  package-relative imports); the inline `MANIFEST` equal to the committed 4-entry snapshot manifest and the inline
  `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cells (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md` and `MODEL_CARD.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `ViTMAEPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path, `read_corpus`,
  `build_sample_dataset(corpus, seed=SPLIT_SEED)` / `load_byod_dataset` + `split_dataset`, `validate_dataset` per
  split, `check_split_disjoint`, `observer_overlap`, `class_names`, `write_dataset_csv`, the four dataset refusal
  probes, the ceiling print, the digest-asserted synthetic shapes, `validate_inputs` with the remote-URL refusal
  probe, `reconstruct` under a seeded mask twice and under another seed, `embed`, the sanity checks including the
  pipeline's masked MSE equal to the model's loss, the per-call `evaluation_report` on the drawn shapes,
  `pipe.evaluate_reconstruction` on the frozen model with the fill assertion, `majority_baseline`,
  `colour_neighbour_baseline`, `pipe.fit_probe` / `pipe.evaluate` on the frozen model with the majority assertion,
  `pipe.adapt` with its explicit hyperparameters and the selector assertion, `pipe.evaluate_reconstruction` on the
  validation and test splits after adaptation with the re-score assertion, `pipe.fit_probe` / `pipe.evaluate` after
  adaptation, `reconstruct` + `evaluation_report` on the shapes after adaptation, `pipe.save_artifact`,
  `ViTMAEPipeline.from_artifact` and the reload-parity assertion (reconstructions and probe decisions),
  `write_provenance`, and the result fields `weight_file` / `weight_format` / `weight_sha256` and the `corpus`
  block), the seven expected `outputs/` paths, the learner-facing statements (Apache-2.0 weights, no user-facing
  task output, the masked-patch MSE and the probe as the two readings, continuation of the pre-training objective,
  the CC0 corpus, the non-neural fills and classifiers, no dispersion estimate, the leakage and readout guidance,
  the excluded tasks, the snapshot note) and the gated-off BYOD default; forbidden patterns (credential-in-URL, any
  `git clone` / `github.com` / repository import on the primary path, a mutable `revision='main'`, direct
  `from transformers import` / `AutoModel` / `AutoImageProcessor` / `ViTMAEForPreTraining` / `patchify(` /
  `from huggingface_hub import` / `snapshot_download` / `urllib.request` / `safetensors` imports / `torch.optim` /
  `.backward(` / `requires_grad` / `mask_ratio =` / `torch.softmax(` / `pipe.model.` / `extractall(` use **outside
  the carried module cells**, `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  checkpoint-invariants section.

CI also installs the frozen CPU reference environment (`requirements.lock.txt`), runs `ruff`, `scripts/check_lock.py`,
`tools/build_notebook.py --check`, and the offline unit suite (`tests/`, including `test_pipeline.py`,
`test_adaptation.py`, `test_role_helpers.py`, `test_notebook_parity.py`; no weights, injected downloader and photo
fetcher — `tests/test_model_backed.py` is skipped without the snapshot). These are source/provenance and unit checks.
They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Repository CI integration job (`tools/run_notebook.py`, manual `workflow_dispatch` or push to `main`) | GitHub-hosted Ubuntu runner, the frozen CPU reference environment with `DIMER_NOTEBOOK_CI_PREINSTALLED=1` | Executes the standalone notebook's code cells sequentially against the real pinned weights; a **pre-flight** on the locked stack, not a fresh-boundary run of the inline `PINS` and not promotion evidence on its own |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/vit-mae-base/` or the photograph cache `weights/inat-birds/` (the standalone path writes the
   manifest itself, stages the missing files from the Hub and fetches the pinned photographs from the iNaturalist
   open-data bucket, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `EPOCHS = 5`, `LEARNING_RATE = 1e-5`, `BATCH_SIZE = 8`,
   `TRAINABLE_BLOCKS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`,
   `pillow==11.3.0`, `huggingface-hub==0.36.2` (an interpreter restart after the install is expected where the
   runtime's preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the six carried module cells execute (defining `ViTMAEPipeline`, `verify_snapshot`, `stage_missing_files`,
     `validate_inputs`, `evaluation_report`, `reconstruction_metrics`, `mean_patch_fill`, `blur_fill`,
     `classification_metrics`, `knn_scores`, `majority_baseline`, `colour_neighbour_baseline`, `fetch_corpus`,
     `read_corpus`, `build_sample_dataset`, `validate_dataset`, `check_split_disjoint`, `observer_overlap`,
     `split_dataset`, `load_byod_dataset`, `write_dataset_csv`, `write_provenance` and the ceilings) with no import
     of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting all 4 manifest entries fetched from `facebook/vit-mae-base` at the immutable
     revision on a clean runtime, `verify_snapshot` returning its dict (4 files, the 448 MB `model.safetensors`
     re-hashed), and `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory;
   - Section 4: `fetch_corpus` fetching the 360 pinned photographs with every byte count and SHA-256 matching; the
     seeded split into 216 / 48 / 96 (36 / 8 / 16 per species) with `check_split_disjoint` reporting no shared
     image, the observer overlap and the three dataset digests printed; `outputs/…_train.csv` written; the four
     dataset refusal probes each raising `ValueError`;
   - Section 5: the ceilings (`DEFAULT_MASK_RATIO` 0.75, `NUM_PATCHES` 196, `MIN_IMAGE_SIDE` 16, `MAX_IMAGE_SIDE`
     4096, `MAX_BATCH` 64, `MIN_RECORDS` 8, `MAX_RECORDS` 20000, `MIN_CLASSES` 2) surfaced; the three synthetic
     PPM images generated in code with SHA-256 `b38ff0c9…` / `2e3e6576…` / `f0f4c38c…` (equal to
     `examples/sample-data/SHA256SUMS`); `validate_inputs` writing `outputs/…_input_manifest.json` (verdict
     `accepted`, 147 hidden patches, one recorded rejection finding from the remote-URL probe); `reconstruct` on the
     three shapes (147 hidden patches each, the same mask under the same seed, a different one under another seed,
     the pipeline's mean masked MSE equal to the model's `loss`) and `embed` with every sanity check `True`, the
     three frozen reconstructions written as PNG and the per-call `evaluation_report` verdict `sample-sanity`;
   - Section 6: `pipe.evaluate_reconstruction` on the 96 test photographs (masked MSE ≈ 0.2281, PSNR ≈
     19.3 dB in the build record) against the mean-patch fill (≈ 0.7652) and the blur fill
     (≈ 0.5452) with the cell's assertion that the frozen model is below both; the majority floor
     (accuracy 0.167), the colour nearest neighbour (≈ 0.260), `pipe.fit_probe` on the 216 training
     photographs and `pipe.evaluate` on the test split (probe accuracy ≈ 0.365, macro F1 ≈
     @P:FROZEN_PROBE_F1@, k-NN ≈ 0.219) with the per-species breakdown and the assertion that the probe is
     above the majority floor;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 40,286,464 trainable of 111,907,840 parameters,
     and a 5-epoch history with the validation masked MSE selecting the epoch (`best_epoch` 3
     in the build record; validation masked MSE 0.2335 → 0.2338 → 0.2338 → 0.2332 → 0.2334 → 0.2337) and the cell's assertion that the kept epoch is
     not worse than epoch 0;
   - Section 8: `pipe.evaluate_reconstruction` on the test and validation splits, `pipe.fit_probe` /
     `pipe.evaluate` again, the comparison on reconstruction (four systems) and probe (six systems), the per-species
     breakdown and `outputs/…_evaluation_report.json` written (the cell asserts the kept epoch re-scores to the
     number that selected it and that the adapted model is below the blur fill — test masked MSE 0.2281 →
     0.2280 and probe accuracy 0.365 → 0.365 in the build record);
   - Section 9: the three shapes re-reconstructed by the adapted model under the same masks with the
     `sample-sanity` report, the adapted reconstructions written as PNG, `outputs/…_shapes.json` written;
     `pipe.save_artifact` writing `outputs/…_adapter/{adapter.safetensors,manifest.json}` (174
     tensors, 161,190,912 bytes, probe record with the six classes) and `ViTMAEPipeline.from_artifact`
     reloading it with 8/8 identical masked MSEs and 8/8 identical probe decisions on eight test photographs (the
     cell asserts it); `outputs/provenance.json` and `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the
     model identity and licence, the snapshot block (`weight_file`, `weight_format`, `weight_sha256`), the `corpus`
     block, the inference-contract reports, the comparison, the artifact digest, the reload parity, the runtime
     versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the photograph cache were
   clean, outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or
   applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `vit_mae_pretraining_colab.ipynb` (`E2E`) | `eb708a9` / `06bc11b5` | 2026-09-21 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-vit-mae-pretraining` v1; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130`, `transformers 4.57.6` after, Python 3.12.13, `cuda`) | **PASSED** — 14/14 code cells ok (1 restart after install cell); 370 files, 487 MB staged from the Hub into a clean cache; five epochs in 25.1 s (validation masked MSE 0.2335 → 0.2338 → 0.2338 → 0.2332 → 0.2334 → 0.2337, epoch 3 kept); comparison {reconstruction: {masked_mse: {mean_patch_fill: 0.7652, blur_fill: 0.5452, frozen: 0.2281, adapted: 0.228}, masked_mse_median: {mean_patch_fill: 0.6763, blur_fill: 0.4753, frozen: 0.1801, adapted: 0.1827}, masked_mse_visible: {mean_patch_fill: 0.0, blur_fill: 0.0, frozen: 0.2794, adapted: 0.2789}, psnr_masked: {mean_patch_fill: 14.0797, blur_fill: 15.5516, frozen: 19.3365, adapted: 19.3375}}, reconstruction_delta_vs_frozen: {masked_mse: -5e-05, masked_mse_median: 0.00261, masked_mse_visible: -0.00054, psnr_masked: 0.00104}, validation_masked_mse: {frozen: 0.2335, selected_epoch: 3, selected: 0.2332, rescored: 0.2332}, probe: {accuracy: {majority: 0.167, colour_neighbour: 0.26, knn_frozen: 0.219, knn_adapted: 0.219, frozen: 0.365, adapted: 0.365}, macro_f1: {majority: 0.048, colour_neighbour: 0.261, knn_frozen: 0.216, knn_adapted: 0.216, frozen: 0.365, adapted: 0.365}}, probe_delta_vs_frozen: {accuracy: 0.0, macro_f1: 0.0}, by_species: {american_goldfinch: {n: 16, frozen_recall: 0.69, adapted_recall: 0.69, frozen_f1: 0.56, adapted_f1: 0.56}, chipping_sparrow: {n: 16, frozen_recall: 0.25, adapted_recall: 0.25, frozen_f1: 0.22, adapted_f1: 0.22}, dark_eyed_junco: {n: 16, frozen_recall: 0.38, adapted_recall: 0.38, frozen_f1: 0.36, adapted_f1: 0.36}, house_finch: {n: 16, frozen_recall: 0.31, adapted_recall: 0.31, frozen_f1: 0.45, adapted_f1: 0.45}, song_sparrow: {n: 16, frozen_recall: 0.31, adapted_recall: 0.31, frozen_f1: 0.33, adapted_f1: 0.33}, white_throated_sparrow: {n: 16, frozen_recall: 0.25, adapted_recall: 0.25, frozen_f1: 0.26, adapted_f1: 0.26}}}; reload parity {max_abs_masked_mse_difference: 0.0, identical_reconstructions: 8, identical_probe_decisions: 8, max_abs_probe_score_difference: 1.1920928955078125e-06, of: 8}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-vit-mae-pretraining/v1/evidence/` in the workspace |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/vit_mae_pretraining_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/vit_mae_pretraining_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-21 | `eb708a9` / `06bc11b5` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-vit-mae-pretraining` v1; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130`, `transformers 4.57.6` after, Python 3.12.13, `cuda`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) — this repository's first GPU run | 366.8 s | **PASSED** — 14/14 code cells ok (1 restart after install cell); 370 files, 487 MB staged from the Hub into a clean cache; five epochs in 25.1 s (validation masked MSE 0.2335 → 0.2338 → 0.2338 → 0.2332 → 0.2334 → 0.2337, epoch 3 kept); comparison {reconstruction: {masked_mse: {mean_patch_fill: 0.7652, blur_fill: 0.5452, frozen: 0.2281, adapted: 0.228}, masked_mse_median: {mean_patch_fill: 0.6763, blur_fill: 0.4753, frozen: 0.1801, adapted: 0.1827}, masked_mse_visible: {mean_patch_fill: 0.0, blur_fill: 0.0, frozen: 0.2794, adapted: 0.2789}, psnr_masked: {mean_patch_fill: 14.0797, blur_fill: 15.5516, frozen: 19.3365, adapted: 19.3375}}, reconstruction_delta_vs_frozen: {masked_mse: -5e-05, masked_mse_median: 0.00261, masked_mse_visible: -0.00054, psnr_masked: 0.00104}, validation_masked_mse: {frozen: 0.2335, selected_epoch: 3, selected: 0.2332, rescored: 0.2332}, probe: {accuracy: {majority: 0.167, colour_neighbour: 0.26, knn_frozen: 0.219, knn_adapted: 0.219, frozen: 0.365, adapted: 0.365}, macro_f1: {majority: 0.048, colour_neighbour: 0.261, knn_frozen: 0.216, knn_adapted: 0.216, frozen: 0.365, adapted: 0.365}}, probe_delta_vs_frozen: {accuracy: 0.0, macro_f1: 0.0}, by_species: {american_goldfinch: {n: 16, frozen_recall: 0.69, adapted_recall: 0.69, frozen_f1: 0.56, adapted_f1: 0.56}, chipping_sparrow: {n: 16, frozen_recall: 0.25, adapted_recall: 0.25, frozen_f1: 0.22, adapted_f1: 0.22}, dark_eyed_junco: {n: 16, frozen_recall: 0.38, adapted_recall: 0.38, frozen_f1: 0.36, adapted_f1: 0.36}, house_finch: {n: 16, frozen_recall: 0.31, adapted_recall: 0.31, frozen_f1: 0.45, adapted_f1: 0.45}, song_sparrow: {n: 16, frozen_recall: 0.31, adapted_recall: 0.31, frozen_f1: 0.33, adapted_f1: 0.33}, white_throated_sparrow: {n: 16, frozen_recall: 0.25, adapted_recall: 0.25, frozen_f1: 0.26, adapted_f1: 0.26}}}; reload parity {max_abs_masked_mse_difference: 0.0, identical_reconstructions: 8, identical_probe_decisions: 8, max_abs_probe_score_difference: 1.1920928955078125e-06, of: 8}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-vit-mae-pretraining/v1/evidence/` in the workspace |
| 2026-09-21 | `2e03553` (the generated notebook at that revision; the committed blob differs only by this row and one prose sentence, the carried modules are identical) | Build workstation CPU via `tools/run_notebook.py` (`CUDA_VISIBLE_DEVICES=-1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, Python 3.12.10, torch 2.14.0, transformers 4.57.6; snapshot and photographs pre-staged) | The generated notebook's code cells executed sequentially top-to-bottom (the CI pre-flight path): 4 manifest files verified, 360 photographs re-hashed, 216 / 48 / 96, every sanity check `True`, frozen test masked MSE 0.2281 / probe 0.365, five epochs (validation 0.2335 → 0.2338 → 0.2338 → 0.2332 → 0.2334 → 0.2337, epoch 3 kept, 463 s with the workstation busy), adapted test masked MSE 0.2280 / probe 0.365, drawn shapes masked MSE 0.0005 / 0.0113 / 0.0228 → 0.0006 / 0.0117 / 0.0219 under the same masks, adapter 174 tensors / 161,190,912 B, reload parity 0.0 and 8/8 probe decisions; all 14 `outputs/` artifacts written | not timed per cell | PASS — pre-flight only (no clean runtime, no GPU); not promotion evidence |
| 2026-09-21 | package API, not the notebook (source at the revision that generated the first committed blob) | Build workstation CPU (`CUDA_VISIBLE_DEVICES=-1`, Python 3.12, torch 2.14.0, transformers 4.57.6; snapshot and the 360 photographs pre-staged) | The notebook's default path replayed cell by cell through the package API (`build_sample_dataset(seed=42)` → 216 / 48 / 96, `validate_dataset` per split, `check_split_disjoint`, `pipe.evaluate_reconstruction` frozen with the two fills, `majority_baseline`, `colour_neighbour_baseline`, `pipe.fit_probe` / `pipe.evaluate` frozen with the k-NN, `pipe.adapt` at the defaults, `pipe.evaluate_reconstruction` and the probe adapted, `save_artifact`, `from_artifact` with reconstruction and probe parity): model load 1.4 s, frozen test reconstructed in 4.4 s and probed in 28.8 s, 5 epochs 159.1 s (validation masked MSE 0.2335 → 0.2338 → 0.2338 → 0.2332 → 0.2334 → 0.2337, epoch 3 kept), adapter 161,190,912 B / 174 tensors, reload max abs masked-MSE difference 0 and 8/8 identical probe decisions; comparison {reconstruction: {masked_mse: {mean_patch_fill: 0.7652, blur_fill: 0.5452, frozen: 0.2281, adapted: 0.228}, masked_mse_median: {mean_patch_fill: 0.6763, blur_fill: 0.4753, frozen: 0.1801, adapted: 0.1827}, masked_mse_visible: {mean_patch_fill: 0.0, blur_fill: 0.0, frozen: 0.2794, adapted: 0.2789}, psnr_masked: {mean_patch_fill: 14.0797, blur_fill: 15.5516, frozen: 19.3365, adapted: 19.3375}}, reconstruction_delta_vs_frozen: {masked_mse: -5e-05, masked_mse_median: 0.00261, masked_mse_visible: -0.00054, psnr_masked: 0.00104}, validation_masked_mse: {frozen: 0.2335, selected_epoch: 3, selected: 0.2332}, probe: {accuracy: {majority: 0.167, colour_neighbour: 0.26, knn_frozen: 0.219, knn_adapted: 0.219, frozen: 0.365, adapted: 0.365}, macro_f1: {majority: 0.048, colour_neighbour: 0.261, knn_frozen: 0.216, knn_adapted: 0.216, frozen: 0.365, adapted: 0.365}}, probe_delta_vs_frozen: {accuracy: 0.0, macro_f1: 0.0}, by_species: {american_goldfinch: {n: 16, frozen_recall: 0.69, adapted_recall: 0.69, frozen_f1: 0.56, adapted_f1: 0.56}, chipping_sparrow: {n: 16, frozen_recall: 0.25, adapted_recall: 0.25, frozen_f1: 0.22, adapted_f1: 0.22}, dark_eyed_junco: {n: 16, frozen_recall: 0.38, adapted_recall: 0.38, frozen_f1: 0.36, adapted_f1: 0.36}, house_finch: {n: 16, frozen_recall: 0.31, adapted_recall: 0.31, frozen_f1: 0.45, adapted_f1: 0.45}, song_sparrow: {n: 16, frozen_recall: 0.31, adapted_recall: 0.31, frozen_f1: 0.33, adapted_f1: 0.33}, white_throated_sparrow: {n: 16, frozen_recall: 0.25, adapted_recall: 0.25, frozen_f1: 0.26, adapted_f1: 0.26}}} | 256.8 s of model time | PASS — pre-flight only (no notebook, no GPU); not promotion evidence |

### Learning-rate sweep behind the default recipe (CPU, 2026-09-21)

Five arms of `pipe.adapt` on the same split, seed and masks, each 5 epochs (arm A 3), batch 8, selected on the
validation masked MSE (frozen 0.2335 on the 48 validation photographs; test 0.2281 frozen):

| Arm | lr | blocks | validation masked MSE per epoch | kept | test masked MSE after | probe accuracy after |
|---|---|---|---|---|---|---|
| A | `1e-4` | 2 | 0.2335 → 0.2475 → 0.2398 → 0.2432 | 0 (frozen) | 0.2281 | 0.365 |
| B | `1e-5` | 2 | 0.2335 → 0.2338 → 0.2338 → 0.2332 → 0.2334 → 0.2337 | 3 | 0.2280 | 0.365 |
| C | `3e-5` | 2 | 0.2335 → 0.2355 → 0.2349 → 0.2341 → 0.2353 → 0.2353 | 0 (frozen) | 0.2281 | 0.365 |
| D | `1e-5` | 0 (decoder only) | 0.2335 → 0.2339 → 0.2338 → 0.2333 → 0.2336 → 0.2338 | 3 | 0.2280 | 0.365 |
| E | `1e-5` | 4 | 0.2335 → 0.2336 → 0.2337 → 0.2331 → 0.2333 → 0.2336 | 3 | 0.2280 | 0.365 |

Reading: on 216 natural photographs the checkpoint is already at its objective's floor; `1e-4` and `3e-5` hurt from
the first epoch (batches of eight with fresh random masks are a noisy gradient) and the selector keeps the frozen
model, `1e-5` moves the validation MSE by 0.0003 and the probe not at all. Arm B is the default (two blocks, the
fleet's convention for a bounded adaptation) — chosen as the configuration that did not hurt, and documented as such.

## Current status

**Release-grade.** The `E2E` notebook blob `06bc11b5` (committed at `eb708a9`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-21 (14/14 ok (1 restart after install cell), 366.8 s, 370 files, 487 MB fetched from the Hub and digest-verified inside the notebook) with no repository checkout — the REL1/REL10 supported-runtime evidence this file gates on; every number it produced reproduced the CPU records above to the printed precision. The CPU rows are what preceded it and remain history. Any later change to the carried modules or to the notebook produces a new blob, and the registry returns to **Candidate** until a clean run of that blob is recorded here.

Facts a reviewer should weigh: the frozen model reconstructs hidden patches far below both non-neural fills (masked
MSE 0.2281 against 0.5452 and 0.7652) and its features lift a linear probe over six bird
species above the majority floor, the colour neighbour and a k-NN (0.365 against 0.167,
0.260 and 0.219) while remaining, as the paper says of MAE features, a poor linear-probe backbone; a
bounded continuation of the pre-training objective on 216 in-domain photographs moves the held-out masked MSE from
0.2281 to 0.2280 and the probe from 0.365 to 0.365 — the honest
reading of continued pre-training on a converged model, which is why the tutorial asserts on the selector rather than
on a gain and points BYOD at unfamiliar domains; the 48-photograph validation split moves the selection metric in the
fourth decimal, which is what "no dispersion estimate" means here; and the drawn shapes re-reconstructed after
adaptation are three images of evidence about behaviour outside the corpus, not a measurement.
