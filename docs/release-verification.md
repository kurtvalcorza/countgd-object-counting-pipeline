# Release verification

`tutorials/countgd_object_counting_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
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
  path; one cell per carried module (`config.py`, `metrics.py`, `modeling.py`, `synthetic.py`, `model.py`,
  `provenance.py`, `samples.py`, `pipeline.py`, in dependency order), each equal to its source after the
  generator's documented rewrites (the two `__file__` uses in `model.py` made working-directory-relative, and the
  removal of package-relative imports); the inline `MANIFEST` and `TOKENIZER_MANIFEST` equal to the committed
  snapshot manifests and the inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical
  (on LF) to `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
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
   `numpy==2.5.3`, `pillow==11.3.0`, `scipy==1.18.1`, `huggingface-hub==0.36.2` (an interpreter restart after
   the install is expected where the runtime's preinstalled torch or numpy differ from the pins);
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
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 3,619,584 trainable of 233,362,816 parameters,
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
| `countgd_object_counting_colab.ipynb` (`E2E`) | — | — | — | no clean hosted-runtime execution recorded yet |

## Recorded executions

Each row is one execution: what ran, where, and what was observed. Static checks are not listed here.

| Date (UTC) | Subject | Runtime | Procedure | Observed result | Caveats |
|---|---|---|---|---|---|
@P:LOCAL_NOTEBOOK_ROW@
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

**Candidate.** The notebook has executed top-to-bottom on the build workstation's CPU (the local harness, a
pre-flight). No clean hosted-runtime execution of the committed notebook blob is recorded yet.
