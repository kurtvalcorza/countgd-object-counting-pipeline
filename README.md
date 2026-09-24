# CountGD Open-World Object Counting Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/countgd-object-counting-pipeline/blob/main/tutorials/countgd_object_counting_colab.ipynb)

DIMER-oriented inference and bounded fine-tuning wrapper for **one immutable open-weight CountGD checkpoint** — the
multi-modal open-world counter of Amini-Naieni, Han and Zisserman, *CountGD: Multi-Modal Open-World Counting*
(NeurIPS 2024). Given an image and a **text prompt**, up to three **exemplar boxes**, or both, it returns a count
with one box and one point per counted object. The network is GroundingDINO (Swin-B image backbone, BERT text
encoder, feature enhancer, six-layer decoder with 900 queries) with the exemplars entered as extra prompt
tokens, carried here as vendored PyTorch code with the multi-scale deformable attention in pure PyTorch, so no
compiled CUDA extension is needed on CPU or GPU.

- model: the authors' Hugging Face Space `nikigoli/countgd`
- pinned revision: `6e82e59569a84ee5c6aafa35d396f2d2bee57be2`
- source checkpoint: `checkpoint_best_regular.pth`, 1,250,122,522 bytes, SHA-256 `c1bab864b17db345b4c6e3aaabb5765bc2c0a90d0bc8defb5e664a74a50aa126` (byte-identical to the Google Drive file the upstream README links)
- served weight file: `countgd.safetensors`, 937,560,480 bytes, SHA-256 `8e44867b951e3a4205d918e022b78bc5fea218fd17c1851b864a01c421d2d443` (converted once from the audited pickle)
- upstream code: `niki-amini-naieni/CountGD` at `b6f362b3f5cd20db4a171faa410dfed8f2f466d8`
- upstream model license: MIT

The wrapper code in this repository is MIT licensed. The weights retain the authors' MIT licence; the vendored
GroundingDINO code carries IDEA's Apache-2.0 header.

## Status

**Release-grade.** The inference, evaluation and adaptation contracts and the real pinned checkpoint have been
exercised on the build workstation's CPU (the unit and model-backed suites, and the tutorial notebook in a fresh
kernel) and — for the `E2E` tutorial at blob `c619a762` — in a clean Kaggle Tesla T4 runtime on 2026-09-24
(recorded in `docs/release-verification.md`). A later notebook revision returns to Candidate until a clean-runtime
execution of that exact blob is recorded. Production HTTP serving and DIMER worker packaging remain a separate
serving-readiness milestone.

## Capabilities

```python
from countgd_pipeline import CountGDPipeline, synthetic_scene

pipe = CountGDPipeline.from_pretrained()          # fetch → audit → convert → verify → load (CUDA when visible)
scene = synthetic_scene(1)                        # 35 blue circles among 16 green triangles
result = pipe.count(scene["image"], text="blue circle", exemplars=scene["exemplars"])
entry = result["results"][0]
entry["count"], entry["boxes"][:2], entry["points"][:2], entry["max_score"]
```

- `count(images, text=None, exemplars=None, threshold=0.23)` — one result per image: `count`, `boxes`
  (`[x0, y0, x1, y1]` in input pixels, best first), `points` (box centres), `scores`, `max_score` over the 900
  queries, and the prompt as validated. Text alone, exemplars alone, or both, as upstream allows.
- `evaluate(records)` — counts every record with its own label and exemplars and reports the count error (MAE,
  RMSE, NAE, exact fraction, per class), point localisation (one-to-one matching of predicted to gold points
  within half the mean exemplar side) and, where gold object boxes exist, box IoU matching (precision, recall
  and F1 at IoU ≥ 0.5, mean matched IoU, mean best IoU per gold box).
- `validate_inputs(...)` — the checks `count` applies, returned as an input manifest.
- `mean_count_baseline`, `template_matching_baseline` — two non-neural baselines scored by the same code.
- `synthetic_scene(seed)`, `build_synthetic_dataset()` — seeded counting scenes with known object boxes.
- `fetch_corpus()`, `read_corpus()`, `build_sample_dataset()` — 80 digest-pinned FSC-147 test photographs.

## Adaptation contract

```python
from countgd_pipeline import CountGDPipeline, build_synthetic_dataset

splits = build_synthetic_dataset()                # 24 / 8 / 12 scenes with gold boxes
# or: splits = split_dataset(load_byod_dataset("my_images.zip"), seed=42)  # labels.csv: id, file, label, count[, points, exemplars, boxes]
pipe = CountGDPipeline.from_pretrained()
report = pipe.adapt(splits["train"], splits["validation"], epochs=4, lr=2e-4)
pipe.save_artifact("outputs/adapter")
reloaded = CountGDPipeline.from_artifact("outputs/adapter")
```

`adapt` trains the last `trainable_layers` decoder layers (two by default), the decoder's final LayerNorm and the
shared box head — 3,619,584 of 233,362,816 parameters — on upstream's objective: the token sigmoid focal loss and
the L1 box loss after Hungarian matching, over the final and every intermediate decoder output. Records with
`boxes` train on their gold object boxes; records with points only train on upstream's 2 × 2-pixel boxes centred
on the points. One image per step, AdamW (weight decay 1e-4), gradient clipping at 0.1, seeded order. Epoch 0
scores the frozen model on the validation set; the epoch with the lowest validation MAE is kept, which can be the
frozen model itself. Any failure restores the base tensors. `save_artifact` writes the trained tensors as
`adapter.safetensors` with a manifest (format, base model id, revision and weight digest, tensor names, file
digest, configuration, history); `load_artifact` / `from_artifact` verify the manifest, the digest and the exact
tensor set before deserialising.

## Live tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/countgd-object-counting-pipeline/blob/main/tutorials/countgd_object_counting_colab.ipynb)

`tutorials/countgd_object_counting_colab.ipynb` is declared `E2E` under DIMER Notebook Specification 2.0 and is
**standalone** (§4): generated by `tools/build_notebook.py`, it carries the package's eight modules, the model
identity, both snapshot manifests and the runtime pins, so the exported notebook runs without this repository
(parity enforced by `tests/test_notebook_parity.py` and `tools/validate_release_assets.py`). It stages and
digest-verifies the snapshots, audits and converts the pickle checkpoint in the kernel, draws the synthetic
scenes and fetches the FSC-147 photographs, counts the demo scene by text, by exemplars and by both, scores the
frozen model beside the mean-count baseline and the template matcher on the held-out scenes and photographs, runs
the bounded fine-tune with validation-MAE epoch selection, scores both held-out sets again, re-counts the demo
scene, exports the adapter and reloads it with verified parity, and writes:

- `countgd_object_counting_train.csv`
- `countgd_object_counting_input_manifest.json`
- `countgd_object_counting_demo.json` (+ the frozen and adapted counts of the demo scene drawn as PNG, one per prompt mode)
- `countgd_object_counting_evaluation_report.json`
- `countgd_object_counting_adapter/` (`adapter.safetensors`, `manifest.json`)
- `countgd_object_counting_result.json`
- `provenance.json`

The build record (CPU, `docs/release-verification.md`): on the 12 held-out synthetic scenes the frozen model
counts with MAE 7.42 and box F1 0.453, and the fine-tuned one with MAE
0.92 and box F1 0.550; on the 24 FSC-147 photographs MAE 4.54 →
3.50. These are one seeded draw of synthetic scenes and 24 photographs — evidence that the
contracts work, not a benchmark or production-fitness evidence. See `tutorials/README.md` for the registry.

## Release status

**Release-grade** — the `E2E` notebook blob `c619a762` (committed at `8d61b94`) executed top-to-bottom in a clean
Kaggle Tesla T4 runtime on 2026-09-24 (16/16 code cells ok, one restart after the install cell, 468.7 s); the record
is in `docs/release-verification.md` and `STATUS.md`. On the GPU the fine-tune kept epoch 2 rather than epoch 3
(synthetic test MAE 7.42 → 1.58; FSC-147 4.54 → 3.54), because CUDA kernels are not bit-deterministic. Static and unit checks — including the standalone generator
parity checks — are necessary but are not the evidence. A later change to the carried modules or the notebook
yields a new blob that returns the status to Candidate until its own clean run is recorded.

## Score semantics

A query's score is the sigmoid of its best similarity to the prompt tokens (the text's word pieces and the
exemplar tokens). It is **not calibrated** and **not a probability**: the count is the number of queries whose
score exceeds a fixed threshold, 0.23 by default (upstream's). Lowering the threshold counts more, raising it
fewer; no threshold is tuned to your images. The predicted boxes are object-sized — upstream fine-tuned on
FSC-147 with 2 × 2-pixel boxes at the annotated points, but the GroundingDINO initialisation still yields boxes
that enclose the counted objects — and are scored by IoU where gold object boxes exist and read as points
otherwise.

## Machine-readable provenance

```python
from countgd_pipeline import CountGDPipeline, build_provenance, write_provenance

pipe = CountGDPipeline.from_pretrained()
record = build_provenance(pipeline=pipe)
write_provenance("outputs/provenance.json", pipeline=pipe)
```

The record includes the model id, repository type and immutable revision, the served weight file with its
SHA-256 and size, the source checkpoint's SHA-256 and pickle-audit digest, the verified checkpoint source and
path, the image and tokenizer contract, the counting and score semantics, the adapter record when one is
loaded, Python version, platform, and runtime package versions.

## Input safety

Image inputs may be:

- a local filesystem path;
- `bytes` containing an image;
- a `PIL.Image.Image`.

Remote `http://` and `https://` image strings are rejected intentionally; the pipeline does not act as a network
fetcher. Images with a side outside 32..4096 px are rejected before any tensor work, a call takes at most 16
images, a text prompt is at most 64 plain characters, and at most three exemplar boxes are accepted, each inside
the image and at least 2 px on a side.

## Supply-chain controls

`CountGDPipeline.from_pretrained()`:

1. resolves the weights directory through `weights_dir` / `weights_path`, `COUNTGD_WEIGHTS_DIR`, the checkout's
   `weights/countgd`, or the pinned checkpoint downloaded from the Space into a cache directory;
2. verifies the snapshot against `dimer-base-manifest.json` and the package's pinned source digest (the
   constants win over an edited manifest);
3. refuses any unsafe serialized file (`.bin`, `.pt`, `.pth`, `.ckpt`, `.pkl`, `.pickle`, `.h5`, `.msgpack`) other
   than the audited source checkpoint;
4. when `countgd.safetensors` is absent, audits the pickle statically (five allowed globals, pinned audit
   digest), opens it once with torch's restricted unpickler and writes the converted file;
5. verifies the converted file's exact byte size and SHA-256 — a converted file that fails is refused, not
   regenerated — and loads it into the vendored network with `strict=True`.

The tokenizer snapshot (`google-bert/bert-base-uncased` at `86b5e0934494bd15c9632b12f734a8a67f723594`) is
verified against its own manifest; BERT's weights come from the CountGD checkpoint. No `trust_remote_code`, no
Hub-hosted code. Details: `docs/WEIGHTS.md`.

```bash
python scripts/fetch_weights.py                  # stage, verify, convert
python scripts/fetch_weights.py --verify-only    # re-verify source and converted file
python scripts/fetch_weights.py --zip countgd-dimer.zip   # DIMER upload archive: the converted file only
```

## Reproducible reference environment

Python 3.12 is the supported runtime. The repository keeps exact direct pins in `pyproject.toml` and a fully
version-pinned Linux/CPU reference graph in `requirements.lock.txt`.

```bash
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps --no-build-isolation -e .
python scripts/check_lock.py
```

`requirements.lock.txt` records the exact dependency versions of the CPU reference environment, including the
official CPU PyTorch wheel. It is a version lock, not a cryptographic hash lock.

## Tests

```bash
ruff check src tests tools scripts
pytest -m "not integration"
```

`tests/test_model_backed.py` (the three prompt modes on the demo scene, box-scored evaluation, a one-epoch
fine-tune with an artifact round trip, the no-validation rule, the artifact tensor-set refusal and the
transactional restore, all on the CPU; a CUDA variant runs only with `COUNTGD_TEST_CUDA=1`) runs only when
`weights/countgd/countgd.safetensors` has been produced by `scripts/fetch_weights.py`. The FSC-147 cache
`weights/fsc147-subset/` is git-ignored and filled by `fetch_corpus()`.

Real-checkpoint integration:

```bash
RUN_INTEGRATION=1 pytest -m integration -q
DIMER_NOTEBOOK_CI_PREINSTALLED=1 python tools/run_notebook.py tutorials/countgd_object_counting_colab.ipynb
```

## Scope boundaries

This repository does **not** claim to provide:

- object detection or segmentation as a product (the boxes are a reading of what was counted);
- counting by density map, in video, or with tracking;
- the upstream test-time cropping for images with more objects than one 800-pixel pass resolves, or the
  upstream SAM-based test-time normalisation;
- calibrated confidence;
- fine-tuning beyond the last decoder layers, the decoder norm and the shared box head, or training from scratch;
- evaluation on the full FSC-147 benchmark (the tutorial scores 24 of its test photographs);
- production HTTP serving or DIMER worker packaging.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code
development and technical writing under maintainer direction. The maintainer remains responsible for reviewing
the implementation, validating results, and making release decisions. AI assistance does not constitute
independent verification, provider endorsement, or release approval.
