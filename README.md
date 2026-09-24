# ViT-MAE Base Pre-training Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/vit-mae-pretraining-pipeline/blob/main/tutorials/vit_mae_pretraining_colab.ipynb)

DIMER-oriented inference and bounded continuation wrapper for **one immutable open-weight ViT-MAE checkpoint** — the ViT-B/16 masked autoencoder of He, Chen, Xie, Li, Dollár and Girshick's *Masked Autoencoders Are Scalable Vision Learners* (CVPR 2022), pre-trained on ImageNet-1K for 1,600 epochs and served through `transformers.ViTMAEForPreTraining` (encoder **and** decoder). The model has **no user-facing task output**: it reconstructs the pixels of randomly hidden patches and returns its own loss, and its encoder is a backbone for downstream fine-tuning. This repository reads it two ways — the **masked-patch MSE** of its reconstructions and the **accuracy of a linear probe** on its mean-pooled patch tokens — and exposes a bounded continuation of the pre-training objective on a photograph set:

- model: `facebook/vit-mae-base`
- pinned revision: `25b184bea5538bf5c4c852c79d221195fdd2778d`
- weight file: `model.safetensors`
- expected SHA-256: `479dcef4bd5df06259399027b789f21e9d9a1b79f37155a64176d55bc26fdae8`
- expected size: `447,670,680` bytes
- upstream model license: Apache-2.0

The wrapper code in this repository is MIT licensed. The model weights retain Meta's Apache-2.0 license.

## Status

**Release-grade.** The inference contract, the adaptation contract and the real pinned checkpoint have been exercised on the build workstation's CPU (the unit and model-backed suites, the default tutorial path through the package API and the generated notebook itself) and — for the `E2E` standalone tutorial at blob `06bc11b5` — in a clean Kaggle Tesla T4 runtime on 2026-09-21 (recorded in `docs/release-verification.md`). A later notebook revision returns to Candidate until a clean-runtime execution of that exact blob is recorded. Production HTTP serving / DIMER worker packaging remains a separate serving-readiness milestone.

## Capabilities

```python
from vit_mae_pipeline import load_pipeline

pipe = load_pipeline()

result = pipe.reconstruct("photo.jpg", seed=0)            # hides a seeded 75 % of the 196 patches
entry = result["results"][0]
entry["masked_mse"], entry["visible_mse"], entry["hidden_patches"]   # the model's own loss on the hidden patches
entry["reconstruction"].save("reconstruction.png")       # decoder output composited over the visible patches
result["model_loss"]                                     # == the mean masked MSE: the checkpoint's objective

features = pipe.embed(["a.jpg", "b.jpg"])["embeddings"]  # (2, 768) mean of the patch tokens, nothing hidden, L2-normalised
```

Public inference operations:

- `reconstruct(images, *, mask_ratio=0.75, seed=0, return_images=True)`
- `embed(images)`
- `validate_inputs(images, *, mask_ratio, names)` — the input manifest, raising exactly what the operations raise
- `evaluation_report(result, *, sample_kind)` — the per-call reading of a `reconstruct` result

## Adaptation contract

```python
from vit_mae_pipeline import (
    ViTMAEPipeline, build_sample_dataset, fetch_corpus, load_byod_dataset, read_corpus, split_dataset,
)

splits = build_sample_dataset(read_corpus(fetch_corpus()), seed=42)   # 360 CC0 iNaturalist bird photographs, 216 / 48 / 96
# or: splits = split_dataset(load_byod_dataset("my_photos.zip"), seed=42)  # labels.csv: id, file, label

pipe = ViTMAEPipeline.from_pretrained(weights_dir="weights/vit-mae-base")   # CUDA when visible
frozen = pipe.evaluate_reconstruction(splits["test"], seed=0)      # masked_mse, psnr_masked, per_image, baselines (two fills)
pipe.fit_probe(splits["train"])                                    # linear probe on standardised encoder features
probe = pipe.evaluate(splits["test"])                              # accuracy, macro_f1, per_class, knn
result = pipe.adapt(splits["train"], splits["validation"], epochs=5, lr=1e-5, trainable_blocks=2)   # labels unused
adapted = pipe.evaluate_reconstruction(splits["test"], seed=0)
pipe.fit_probe(splits["train"]); pipe.evaluate(splits["test"])    # the probe is re-fitted on the adapted features
pipe.save_artifact("outputs/adapter")                             # adapter.safetensors (+ probe head) + manifest.json
again = ViTMAEPipeline.from_artifact("outputs/adapter", weights_dir="weights/vit-mae-base")
```

- `validate_dataset(records, *, require_labels=True)` checks `{id, image, label}` records structurally (decodable image with sides in 16..4096 px, label of at most 64 plain characters, 8..20,000 records over 2..100 labels, unique ids) and returns a manifest with a dataset digest; the reconstruction contract accepts unlabelled records (`require_labels=False`). `split_dataset` is a seeded, stratified, pixel-digest-deduplicated split; `check_split_disjoint` asserts no image is shared.
- `evaluate_reconstruction(records, *, mask_ratio=0.75, seed=0)` hides one seeded mask per record (the seed is derived from the record id) and returns the mean and median **masked MSE** in the processor's normalised pixel space, the visible-patch MSE, the **PSNR** of the hidden patches, the per-image rows, and the same metrics for two non-neural fills scored on the same masks — the **mean-patch fill** and the **blur fill** (`metrics.py`).
- `fit_probe(train, *, steps=300, lr=0.01, weight_decay=1e-3, seed=0)` embeds the training set with nothing hidden, standardises the features with the training set's mean and standard deviation (the paper's affine-free BatchNorm) and fits a multinomial logistic-regression head by full-batch Adam; `evaluate(records, *, k=5)` returns accuracy, macro F1, per-class precision / recall / F1, `predictions`, a **k-NN** vote on the same features, `verdict` (`measured` / `small-sample`) and `adapted`; `classify(images)` returns the probe's softmax scores. `majority_baseline` and `colour_neighbour_baseline` are the two further non-neural references the tutorial scores beside the probe.
- `adapt(train, val=None, *, epochs=5, lr=1e-5, batch_size=8, trainable_blocks=2, mask_ratio=0.75, seed=0, progress=None)` continues masked autoencoding on the checkpoint's own loss — a fresh seeded random mask every step — training the whole decoder, the last `trainable_blocks` encoder blocks and the encoder's final LayerNorm (40,286,464 of 111,907,840 parameters by default); AdamW with weight decay 0.05, gradient clipping at 1.0, seeded shuffling and masks, no scheduler, **no labels**. Epoch 0 records the frozen model's validation reconstruction; the epoch with the lowest validation masked MSE is kept, so the selector can return the frozen model itself and never returns a worse one (the final epoch without validation). The update is transactional: an exception restores the frozen weights. The fitted probe is discarded, because its features no longer exist.
- `save_artifact(dir)` writes the trained tensors — and the probe head with its standardisation statistics when a probe is fitted — as `adapter.safetensors` plus a `manifest.json` (format `org.valcorza.vit-mae-base.adapter.v1`: base id, revision and weight digest, objective, mask ratio and trainable blocks, the probe record, tensor names, file size and SHA-256, training configuration, epoch history); `from_artifact(dir)` re-verifies the base snapshot, checks the manifest, the digest and the exact tensor set before deserialising, refuses any tensor outside the declared blocks, and overlays the tensors onto a freshly loaded base (a reloaded probe classifies but carries no training features, so `evaluate()["knn"]` is `None`).

## Live tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/vit-mae-pretraining-pipeline/blob/main/tutorials/vit_mae_pretraining_colab.ipynb)

`tutorials/vit_mae_pretraining_colab.ipynb` is declared `E2E` under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the package's six modules, the model identity (`facebook/vit-mae-base` at the immutable revision `25b184bea5538bf5c4c852c79d221195fdd2778d`), the 4-file manifest digests and the runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py` and `tools/validate_release_assets.py`). It stages and digest-verifies the snapshot, fetches 360 digest-pinned CC0 iNaturalist photographs of six bird species and splits them per species without leakage, masks and reconstructs the three deterministic sample shapes through the inference contract with an input manifest and a rejection probe, measures the frozen model's masked-patch MSE on the 96 held-out photographs beside the mean-patch and blur fills and its linear-probe accuracy beside the majority floor, a colour nearest neighbour and a k-NN, runs a bounded continuation of the masked-autoencoding objective with validation-MSE epoch selection, reconstructs and probes the held-out split again, re-reconstructs the shapes, exports the adapter with the probe head and reloads it with verified parity, and writes:

- `vit_mae_pretraining_train.csv`
- `vit_mae_pretraining_input_manifest.json`
- `vit_mae_pretraining_evaluation_report.json`
- `vit_mae_pretraining_shapes.json` (+ the frozen and adapted reconstructions of the three shapes as PNG)
- `vit_mae_pretraining_adapter/` (`adapter.safetensors`, `manifest.json`)
- `vit_mae_pretraining_result.json`
- `provenance.json`

The default path runs on CPU and uses CUDA automatically when present (about four minutes of model time on the build workstation's CPU after the downloads — 159.1 s of it the 5 continuation epochs — longer on a 2-vCPU hosted runtime; a hosted T4 finishes in a few minutes). The metrics it prints are one seeded split of one 360-photograph sample — evidence that the adaptation contract works, not a benchmark or production-fitness evidence. The build record's own finding is that continued pre-training on 216 in-domain photographs barely moves a converged checkpoint (held-out masked MSE 0.2281 → 0.2280; a learning rate of `1e-4` hurts from the first epoch), which is why the tutorial's assertion is on the selector — the kept epoch is never worse than the frozen model — rather than on a gain. The `main` integration workflow executes the notebook's code cells on the frozen CPU reference environment as a pre-flight; see `tutorials/README.md` for the registry and `docs/release-verification.md` for the release gate.

## Release status

**Release-grade** — the `E2E` notebook blob `06bc11b5` (committed at `eb708a9`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-21 (14/14 ok (1 restart after install cell), 366.8 s); the record is in `docs/release-verification.md` and `STATUS.md`. Static and unit checks — including the standalone generator parity checks — are necessary but were never the evidence; the hosted run is. A later change to the carried modules or the notebook returns the status to Candidate until re-verified.

## Loss semantics

The pipeline's `masked_mse` is the checkpoint's own training objective: the mean squared error between the decoder's prediction and the true pixels over the patches hidden from the encoder, computed in the processor's normalised pixel space (ImageNet mean / std, `norm_pix_loss` off in this checkpoint). `reconstruct()["model_loss"]` is the value `ViTMAEForPreTraining` returns and equals the mean of the per-image `masked_mse` rows. It is **not calibrated** and **not a perceptual judgement**: lower is better, a raw number means nothing without the fills and the frozen model beside it, and the decoder's prediction on the visible patches (`visible_mse`) is reported because the model predicts those too but was never trained on them. `psnr_masked` converts the same error to decibels in 0..1 pixel space for readers who think in PSNR.

## Embeddings and the probe

`embed()` returns the mean of the encoder's 196 patch tokens after its final LayerNorm with **nothing hidden** (mask ratio 0 for that call), L2-normalised, 768-wide. The CLS token is not used: the MAE encoder was never trained to summarise into it. The linear probe standardises those features with the training set's statistics before the head, which is what makes a linear readout of MAE features usable; MAE features are known to probe poorly before fine-tuning, and the probe is a **readout of representation quality**, not a classifier to ship.

## Machine-readable provenance

```python
from vit_mae_pipeline import build_provenance, load_pipeline, write_provenance

pipe = load_pipeline()
record = build_provenance(pipeline=pipe)
write_provenance("outputs/provenance.json", pipeline=pipe)
```

The record includes model ID, immutable revision, weight filename/SHA-256/size, verified checkpoint source and path, the processor contract (224×224, 16×16 patches, ImageNet normalisation), the reconstruction and embedding semantics, the adapter record when one is loaded, Python version, platform, and runtime package versions.

## Input safety

Image inputs may be:

- a local filesystem path;
- `bytes` containing an image;
- a `PIL.Image.Image`.

Remote `http://` and `https://` image strings are rejected intentionally. The pipeline does not act as a network fetcher. Images with a side outside 16..4096 px are rejected before any tensor work; a call takes at most 64 images.

## Supply-chain controls

`load_pipeline()`:

1. resolves offline weights through `weights_path`, `VIT_MAE_WEIGHTS_DIR`, dev repo `weights/vit-mae-base`, or pinned Hugging Face revision fallback;
2. verifies snapshot files against `dimer-base-manifest.json` when present (checking hashes and sizes of configurations and weights);
3. rejects unsafe serialized formats (`.bin`, `.pt`, `.pth`, `.ckpt`, `.pkl`, `.pickle`, `.h5`, `.msgpack`);
4. verifies the exact safetensors byte size and SHA-256 before model load;
5. loads the verified local snapshot with `trust_remote_code=False`, `use_safetensors=True`, and `local_files_only=True`.

## Reproducible reference environment

Python 3.12 is the supported runtime. The repository keeps exact direct pins in `pyproject.toml` and a fully version-pinned Linux/CPU reference graph in `requirements.lock.txt`.

```bash
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps --no-build-isolation -e .
python scripts/check_lock.py
```

`requirements.lock.txt` records the exact dependency versions proven by the real-checkpoint `main` CI path, including the official CPU PyTorch wheel. It is a version lock, not a cryptographic hash lock.

## Tests

```bash
ruff check .
pytest -m "not integration"
```

`tests/test_model_backed.py` (reconstruction on the real checkpoint / probe and k-NN / one-epoch continuation and artifact round trip / loader scope / transactional restore, and the same path on CUDA where visible; the build ran it CPU-only) runs only when the snapshot is staged under `weights/vit-mae-base/`; the photograph cache `weights/inat-birds/` is git-ignored and filled by `fetch_corpus()`.

Real-checkpoint integration:

```bash
RUN_INTEGRATION=1 pytest -m integration -q
python tools/run_notebook.py tutorials/vit_mae_pretraining_colab.ipynb
```

## Scope boundaries

This repository does **not** claim to provide:

- image classification as a product (the probe is a readout, not a classifier to ship);
- object detection, segmentation or image generation;
- inpainting of user-chosen regions (the mask is random, as in pre-training);
- full fine-tuning of the encoder for a downstream task, or pre-training from scratch;
- normalised-pixel targets (`norm_pix_loss` is off in this checkpoint);
- any adaptation beyond the decoder, the encoder's last blocks and its final LayerNorm;
- production HTTP serving or DIMER worker packaging.

Those require separate downstream heads, models, or serving work.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
