---
license: Apache-2.0
model_card_spec: "1.1"
pipeline_tag: image-feature-extraction
task: "Others - Self-Supervised Reconstruction"
tags:
  - masked-autoencoder
  - self-supervised
  - masked-image-modelling
  - reconstruction
  - linear-probe
  - backbone
base_model: facebook/vit-mae-base
date_published: "2022-03-02"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt`, https://huggingface.co/api/models/facebook/vit-mae-base)"
---

# ViT-MAE Base — Masked Autoencoder (Masked-Patch Reconstruction, Linear Probe & Bounded Continuation of Pre-training)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-facebook%2Fvit--mae--base-ffcc4d?style=flat)](https://huggingface.co/facebook/vit-mae-base)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-facebookresearch%2Fmae-181717?style=flat&logo=github&logoColor=white)](https://github.com/facebookresearch/mae)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2111.06377-b31b1b.svg)](https://arxiv.org/abs/2111.06377)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This repository ships one standalone Google Colab tutorial that exercises its public pipeline API end to end — bootstrap a fresh runtime, stage and verify the pinned upstream revision, fetch and validate a digest-pinned photograph set, measure the frozen model's reconstructions against two non-neural fills and its features through a linear probe against three non-neural classifiers, run a bounded continuation of the pre-training objective, evaluate on an image-disjoint split both ways, and export and reload the adapter:

- **E2E Masked-Image-Modelling Tutorial**: \
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/vit-mae-pretraining-pipeline/blob/main/tutorials/vit_mae_pretraining_colab.ipynb) [`vit_mae_pretraining_colab.ipynb`](https://github.com/kurtvalcorza/vit-mae-pretraining-pipeline/blob/main/tutorials/vit_mae_pretraining_colab.ipynb) \
  *Masked-patch reconstruction and mean-pooled embeddings with the pinned `facebook/vit-mae-base` weights, then a bounded continuation of masked autoencoding on the decoder and the last two encoder blocks over 360 CC0 iNaturalist photographs of six bird species: the frozen model's masked-patch MSE beside the mean-patch and blur fills, a linear probe beside the majority floor, a colour nearest neighbour and a k-NN, validation-MSE epoch selection that never returns a worse epoch than the frozen model, held-out evaluation both ways, the drawn shapes re-reconstructed, and a safetensors adapter with its probe head that reloads with verified parity.*

> [!NOTE]
> The notebook runs on CPU and uses CUDA automatically when present. Its clean-runtime execution record and the promotion requirements are in [release verification](docs/release-verification.md).

---

#### Description

ViT-MAE Base is the open-weight ViT-B/16 masked autoencoder of He, Chen, Xie, Li, Dollár and Girshick's *Masked Autoencoders Are Scalable Vision Learners* (CVPR 2022), released by Meta AI (`facebook/vit-mae-base`, pinned revision `25b184bea5538bf5c4c852c79d221195fdd2778d`) and packaged by this repository as a verified DIMER pipeline — the fleet's first self-supervised, task-free row. The checkpoint is the **pre-training** model, `ViTMAEForPreTraining`: a 12-layer, 768-wide encoder (85,798,656 parameters) that sees only the visible 25 % of an image's 196 16×16 patches, and an 8-layer, 512-wide decoder (26,109,184 parameters) that predicts the pixels of the hidden 75 % from the encoder's tokens plus a shared mask token; 111,907,840 parameters in `model.safetensors`, pre-trained on ImageNet-1K for 1,600 epochs with per-patch pixel MSE on the hidden patches only (`norm_pix_loss` off in this checkpoint). It has **no user-facing task output**: the model returns a reconstruction and its own loss, and its encoder is what downstream fine-tuning consumes. This repository reads the checkpoint two ways — `reconstruct` (a seeded random mask, the masked-patch and visible-patch MSE per image, the model's own loss, the composited reconstruction) and `embed` (the mean of the patch tokens with nothing hidden, L2-normalised) with a linear probe on those features — and contributes a hardened supply-chain wrapper (SHA-256 and byte-size verification before loading, exclusion of pickle checkpoints, local-only snapshot execution with `trust_remote_code=False`, rejection of remote HTTP(S) image fetches, full provenance tracking) and a bounded adaptation contract: `evaluate_reconstruction` scores a photograph set with one seeded mask per record beside two non-neural fills, `fit_probe` / `evaluate` read the features through a standardised linear head beside a k-NN, `adapt` continues masked autoencoding on the decoder and the last encoder blocks (40,286,464 of 111,907,840 parameters by default) with validation-MSE epoch selection, and `save_artifact` / `from_artifact` export the trained tensors and the probe head as a digest-manifested safetensors adapter that reloads onto a freshly verified base. The tutorial demonstrates the contract on 360 CC0 iNaturalist photographs of six North American bird species — an in-domain set on which the build record found continued pre-training barely moves a converged checkpoint, which the documentation states rather than hides.

#### Intended Use and Limitations

The sections below outline the primary machine learning tasks, targeted user cohorts, and explicit capability boundaries established for this pipeline.

###### Primary Intended Uses

The primary intended uses of this pipeline comprise six technical capabilities:
1. Masked-patch reconstruction (`ViTMAEPipeline.reconstruct`): hiding a seeded random fraction of an image's patches (75 % by default, 147 of 196) and returning the decoder's reconstruction with the masked-patch MSE, the visible-patch MSE, the mask and the model's own loss.
2. Image feature extraction (`ViTMAEPipeline.embed`): dense, unit-norm 768-dimensional vectors — the mean of the encoder's patch tokens with nothing hidden — for indexing, clustering, probing or as the input of a downstream head.
3. Reconstruction evaluation (`ViTMAEPipeline.evaluate_reconstruction`): scoring a validated image set with one seeded mask per record and reporting mean and median masked MSE, visible MSE and hidden-patch PSNR beside the mean-patch and blur fills on the same masks.
4. Linear-probe evaluation (`ViTMAEPipeline.fit_probe`, `evaluate`, `classify`): a standardised multinomial logistic-regression head on the frozen features of a labelled set, reporting accuracy, macro F1 and a per-class breakdown beside a k-NN vote on the same features.
5. Bounded continuation of pre-training (`ViTMAEPipeline.adapt`): continuing the masked-autoencoding objective on the decoder, the last encoder blocks and the encoder LayerNorm over a photograph set, labels unused, with validation-MSE epoch selection.
6. Adapter export and reload (`save_artifact`, `from_artifact`): exporting the trained tensors and the probe head as safetensors with a manifest and reloading them onto a freshly verified base with verified parity.
Target application domains include representation learning for downstream vision heads, domain-adaptive pre-training on unlabelled imagery from an unfamiliar domain, and feature indexing within the DIMER platform.

###### Primary Intended Users

Primary intended users are computer vision engineers, machine learning researchers and data scientists who understand what a masked autoencoder hides and predicts, that a reconstruction loss is the model's own objective and not a perceptual or quality judgement, that a linear probe is a readout of representation quality rather than a classifier to ship (the paper reports 67.8 % linear-probe top-1 on ImageNet against 83.6 % after full fine-tuning), why a gain — or the absence of one — on one seeded split of one sample is evidence that the contract works rather than a benchmark, why a split must be image-disjoint (and source-disjoint when photographs come from few photographers or sessions), and why the non-neural fills and classifiers are read before any adapted number.

###### Out-of-scope use cases

1. **Capability boundaries:** This model is a pre-training autoencoder. It classifies nothing, detects nothing, segments nothing and generates no new images; its reconstructions are blurry pixel predictions under a random mask, not inpainting of a user-chosen region. It must not be marketed or deployed as a classifier (the probe is a readout), an object detector (use dedicated detection pipelines such as `swin-detection-pipeline`), a semantic segmenter (`swin-segmentation-pipeline`) or an image generator.
2. **Input boundaries:** Accepts local image paths, raw image bytes and PIL Image instances with sides in 16..4096 px, at most 64 per call. Resolution is fixed to 224×224 pixels through the upstream processor; images with drastic aspect ratios or fine details lose them in the resize. Remote URLs (`http://`, `https://`) are strictly rejected at the API boundary.
3. **Adaptation boundaries:** `adapt` trains the decoder, the last `trainable_blocks` encoder blocks and the encoder's final LayerNorm only; the patch embedding, the position embeddings and the earlier blocks stay frozen. It continues the pre-training objective and does not fine-tune for a task; full supervised fine-tuning of the encoder, pre-training from scratch and normalised-pixel targets are not provided. The build record's learning-rate sweep found that on a set the checkpoint already covers (natural photographs) a learning rate of `1e-4` makes the validation masked MSE worse from the first epoch and `1e-5` moves it in the fourth decimal; the selector keeps the frozen model when nothing beats it. Datasets are validated structurally, never semantically.
4. **Decision boundaries:** Autonomous, unreviewed deployment in safety-critical, legal or punitive workflows — automated identification, forensic analysis, medical diagnosis, automated content blocking without human review — is strictly prohibited.

---

#### Factors

This section describes factors influencing model representation and behavior, including demographic categories, capturing instruments, and operational runtime environments.

###### Groups

ViT-MAE was pre-trained on ImageNet-1K (Russakovsky et al., 2015) without labels. ImageNet-1K contains people incidentally and in person-related categories, was collected from web image search, and was not demographically balanced or audited for demographic parity; the model's reconstructions and features therefore reflect ImageNet's distribution of subjects, scenes, regions and cultural artefacts. The pipeline itself does not introduce demographic filters; operators using the features on human imagery bear the direct responsibility of conducting independent fairness audits and bias evaluations on their target-domain datasets before any downstream head is trained on them.

###### Instrumentation

ImageNet-1K photographs originate from diverse consumer and professional cameras, web uploads and scans. Key instrumentation factors that affect data representation include optical resolution, lens distortion, sensor noise, compression artefacts (heavy JPEG quantisation), illumination and dynamic range. Because the processor resizes inputs to 224×224 pixels and the model reasons in 16×16 patches, fine detail below the patch scale is not represented, and a reconstruction loss on an image family the checkpoint has not seen (medical, satellite, microscopy, line art) says only how far that family is from ImageNet's pixel statistics. The pipeline validates image decoding integrity and enforces RGB colour space, but cannot detect underlying camera miscalibration or sensor degradation.

###### Environment

1. **Operating environment:** Designed to run on Python 3.12 with PyTorch 2.14 and `transformers` 4.57.6. Supported hardware includes x86_64 CPUs and NVIDIA GPUs supporting CUDA 12.x. A single inference instance requires approximately 0.45 GB of memory for model weights and minimal RAM for batch activations; continuation with the default two blocks needs the activations and optimiser state of 40 M parameters. Float32 precision is default and fully qualified on CPU; half-precision formats require compatible GPU accelerators.
2. **Data environment:** Assumes photographic or photographic-like RGB images. The masked MSE is low on natural photographs (the domain of pre-training) and rises on images whose pixel statistics diverge from it; that rise is the signal domain-adaptive continuation is for, and also the case in which the reconstructions are least trustworthy as pictures.

---

#### Metrics

This section details performance metrics, decision thresholds, and uncertainty management applied across pipeline operations.

###### Performance Measures

The pipeline reports **masked-patch MSE** — the mean squared error between the decoder's prediction and the true pixels over the patches hidden from the encoder, in the processor's normalised pixel space (ImageNet mean / std), the checkpoint's own training objective and the value `ViTMAEForPreTraining` returns as `loss` — beside the **visible-patch MSE** (the decoder predicts those too, but was never trained on them) and the **PSNR of the hidden patches** in 0..1 pixel space. `evaluate_reconstruction` (`metrics.reconstruction_metrics`) reports the mean and median over a set with one seeded mask per record, and scores two non-neural fills on the same masks: the **mean-patch fill** (every hidden patch is the mean colour of the visible patches) and the **blur fill** (every hidden patch is the mean colour of its visible neighbours). The features are read through a **linear probe** (`fit_probe`: training-set standardisation, the paper's affine-free BatchNorm, then a multinomial logistic-regression head by full-batch Adam) reporting `accuracy`, `macro_f1` and per-class `precision` / `recall` / `f1` (`metrics.classification_metrics`), beside a **k-NN** vote on the same features (`knn_scores`, k = 5), the **majority floor** and the **colour nearest neighbour** (3×3 mean-colour grid). In the literature ViT-MAE Base is quantified by ImageNet-1K top-1 after fine-tuning (83.6 %) and by linear probing (67.8 %); nothing here reproduces those. On the tutorial's 96-photograph test split (CPU build record, seed 42): masked MSE — mean-patch fill 0.7652, blur fill 0.5452, frozen 0.2281 (PSNR 19.3 dB), adapted 0.2280; probe accuracy — majority 0.167, colour neighbour 0.260, k-NN 0.219, frozen probe 0.365 (macro F1 @P:FROZEN_PROBE_F1@), adapted probe 0.365 (macro F1 @P:ADAPTED_PROBE_F1@). These are observations on one seeded split with no dispersion estimate, not a benchmark, and the adapted numbers are the honest reading of continued pre-training on a converged model.

###### Decision thresholds

The pipeline applies **no threshold** to any number it reports. A masked MSE has no natural cut-off — it is compared with the fills and the frozen model, never read alone — and the probe's softmax scores are ranking scores for a readout, not calibrated class probabilities; the API emits raw values. The only decision the pipeline makes is epoch selection inside `adapt`: the epoch with the lowest validation masked MSE, epoch 0 (the frozen model) included, so the selector can and does return the frozen model when nothing beats it. Downstream operators own every other decision.

###### Approaches to uncertainty and variability

Inference is deterministic given a seed: the mask is drawn from a seeded generator (`reconstruct(seed=)`; `evaluate_reconstruction` derives one seed per record id so the same photograph gets the same mask on every evaluation), no dropout is active (`model.eval()`), and the same seed hides the same patches on every run; the masked MSE of one image varies with the mask, which is why the set-level measures are read and the same masks are used before and after adaptation. Variations across runs arise from floating-point kernel differences across hardware or non-deterministic GPU routines. Adaptation is seeded (`seed=0`: shuffling order and training masks) but not bit-reproducible across devices; every corpus metric the tutorial reports is one value on one seeded split (`build_sample_dataset(seed=42)`) of one 360-photograph sample, with a 48-photograph validation split that selects the epoch and a probe whose accuracy moves in steps of one photograph — the build record's learning-rate sweep (four arms at `1e-5`, `3e-5` and `1e-4`) moved the validation masked MSE between 0.2335 and 0.2475, which is the size of the uncertainty a reader should attach to any single number here. Deployments requiring rigorous uncertainty quantification must measure it on their own data.

---

#### Ethical considerations and biases

This section examines data sensitivity, life-critical implications, implemented mitigations, failure risks, and prohibited uses.

###### Data

The model weights were pre-trained by Meta AI on ImageNet-1K without labels. ImageNet's images were collected from web image search and carry their own licences and known concerns (people photographed without consent, offensive category labels in the wider ImageNet, uneven geographic coverage); the pre-training corpus is not distributed here and pre-training exposure to copyrighted or sensitive imagery cannot be ruled out. This repository distributes only open-source Python code, tests and configuration manifests; no datasets or model weight blobs are distributed through git. The tutorial's corpus is 360 research-grade iNaturalist photographs of six common North American birds (American Goldfinch, Chipping Sparrow, Dark-eyed Junco, House Finch, Song Sparrow, White-throated Sparrow; 60 per species, one per observer), each published by its observer under CC0 1.0 and fetched at run time from the iNaturalist open-data bucket by photo id with a byte-size and SHA-256 pin recorded in `samples.py`; nothing is redistributed, every record keeps its observation URL and observer login, and the photographs contain wildlife, not people. Operators supplying images must verify that their input data complies with data privacy laws (e.g., GDPR, HIPAA) and does not contain unauthorised personal identifiable information or classified material — and note that a masked autoencoder trained on private images can reproduce parts of them.

###### Human Life

ViT-MAE is a research pre-training model and is **not** certified, tested or approved for life-critical applications or high-stakes decision-making. It must never be deployed as an autonomous decision-making engine in healthcare diagnostics, patient monitoring, autonomous vehicle navigation, industrial safety trips or criminal justice profiling, and a reconstruction it produces must never be presented as a photograph of what was there. Any secondary deployment in human-adjacent safety workflows demands extensive independent domain verification, redundant fail-safes and continuous human-in-the-loop oversight.

###### Mitigations

This repository enforces concrete, inspectable architectural and supply-chain mitigations:
1. **Cryptographic supply-chain locking:** Pinned to immutable commit `25b184bea5538bf5c4c852c79d221195fdd2778d`, verifying exact safetensors byte size (`447,670,680`) and SHA-256 (`479dcef4bd5df06259399027b789f21e9d9a1b79f37155a64176d55bc26fdae8`) prior to instantiation.
2. **Pickle execution refusal:** Scans the snapshot tree and raises a fatal `RuntimeError` if any `*.bin` weight file is detected.
3. **SSRF protection:** Rejects remote `http://` and `https://` image paths at the API boundary, accepting only validated local filesystem paths, in-memory bytes or PIL images. The public `validate_inputs` helper applies exactly these input checks and returns an input manifest of the schema, ceilings, per-image observations and verdict before the model runs.
4. **Loss fidelity:** The pipeline's masked MSE is computed from the model's own `logits` and `mask` in the same normalised pixel space as training, and the tutorial asserts it equals the model's returned `loss` — the number reported is the objective, not a re-implementation of it.
5. **Deterministic normalisation:** Enforces explicit L2 normalisation on the mean-pooled features, and standardises them with the training set's statistics before the probe head so the readout does not depend on feature scale.
6. **Adaptation integrity:** `adapt` validates the dataset before any tensor is built, trains only the named decoder / encoder-block / LayerNorm tensors with every other parameter's `requires_grad` false, selects the epoch on the validation masked MSE with the frozen model as epoch 0, restores the frozen weights on any exception, discards a probe whose features no longer exist, and records the configuration and epoch history in the artifact; `from_artifact` re-verifies the base snapshot and checks the manifest's format, base identity and weight digest, the file size and SHA-256 and the exact tensor set **before** deserialising, refuses any tensor outside the declared blocks, and overlays onto a freshly loaded base.

###### Risks and harms

Key identified risks include:
1. **Reconstruction misread as recovery:** Operators may present a decoder's prediction of hidden patches as what the hidden region contained; it is a statistical guess from ImageNet pixel priors, blurred by the pixel-MSE objective, and can hallucinate plausible content.
2. **Probe misread as product:** A linear probe's accuracy is a measurement of the features, not a deployable classifier; shipping it as one inherits every failure of a 768-dimensional linear head trained on a few hundred photographs.
3. **Memorisation:** continued pre-training on a small set can memorise it; on private images the reconstructions of masked regions can leak training content, and a narrow continuation can erode the features elsewhere (the tutorial re-reconstructs three drawn shapes as a small look at this, not a measurement).
4. **Dataset bias:** the features reflect ImageNet's distribution of subjects, regions and cultural artefacts; downstream heads trained on them inherit it.
5. **Adaptation risks:** a gain measured on an image-disjoint but observer-overlapping split can overstate transfer; and on an in-domain set the selector's honest answer may be the frozen model, which a reader expecting improvement may misread as failure.

###### Use cases

The following use cases are strictly prohibited by policy and developer intent:
1. Mass biometric surveillance, unauthorised facial identification or social credit tracking in public spaces, including through downstream heads trained on the features.
2. Automated demographic profiling or discriminatory filtering in housing, lending, employment, insurance or public benefits access.
3. Presenting reconstructions of masked or damaged regions as authentic recovered imagery in evidentiary, journalistic or medical contexts.
4. Autonomous lethal systems or automated targeting applications.
5. Any application that violates Meta's upstream Apache-2.0 license terms or applicable national and international privacy regulations.

---

## Technical Specifications and Architecture

### Architecture Overview

ViT-MAE is an asymmetric encoder–decoder over image patches:
- **Patchification:** 224×224 RGB input, 16×16 patches, 196 patches of 768 values each; a learned patch embedding plus fixed sine-cosine position embeddings.
- **Encoder:** ViT-B — 12 Transformer blocks, hidden size 768, 12 heads, MLP 3072 — applied only to the visible patches (25 % by default) plus a CLS token; 85,798,656 parameters.
- **Decoder:** 8 Transformer blocks, hidden size 512, 16 heads — applied to the encoder's tokens projected to 512 plus a shared learned mask token at every hidden position, with its own position embeddings, predicting 768 pixel values per patch; 26,109,184 parameters.
- **Loss Formulation:** mean squared error between predicted and true pixels over the hidden patches only, in the processor's normalised pixel space (`norm_pix_loss = false` in this checkpoint):
  $$\mathcal{L} = \frac{1}{|M|} \sum_{p \in M} \frac{1}{768} \lVert \hat{x}_p - x_p \rVert_2^2$$
  where $M$ is the set of hidden patches, $x_p$ the true pixels of patch $p$ and $\hat{x}_p$ the decoder's prediction.

### Checkpoint Invariants and Loading Controls

The snapshot loader (`vit_mae_pipeline.model.load_components`) enforces strict supply-chain controls:
1. Pinned Hugging Face repository: `facebook/vit-mae-base`
2. Pinned commit revision: `25b184bea5538bf5c4c852c79d221195fdd2778d`
3. Primary weight file: `model.safetensors`
4. Expected weight byte size: `447,670,680` bytes
5. Expected weight SHA-256: `479dcef4bd5df06259399027b789f21e9d9a1b79f37155a64176d55bc26fdae8`
6. Upstream parameters: `111,907,840` F32 parameters (encoder 85,798,656; decoder 26,109,184)
7. Execution policy: `trust_remote_code=False`, `use_safetensors=True`, `local_files_only=True`
8. Adapter artifact format: `org.valcorza.vit-mae-base.adapter.v1` — `adapter.safetensors` (the trained tensors plus the probe head and its standardisation statistics when a probe is fitted; 174 tensors, 161,190,912 bytes for the default two blocks with the probe) plus `manifest.json` naming the base id, revision and `model.safetensors` digest, the objective, mask ratio and trainable blocks, the probe record, the tensor names, the file size and SHA-256, the training configuration and the epoch history
9. Tutorial corpus: 360 iNaturalist photographs (CC0 1.0; six species, 60 each, one per observer), `CORPUS_BYTES = 39,223,447`, each pinned by photo id, byte size and SHA-256 in `vit_mae_pipeline/samples.py` and fetched from `https://inaturalist-open-data.s3.amazonaws.com/photos/<id>/medium.<ext>`; split 216 / 48 / 96 by `build_sample_dataset(seed=42)`
10. Build record (CPU, 2026-09-21): the default tutorial path run through the package API on the build workstation's CPU (`torch 2.14.0`, `transformers 4.57.6`, Python 3.12, `CUDA_VISIBLE_DEVICES=-1`, snapshot and photographs pre-staged) — split 216 / 48 / 96 (seed 42), the frozen model reconstructed the 96 test photographs in 4.4 s and was probed in 28.8 s, 5 epochs of the default recipe (lr `1e-5`, 2 blocks, batch 8) in 159.1 s (validation masked MSE 0.2335 → 0.2338 → 0.2338 → 0.2332 → 0.2334 → 0.2337, epoch 3 kept), the adapter 174 tensors / 161,190,912 bytes with reload parity 0 (max abs masked-MSE difference over eight test photographs) and 8 of 8 identical probe decisions; comparison {reconstruction: {masked_mse: {mean_patch_fill: 0.7652, blur_fill: 0.5452, frozen: 0.2281, adapted: 0.228}, masked_mse_median: {mean_patch_fill: 0.6763, blur_fill: 0.4753, frozen: 0.1801, adapted: 0.1827}, masked_mse_visible: {mean_patch_fill: 0.0, blur_fill: 0.0, frozen: 0.2794, adapted: 0.2789}, psnr_masked: {mean_patch_fill: 14.0797, blur_fill: 15.5516, frozen: 19.3365, adapted: 19.3375}}, reconstruction_delta_vs_frozen: {masked_mse: -5e-05, masked_mse_median: 0.00261, masked_mse_visible: -0.00054, psnr_masked: 0.00104}, validation_masked_mse: {frozen: 0.2335, selected_epoch: 3, selected: 0.2332}, probe: {accuracy: {majority: 0.167, colour_neighbour: 0.26, knn_frozen: 0.219, knn_adapted: 0.219, frozen: 0.365, adapted: 0.365}, macro_f1: {majority: 0.048, colour_neighbour: 0.261, knn_frozen: 0.216, knn_adapted: 0.216, frozen: 0.365, adapted: 0.365}}, probe_delta_vs_frozen: {accuracy: 0.0, macro_f1: 0.0}, by_species: {american_goldfinch: {n: 16, frozen_recall: 0.69, adapted_recall: 0.69, frozen_f1: 0.56, adapted_f1: 0.56}, chipping_sparrow: {n: 16, frozen_recall: 0.25, adapted_recall: 0.25, frozen_f1: 0.22, adapted_f1: 0.22}, dark_eyed_junco: {n: 16, frozen_recall: 0.38, adapted_recall: 0.38, frozen_f1: 0.36, adapted_f1: 0.36}, house_finch: {n: 16, frozen_recall: 0.31, adapted_recall: 0.31, frozen_f1: 0.45, adapted_f1: 0.45}, song_sparrow: {n: 16, frozen_recall: 0.31, adapted_recall: 0.31, frozen_f1: 0.33, adapted_f1: 0.33}, white_throated_sparrow: {n: 16, frozen_recall: 0.25, adapted_recall: 0.25, frozen_f1: 0.26, adapted_f1: 0.26}}}. Recorded in `docs/release-verification.md` as a pre-flight. Executed 2026-09-21: the **committed notebook blob** (`eb708a9` / `06bc11b5`) run top-to-bottom on a clean Kaggle Tesla T4 kernel (`kurtvalcorza/dimer-nb2-vit-mae-pretraining` v1, `torch 2.14.0+cu130`, `transformers 4.57.6`, Python 3.12.13, `cuda`, empty Hugging Face cache, no repository checkout, blob SHA-1 verified against GitHub before execution): 14/14 ok (1 restart after install cell), 366.8 s, 370 files, 487 MB fetched from the Hub and digest-verified inside the notebook; five epochs in 25.1 s (validation masked MSE 0.2335 → 0.2338 → 0.2338 → 0.2332 → 0.2334 → 0.2337, epoch 3 kept); comparison {reconstruction: {masked_mse: {mean_patch_fill: 0.7652, blur_fill: 0.5452, frozen: 0.2281, adapted: 0.228}, masked_mse_median: {mean_patch_fill: 0.6763, blur_fill: 0.4753, frozen: 0.1801, adapted: 0.1827}, masked_mse_visible: {mean_patch_fill: 0.0, blur_fill: 0.0, frozen: 0.2794, adapted: 0.2789}, psnr_masked: {mean_patch_fill: 14.0797, blur_fill: 15.5516, frozen: 19.3365, adapted: 19.3375}}, reconstruction_delta_vs_frozen: {masked_mse: -5e-05, masked_mse_median: 0.00261, masked_mse_visible: -0.00054, psnr_masked: 0.00104}, validation_masked_mse: {frozen: 0.2335, selected_epoch: 3, selected: 0.2332, rescored: 0.2332}, probe: {accuracy: {majority: 0.167, colour_neighbour: 0.26, knn_frozen: 0.219, knn_adapted: 0.219, frozen: 0.365, adapted: 0.365}, macro_f1: {majority: 0.048, colour_neighbour: 0.261, knn_frozen: 0.216, knn_adapted: 0.216, frozen: 0.365, adapted: 0.365}}, probe_delta_vs_frozen: {accuracy: 0.0, macro_f1: 0.0}, by_species: {american_goldfinch: {n: 16, frozen_recall: 0.69, adapted_recall: 0.69, frozen_f1: 0.56, adapted_f1: 0.56}, chipping_sparrow: {n: 16, frozen_recall: 0.25, adapted_recall: 0.25, frozen_f1: 0.22, adapted_f1: 0.22}, dark_eyed_junco: {n: 16, frozen_recall: 0.38, adapted_recall: 0.38, frozen_f1: 0.36, adapted_f1: 0.36}, house_finch: {n: 16, frozen_recall: 0.31, adapted_recall: 0.31, frozen_f1: 0.45, adapted_f1: 0.45}, song_sparrow: {n: 16, frozen_recall: 0.31, adapted_recall: 0.31, frozen_f1: 0.33, adapted_f1: 0.33}, white_throated_sparrow: {n: 16, frozen_recall: 0.25, adapted_recall: 0.25, frozen_f1: 0.26, adapted_f1: 0.26}}}; reload parity {max_abs_masked_mse_difference: 0.0, identical_reconstructions: 8, identical_probe_decisions: 8, max_abs_probe_score_difference: 1.1920928955078125e-06, of: 8}. Not executed: any corpus other than the one 360-photograph iNaturalist sample, repeated seeds or splits (no dispersion), BYOD, and the adapted model on any photographs but that test split.

### Public Inference API

```python
from vit_mae_pipeline import load_pipeline

pipe = load_pipeline(device="cpu")

# 1. Masked-patch reconstruction (a seeded random 75 % of the patches hidden)
result = pipe.reconstruct("scene.jpg", mask_ratio=0.75, seed=0)
entry = result["results"][0]
entry["masked_mse"], entry["visible_mse"], entry["hidden_patches"], entry["mask"]
entry["reconstruction"].save("scene_reconstructed.png")
result["model_loss"]                       # the checkpoint's own loss == mean masked_mse

# 2. Dense feature embeddings (mean of the patch tokens, nothing hidden, L2-normalised)
features = pipe.embed(["scene1.jpg", "scene2.jpg"])["embeddings"]   # (2, 768)
```

### Public Adaptation API

```python
from vit_mae_pipeline import ViTMAEPipeline, build_sample_dataset, fetch_corpus, read_corpus

splits = build_sample_dataset(read_corpus(fetch_corpus()), seed=42)  # 216 / 48 / 96 photographs
pipe = ViTMAEPipeline.from_pretrained(weights_dir="weights/vit-mae-base")

frozen = pipe.evaluate_reconstruction(splits["test"], seed=0)   # masked_mse, psnr_masked, per_image, baselines
pipe.fit_probe(splits["train"]); probe = pipe.evaluate(splits["test"])   # accuracy, macro_f1, per_class, knn
result = pipe.adapt(splits["train"], splits["validation"],
                    epochs=5, lr=1e-5, batch_size=8, trainable_blocks=2)   # labels unused
adapted = pipe.evaluate_reconstruction(splits["test"], seed=0)
pipe.fit_probe(splits["train"]); pipe.evaluate(splits["test"])
pipe.save_artifact("outputs/adapter")                          # adapter.safetensors (+ probe head) + manifest.json
again = ViTMAEPipeline.from_artifact("outputs/adapter", weights_dir="weights/vit-mae-base")
```

Dataset contract (`samples.py`): records `{id, image, label}` (`id` matching `[A-Za-z0-9_.:-]{1,64}` and unique; a PIL image or a decodable file with sides in `MIN_IMAGE_SIDE = 16` .. `MAX_IMAGE_SIDE = 4096`; a label of at most 64 plain characters, required by the probe and optional for the reconstruction contract); `validate_dataset(records, *, min_records=8, max_records=20000, require_labels=True)` (2..100 labels); `split_dataset(records, *, val_fraction=0.15, test_fraction=0.2, seed=0)` (stratified, pixel-digest de-duplicated); `check_split_disjoint(splits)`; `observer_overlap(splits)`; `load_byod_dataset(path)` (directory or zip with `labels.csv`: `id`, `file`, `label`); `write_dataset_csv(records, path)`; `fetch_corpus(cache_dir=None)`, `read_corpus(files)`, `build_sample_dataset(records, *, seed=42, sizes=SAMPLE_SPLIT)`. Metrics (`metrics.py`): `masked_mse`, `psnr_from_mse`, `reconstruction_metrics(rows, *, mask_ratio)`, `mean_patch_fill`, `blur_fill`, `classification_metrics(scores, gold, classes)`, `knn_scores`, `majority_baseline(train, records, classes)`, `colour_signature(image)`, `colour_neighbour_baseline(train, records, classes)`.

### Upstream References and Citations

- **MAE Paper:** He, Chen, Xie, Li, Dollár and Girshick, *"Masked Autoencoders Are Scalable Vision Learners"*, CVPR 2022, arXiv:2111.06377.
- **Backbone:** Dosovitskiy et al., *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"*, ICLR 2021, arXiv:2010.11929.
- **Pre-training data:** Russakovsky et al., *"ImageNet Large Scale Visual Recognition Challenge"*, IJCV 2015 (ImageNet-1K, used without labels).
- **Upstream Repository:** https://github.com/facebookresearch/mae — served through Hugging Face Transformers (`ViTMAEForPreTraining`).
- **Sibling row in this fleet:** `siglip-v1-zero-shot-pipeline`, sharing this repository's tutorial corpus and code shape.
- **Tutorial corpus:** iNaturalist open data (CC0 photographs under each observer's own licence), https://www.inaturalist.org/pages/developers — bucket https://inaturalist-open-data.s3.amazonaws.com/
