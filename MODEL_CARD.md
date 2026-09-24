---
license: mit
model_card_spec: "1.2"
pipeline_tag: zero-shot-object-detection
task: "Others - Open-World Object Counting"
tags:
  - counting
  - open-world
  - grounding-dino
  - exemplars
  - text-prompted
  - object-counting
base_model: nikigoli/countgd
date_published: "2024-07-05"
date_published_source: "first commit of the authors' Hugging Face Space nikigoli/countgd holding checkpoint_best_regular.pth with the pinned LFS object id (commit a277bb81, 2024-07-05T19:39:02Z, https://huggingface.co/spaces/nikigoli/countgd/commits/main)"
---

# CountGD — Multi-Modal Open-World Counting (Text & Exemplar Prompts, Box-Scored Evaluation & Bounded Counting Fine-Tune)

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-nikigoli%2Fcountgd-ffcc4d?style=flat)](https://huggingface.co/spaces/nikigoli/countgd)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-niki--amini--naieni%2FCountGD-181717?style=flat&logo=github&logoColor=white)](https://github.com/niki-amini-naieni/CountGD)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2407.04619-b31b1b.svg)](https://arxiv.org/abs/2407.04619)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/niki-amini-naieni/CountGD/blob/main/LICENSE)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This repository ships one standalone Google Colab tutorial that exercises its public pipeline API end to end — bootstrap a fresh runtime, stage and verify the pinned upstream checkpoint, audit and convert it, count one scene three ways, score the frozen model against two non-neural baselines on held-out scenes and photographs, run a bounded counting fine-tune, evaluate again, and export and reload the adapter:

- **E2E Open-World Counting Tutorial**: \
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/countgd-object-counting-pipeline/blob/main/tutorials/countgd_object_counting_colab.ipynb) [`countgd_object_counting_colab.ipynb`](https://github.com/kurtvalcorza/countgd-object-counting-pipeline/blob/main/tutorials/countgd_object_counting_colab.ipynb) \
  *Counting with the pinned CountGD checkpoint from a text prompt, three exemplar boxes, or both; count error, point localisation and box IoU on synthetic scenes with known boxes and on 24 FSC-147 test photographs, beside a mean-count baseline and a template matcher; a fine-tune of the last two decoder layers and the shared box head with validation-MAE epoch selection; and a safetensors adapter that reloads with verified parity.*

> [!NOTE]
> The notebook runs on CPU and uses CUDA automatically when present. Its execution records and the promotion requirements are in [release verification](docs/release-verification.md).

---

#### Description

This repository packages **CountGD** (Amini-Naieni, Han and Zisserman, *CountGD: Multi-Modal Open-World Counting*, NeurIPS 2024), the paper checkpoint the authors distribute as `checkpoint_best_regular.pth` in their Hugging Face Space `nikigoli/countgd`, pinned at revision `6e82e59569a84ee5c6aafa35d396f2d2bee57be2`.

CountGD is an open-set object detector used as a counter. It is GroundingDINO: a Swin-B image backbone, a BERT text encoder, a feature enhancer that fuses image and text features, and a six-layer transformer decoder with 900 object queries. CountGD extends it so that each exemplar box the user draws is pooled from the image features and entered as an extra prompt token beside the words of the text. The network has 233,362,816 parameters.

At inference time every query predicts one box and a similarity score to each prompt token. The queries whose best score exceeds a threshold — 0.23, the authors' value — are the counted objects. The output is a count, one box and one point (the box centre) per counted object, and the scores. The prompt is a text naming the category, up to three exemplar boxes, or both. No training happens at inference; adaptation happens only through the bounded fine-tune this repository ships.

This repository adds the following to the upstream weights:

- **pinned and verified weights:** the source checkpoint is pinned by revision, byte size and SHA-256; its pickle is audited statically, opened once with torch's restricted unpickler, and converted into `countgd.safetensors`, whose digest is pinned and is the only file the loader reads;
- **vendored model code:** the upstream network carried as PyTorch source with the multi-scale deformable attention in pure PyTorch, so no compiled CUDA extension is needed;
- **an inference contract:** `CountGDPipeline.count` and `validate_inputs`, which validate the image, the prompt and the exemplar boxes before any tensor work;
- **an evaluation path:** `CountGDPipeline.evaluate`, reporting count error, point localisation and box IoU, and two non-neural baselines scored by the same code;
- **a fine-tuning entry point:** `CountGDPipeline.adapt`, a bounded fine-tune on upstream's objective with validation-based epoch selection, and a safetensors adapter format that `from_artifact` verifies before loading;
- **sample data code:** seeded synthetic counting scenes with known object boxes, and a digest-pinned set of 80 FSC-147 test photographs fetched at run time.

#### Intended Use and Limitations

The pipeline is meant for counting instances of a category a user can name or point at in still images, and for studying how a small fine-tune changes that count.

###### Primary Intended Uses

The task is prompt-conditioned object counting. The input is one RGB image and a prompt: a category name of at most 64 plain characters, up to three boxes around single instances, or both. The output is an integer count, one box `[x0, y0, x1, y1]` and one point per counted object in the input image's pixels, and one uncalibrated score per counted object.

Envisioned application domains are counting tasks where the objects are separable instances at moderate density: inventory and shelf items, produce and seeds on a tray, vehicles or animals in a still frame, colonies or cells in a microscopy field, and teaching and research on open-world counting. The authors trained and evaluated the checkpoint on FSC-147, a benchmark of 147 everyday object categories with 7 to 3,701 objects per image.

The pipeline is designed as a component that a reader's own application embeds: a zero-shot counter for categories it was not trained on, a baseline to beat before training a category-specific counter, and a starting point for a bounded fine-tune on a reader's own annotated images.

###### Primary Intended Users

The intended users are machine learning engineers, computer-vision researchers, data scientists and application developers who need counts of named objects in images.

The envisioned deployment settings are research, teaching, and a self-hosted application or batch job that runs this repository's code on the reader's own hardware or a hosted notebook runtime.

The pipeline assumes that its users understand:

- that the count is the number of scores above a threshold, the scores are not calibrated, and the threshold may need to change for their images;
- how to read MAE, RMSE, point matching and box IoU, and why a count can be right while the boxes are wrong, or the reverse;
- that exemplar boxes describe what an object looks like, so exemplars alone can count look-alike objects of another category;
- that a fine-tune can improve one image distribution and degrade another, and that a held-out set from the original distribution is needed to see it.

###### Out-of-scope use cases

- **Capability boundaries:**
  - not an object detector or segmenter to ship: the boxes are a reading of what was counted and are not evaluated as detections;
  - no density-map counting, video counting or tracking;
  - no segmentation masks; the upstream SAM-based test-time normalisation is not carried;
  - no calibrated confidence for a count.
- **Input boundaries:**
  - not for images with more objects than one pass of 900 queries at a shortest side of 800 px can separate: the upstream test-time cropping for dense images is not carried, and counts saturate near the query budget;
  - images must have sides between 32 and 4096 px; smaller or larger images are rejected;
  - prompts longer than 64 characters, more than three exemplar boxes, exemplar boxes outside the image or under 2 px on a side, and remote URLs as image inputs are rejected;
  - at most 16 images per call and 2,000 gold objects per training or evaluation record;
  - behaviour on categories that look alike in shape and texture (for example exemplars alone among similar distractors) degrades; the tutorial demonstrates it.
- **Decision boundaries:**
  - not for autonomous decisions with safety, medical, legal or financial consequences, such as dosing from cell counts, crowd-safety limits or billing, without human review of the counts and boxes;
  - not for counting people to identify, track or profile them (see Use cases).

---

#### Factors

The model's behaviour varies with the category, the object density and size, the prompt mode, and the image conditions.

###### Groups

The pipeline is not human-centric: it counts objects of a category the user names or draws, and nothing in the repository's code, sample data or evaluation measures behaviour for groups of people.

The upstream training data, FSC-147, is web images of 147 everyday object categories collected by its authors (6,146 annotated images in the pinned mirror's annotation file, 3,659 of them in the training split). It was not audited by them or by this repository for the demographic or geographic composition of the scenes. The BERT text encoder was pre-trained on English text, and the image backbone on ImageNet-22k; neither was group-audited here.

If a deployment counts people, or objects whose appearance correlates with a group of people, the operator must audit counting error and box quality per group on their own data before relying on the counts. The groups the evaluation in this repository does distinguish are object categories: the eight FSC-147 test categories of the sample, and the target and distractor classes of the synthetic scenes, reported per class by `evaluate`.

###### Instrumentation

The FSC-147 images were collected from the web by the benchmark's authors and resized to a height of 384 px; the cameras, lenses and processing behind them are not recorded. The 80 photographs this repository fetches are that 384-px version, from the Hub mirror `isentropic/FSC147` at revision `3e420cb6537e803dd6d4516623ce82a79c0317b8`. The synthetic scenes are drawn in code: flat-colour shapes on a flat background with clutter specks, 512 × 384 px, lossless.

Every image is resized by the pipeline to a shortest side of 800 px (longest at most 1333 px) and normalised with ImageNet statistics, as the authors' test transform does. Instrument characteristics that change apparent object size, sharpness or colour — resolution, lens blur, compression, white balance — change what the model counts, and the pipeline has no check that detects them. JPEG artefacts and motion blur can merge or split small objects; a change in camera or magnification between exemplars and the rest of the image weakens the exemplar prompt.

###### Environment

**Operating environment.** The pipeline runs in float32 on CPU or on a CUDA GPU and picks CUDA when it is visible. The deformable attention runs in pure PyTorch on both, so no compiled extension is needed and no feature is unavailable on CPU. On the build workstation's CPU, counting took about 4.6 s per image and a fine-tuning step about 4.7 s (measured in the notebook run recorded in `docs/release-verification.md`). The converted weights are 938 MB; GPU memory use was not measured.

**Data environment.** The model counts categories it has not seen in training, provided the objects are distinct instances and the prompt identifies them. Its behaviour holds best for photographs like FSC-147's: everyday objects, moderate density, objects a few pixels to a few tens of pixels across after resizing. It degrades when objects are very small or overlap heavily, when the scene holds more instances than the query budget separates, when a distractor category shares the target's shape and texture, and when the image distribution is far from web photographs (for example microscopy or aerial imagery) without a fine-tune.

---

#### Metrics

The measures are chosen for a counter whose output is a number and a set of boxes: one measure for the number, one for which objects were counted, and one for the boxes.

###### Performance Measures

`CountGDPipeline.evaluate` reports, named as the code names them:

- `mae`, `rmse`, `nae`, `under_count_fraction`, `exact_fraction` — the count error. `mae` and `rmse` are FSC-147's own metrics: `mae` is the mean absolute count error; `rmse` weights the largest errors, which a few dense images dominate; `nae` divides each error by the gold count so that images of different density compare.
- `localisation.precision`, `localisation.recall`, `localisation.f1` — point localisation: predicted points (box centres) matched one-to-one to gold points by Hungarian matching within half the mean exemplar side (at least 4 px). This is what separates a correct count of the wrong objects from a correct count of the right ones.
- `boxes.precision`, `boxes.recall`, `boxes.f1`, `boxes.mean_matched_iou`, `boxes.mean_best_iou` — box extent, where gold object boxes exist: predicted boxes matched one-to-one to gold boxes by Hungarian matching on IoU, a true positive at IoU ≥ 0.5.

A count alone hides compensating errors: a model that misses five objects and counts five distractors scores MAE 0. Points alone do not say whether the boxes enclose the objects. Reading all three is why the tutorial uses synthetic scenes with gold boxes beside FSC-147, which has points only.

Recorded values, observed in the notebook run on the build workstation's CPU (one seeded draw; see Approaches to uncertainty):

| Set | System | `mae` | `rmse` | `localisation.f1` | `boxes.f1` |
|---|---|---|---|---|---|
| 12 synthetic test scenes | mean-count baseline | 10.00 | 11.11 | — | — |
| 12 synthetic test scenes | template matcher | 14.50 | 17.84 | 0.704 | — |
| 12 synthetic test scenes | frozen CountGD | 7.42 | 9.97 | 0.862 | 0.453 |
| 12 synthetic test scenes | fine-tuned CountGD | 0.92 | 3.18 | 0.981 | 0.550 |
| 24 FSC-147 test photographs | mean-count baseline | 16.21 | 21.47 | — | — |
| 24 FSC-147 test photographs | template matcher | 34.00 | 41.59 | 0.353 | — |
| 24 FSC-147 test photographs | frozen CountGD | 4.54 | 8.69 | 0.922 | — |
| 24 FSC-147 test photographs | fine-tuned CountGD | 3.50 | 8.34 | 0.925 | — |

The table is the CPU run. In the clean Kaggle T4 run of the same notebook the frozen and baseline values were the same except the FSC-147 frozen RMSE (8.72); the fine-tune kept epoch 2 instead of 3 and reached synthetic MAE 1.58, box F1 0.546 and FSC-147 MAE 3.54 (see Verification records). The authors report FSC-147 test MAE and RMSE for the full test split in their paper; this repository does not reproduce that evaluation and makes no claim about it.

###### Decision thresholds

The default decision rule is a threshold: a query counts when its best token score exceeds `CONFIDENCE_THRESHOLD = 0.23`, the authors' value (`--confidence_thresh` in their inference scripts). It is not tuned by this repository and not calibrated for any image distribution. `count` and `evaluate` take a `threshold` argument in (0, 1).

Two further thresholds are fixed in the evaluation and not in the output: a predicted box matches a gold box at IoU ≥ 0.5, and a predicted point matches a gold point within half the mean exemplar side (at least 4 px). No acceptance threshold on any metric was set during development; the tutorial asserts only that the fine-tune's kept epoch is not worse on the validation MAE than the frozen model.

A deployment owns its threshold. Raising it trades missed objects (under-counting) for fewer false counts; lowering it does the reverse. Where an over-count is costly — for example counting defects that trigger rejection — raise it and measure the recall lost; where a missed object is costly, lower it and measure the precision lost, on labelled images from the deployment.

###### Approaches to uncertainty and variability

Every value in Performance Measures comes from a single execution on one seeded draw: 12 synthetic test scenes (seeds 3000–3011), and 24 FSC-147 photographs drawn with seed 42 from the 80 pinned ones. No repetition, cross-validation or bootstrap was run, and no standard deviation or confidence interval is reported. A count error moves in steps of one object per image, so on 12 scenes a difference of 0.1 in MAE is about one object.

Sources of variability: the synthetic scenes, the split, the fine-tune's example order and the model's initialisation of trainable tensors are all seeded (`synthetic_scene(seed)`, `SPLIT_SEED = 42`, `adapt(seed=0)`). CUDA and CPU kernels differ in floating-point results, so counts near the threshold can differ between devices by an object; the recorded runs state their device.

The scores are sigmoid similarities, not calibrated probabilities, and the count carries no confidence of its own. A caller who needs a calibrated count must measure the count error against labelled images of their own distribution and report its dispersion over enough images.

---

#### Ethical considerations and biases

No external review board or group-specific testing reviewed this pipeline.

###### Data

Pretraining and training data, as the upstream authors disclose it: the Swin-B backbone was pre-trained on ImageNet-22k, BERT on BooksCorpus and English Wikipedia, GroundingDINO on the object-detection and grounding corpora its authors list, and CountGD was fine-tuned on FSC-147's 3,659 training images. The authors disclose these corpora by name; their full contents were not audited here, and whether they contain personal or sensitive material is not known — web-collected images can contain people and private scenes.

This repository distributes code, the vendored model code, manifests, the tokenizer's vocabulary and configuration, and FSC-147 point and box annotations for 80 test images. It does not distribute the checkpoint (it is downloaded from the authors' Space), the converted weights, or any image: the FSC-147 photographs are downloaded at run time, and the synthetic scenes are drawn in code.

The operator is responsible for the images they count or fine-tune on: whether they contain people, faces, licence plates, private premises, or proprietary products, and whether processing them is permitted. The pipeline performs no such check.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit or housing. It has been validated for none of them: the only evidence is the recorded runs in this repository, on synthetic scenes and 24 everyday-object photographs.

Use in a sensitive domain is foreseeable — cell counting in pathology, crowd counting for safety, livestock or pest counts that trigger interventions. It would be admissible only with human review of every count that feeds a decision, validation of count error on the domain's own labelled images by people qualified in that domain, and any regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** the source checkpoint is pinned to Space revision `6e82e59569a84ee5c6aafa35d396f2d2bee57be2`, 1,250,122,522 bytes and SHA-256 `c1bab864b17db345b4c6e3aaabb5765bc2c0a90d0bc8defb5e664a74a50aa126`; `verify_snapshot` re-hashes the manifest entries and refuses a digest that differs from the package constants. `audit_pickle` lists the pickle's globals statically and refuses any outside five allowed names; `convert_checkpoint` refuses an audit digest other than `4606eaf365d27d2fddd901bdc069218dabbd418f9856ed2d0e3707612ad4c527` and loads with `weights_only=True`. The converted file must be 937,560,480 bytes with SHA-256 `8e44867b951e3a4205d918e022b78bc5fea218fd17c1851b864a01c421d2d443`, or loading raises; a converted file that fails is refused, not regenerated. Any other pickle-format file in the weights directory is refused. The model is loaded with `strict=True`.
- **Input integrity:** `validate_inputs` and `count` reject remote URLs, images with a side outside 32..4096 px, more than 16 images per call, prompts that are not plain characters or exceed 64, more than three exemplar boxes, and boxes outside the image or under 2 px. `validate_dataset` rejects duplicate ids, counts that disagree with the points or boxes, and annotations outside their image, before the model runs.
- **Statistical mitigations:** the fine-tune keeps the epoch with the lowest validation MAE, including the frozen model when no epoch beats it; a failure during training restores the base tensors. The tutorial scores a held-out set from the original distribution (FSC-147) beside the fine-tuning distribution to expose regressions.
- **Reproducibility:** seeds control the synthetic scenes, the splits and the fine-tune order; dependencies are pinned exactly in `pyproject.toml` and version-locked in `requirements.lock.txt`; `build_provenance` records the weight digests, the source and audit digests, the runtime versions and the adapter; the tutorial notebook carries the package, the manifests and the pins, and a parity check fails when they drift.
- **Refusals:** no `trust_remote_code`, no Hub-hosted code, no pickle on the load path; `load_artifact` refuses an adapter whose manifest names another base model, revision or weight digest, whose file digest differs, or whose tensor set differs from its recorded configuration, before deserialising it.

###### Risks and harms

- **Silent over- or under-counting out of distribution.** The model returns a count and boxes with no sign that the image is unlike its training data. Dense scenes, tiny objects and unfamiliar imaging (microscopy, aerial) produce wrong counts with plausible-looking scores. The operator and anyone acting on the count bear the harm; the likelihood is high whenever such images are counted without a labelled check; the magnitude depends on what the count decides.
- **Counting the wrong category.** Exemplar boxes alone describe appearance, not category, and count look-alike distractors; the tutorial's demo scene counts 51 shapes for 35 circles with exemplars alone. A text prompt reduces this but does not remove it for categories that look alike. The operator bears the harm; it is likely whenever distractors resemble the target.
- **Automation bias.** A precise-looking integer invites trust. Users may accept counts without reviewing the boxes, especially at scale. Third parties affected by a count-based decision bear the harm.
- **Regression after fine-tuning.** A fine-tune that improves one distribution can degrade another; the tutorial measures this on FSC-147, but a user who fine-tunes without a held-out set from their original distribution will not see it.
- **Threshold misuse.** Treating the 0.23 threshold as calibrated, or the scores as probabilities, misstates confidence. Moving the threshold changes the count systematically.
- **Data leakage in evaluation.** Near-duplicate images across training and test splits inflate the measured improvement; the split is de-duplicated by decoded pixels only, so re-encoded duplicates are not caught.
- **Bias amplification.** Categories and scene types under-represented in FSC-147 and the pretraining corpora are likely to be counted worse; this was not measured.

###### Use cases

The developers consider the following uses unacceptable even where the model would work:

- counting, locating or tracking people for surveillance, crowd monitoring that targets individuals, or profiling by appearance;
- counts used for unlawful discrimination in employment, housing, credit, insurance, education or healthcare access;
- deceptive uses, such as fabricating inventory, attendance or yield figures, or presenting an unverified count as an audited one;
- any use prohibited by the MIT licence terms of the weights and code, by the licences of the pretraining corpora, or by the terms of the images a user supplies.

---

## Technical Specifications and Architecture

### Architecture Overview

- **Image backbone:** Swin-B (patch size 4, window 12, ImageNet-22k pre-training at 384 px), features at three scales plus one extra level, projected to 256 channels.
- **Text encoder:** BERT-base (uncased), up to 256 tokens; the caption is `<label> .`, and each exemplar is inserted as an extra token.
- **Feature enhancer and encoder:** six layers of multi-scale deformable self-attention over the image features with text–image cross-attention (bi-directional fusion).
- **Decoder:** six layers, 900 queries, deformable cross-attention to the image and cross-attention to the text; one box head shared by the six layers and the encoder output (`dec_pred_bbox_embed_share=True`).
- **Exemplar tokens:** each exemplar box is pooled from the image features by RoI alignment and added to the prompt tokens.
- **Output reading:** a query counts when the maximum over prompt tokens of its sigmoid similarity exceeds the threshold; its box is returned in input pixels and its centre as the point.

### Checkpoint Invariants and Loading Controls

The loader (`countgd_pipeline.model.load_components`, called by `CountGDPipeline.from_pretrained`) enforces:

1. Pinned source: the Hugging Face Space `nikigoli/countgd` at revision `6e82e59569a84ee5c6aafa35d396f2d2bee57be2`, manifest `weights/countgd/dimer-base-manifest.json` (the Space `README.md` and `checkpoint_best_regular.pth`).
2. Source checkpoint: `checkpoint_best_regular.pth`, 1,250,122,522 bytes, SHA-256 `c1bab864b17db345b4c6e3aaabb5765bc2c0a90d0bc8defb5e664a74a50aa126`; byte-identical to the Google Drive file linked by the upstream README at commit `b6f362b3f5cd20db4a171faa410dfed8f2f466d8`.
3. Pickle audit: globals `argparse.Namespace`, `collections.OrderedDict`, `torch.FloatStorage`, `torch.LongStorage`, `torch._utils._rebuild_tensor_v2`; audit digest `4606eaf365d27d2fddd901bdc069218dabbd418f9856ed2d0e3707612ad4c527`; restricted load with `weights_only=True` and only `argparse.Namespace` added.
4. Served file: `countgd.safetensors`, 937,560,480 bytes, SHA-256 `8e44867b951e3a4205d918e022b78bc5fea218fd17c1851b864a01c421d2d443`; 1,042 stored tensors and 66 tied-head aliases in the one metadata entry `countgd_aliases`; 38 unused `feature_map_encoder.*` tensors of the source dropped. Its tensor data are byte-identical to the authors' own safetensors export on the Hub (`nikigoli/CountGD`).
5. Parameters: 233,362,816 (float32); loaded with `strict=True`.
6. Tokenizer: `google-bert/bert-base-uncased` at revision `86b5e0934494bd15c9632b12f734a8a67f723594` (vocabulary and configuration only; BERT's weights come from the checkpoint).
7. Execution policy: no `trust_remote_code`, no Hub-hosted code, no pickle on the load path; the multi-scale deformable attention runs in pure PyTorch.
8. Adapter artifact format: `org.valcorza.countgd.adapter.v1` — `adapter.safetensors` (the trained tensors: 3,619,584 parameters for the default two decoder layers, the decoder norm and the shared box head) plus `manifest.json` naming the base id, revision and `countgd.safetensors` digest, the objective, the tensor names, the file's size and SHA-256, the configuration and the history.
9. Tutorial data: 80 FSC-147 test photographs (2,732,222 bytes) from `isentropic/FSC147` at revision `3e420cb6537e803dd6d4516623ce82a79c0317b8`, each pinned by byte size and SHA-256 in `countgd_pipeline/samples.py`; synthetic scenes from `countgd_pipeline/synthetic.py`.

### Public Inference API

```python
from countgd_pipeline import CountGDPipeline, synthetic_scene

pipe = CountGDPipeline.from_pretrained()
scene = synthetic_scene(1)
result = pipe.count(scene["image"], text=scene["label"], exemplars=scene["exemplars"])
print(result["results"][0]["count"], result["threshold"])
```

### Public Adaptation API

```python
from countgd_pipeline import CountGDPipeline, build_synthetic_dataset

splits = build_synthetic_dataset()
pipe = CountGDPipeline.from_pretrained()
frozen = pipe.evaluate(splits["test"])
report = pipe.adapt(splits["train"], splits["validation"], epochs=4, lr=2e-4)
adapted = pipe.evaluate(splits["test"])
pipe.save_artifact("outputs/adapter")
reloaded = CountGDPipeline.from_artifact("outputs/adapter")
```

### Verification records

- **Date:** 2026-09-24
- **Subject:** `tutorials/countgd_object_counting_colab.ipynb` at commit `8d61b94`, Git blob `c619a762`
- **Runtime:** CPU only, Python 3.12, `torch 2.14.0`, `transformers 4.57.6`
- **Procedure:** fresh Jupyter kernel, all cells in order, the pinned dependencies already installed, the source checkpoint pre-staged; the notebook fetched the Space card, the tokenizer and the photographs, and converted the checkpoint itself
- **Observed result:** 0 errors in 1,281.8 s; the pinned converted digest reproduced in the kernel; the values in Performance Measures; demo scene 35 / 51 / 35 → 35 / 35 / 35 (text / exemplars / both); reload parity 4 / 4 identical counts
- **Caveats:** a pre-flight on the build workstation, not a clean hosted runtime; one seeded draw, not an evaluation of FSC-147

- **Date:** 2026-09-24
- **Subject:** the same notebook, commit `8d61b94`, Git blob `c619a762`
- **Runtime:** Kaggle Tesla T4, Python 3.12.13, `torch 2.14.0+cu130`, `transformers 4.57.6`, installed by the notebook from its own pins
- **Procedure:** clean container with no repository checkout and empty caches, `Run all` in a fresh interpreter, all form fields at their defaults; the notebook fetched the checkpoint from the Space and converted it on the runtime
- **Observed result:** 16 / 16 code cells ok, one interpreter restart after the install cell, 468.7 s; synthetic test MAE 7.417 → 1.583, box F1 0.453 → 0.546; FSC-147 MAE 4.542 → 3.542; kept epoch 2 (validation MAE 0.875); reload parity 4 / 4 identical counts
- **Caveats:** the fine-tune's result differs from the CPU run because CUDA kernels are not bit-deterministic; one observation on one seeded draw

### Upstream References and Citations

- CountGD code: https://github.com/niki-amini-naieni/CountGD (MIT)
- CountGD Space (pinned checkpoint host): https://huggingface.co/spaces/nikigoli/countgd
- Amini-Naieni, N., Han, T. and Zisserman, A. (2024). *CountGD: Multi-Modal Open-World Counting.* NeurIPS 2024. https://arxiv.org/abs/2407.04619
- Liu, S. et al. (2023). *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection.* https://arxiv.org/abs/2303.05499
- Ranjan, V., Sharma, U., Nguyen, T. and Hoai, M. (2021). *Learning To Count Everything.* CVPR 2021 (FSC-147). https://github.com/cvlab-stonybrook/LearningToCountEverything
- FSC-147 Hub mirror: https://huggingface.co/datasets/isentropic/FSC147
- BERT tokenizer: https://huggingface.co/google-bert/bert-base-uncased

```bibtex
@inproceedings{AminiNaieni24,
    author    = "Amini-Naieni, N. and Han, T. and Zisserman, A.",
    title     = "CountGD: Multi-Modal Open-World Counting",
    booktitle = "NeurIPS",
    year      = "2024",
}
```
