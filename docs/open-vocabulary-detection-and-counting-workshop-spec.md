# DIMER Workshop Specification: Open-Vocabulary Detection & Counting

**Status:** Proposed  
**Notebook specification:** DIMER `NOTEBOOK_SPEC` **2.1**  
**Notebook profile:** `MULTI-CAPABILITY`  
**Pedagogical mode:** `WORKSHOP`  
**Proposed filename:** `DIMER_Open_Vocabulary_Detection_and_Counting_Workshop.ipynb`  
**Canonical runtime:** NVIDIA Tesla T4 or equivalent  
**Canonical execution:** standalone, credential-free, top-to-bottom `Run all`

---

# 1. Purpose

This workshop demonstrates two related prompt-driven computer-vision capabilities:

### Capability A — Open-vocabulary object detection

Compare:

- **Grounding DINO Tiny**
- **OWLv2 Base P16 Ensemble**

under the same:

- images;
- text phrases;
- ground-truth boxes;
- prompt templates;
- detection evaluator.

### Capability B — Open-world object counting

Use:

- **CountGD**

with:

- text-only prompts;
- exemplar-only prompts;
- text + exemplar prompts.

### Capability C — Detection-as-counting

Take the Grounding DINO and OWLv2 detections, de-duplicate them under one common post-processing rule, and simply count their boxes.

This lets the workshop ask:

> When is a dedicated counting model materially different from just counting detections?

---

# 2. Central workshop questions

The notebook should answer:

1. How do Grounding DINO and OWLv2 differ when asked to detect categories specified only in text?
2. How sensitive are detections to prompt wording?
3. How do threshold and duplicate-box behavior affect open-vocabulary detection?
4. Can an ordinary detector be turned into a counter by counting boxes?
5. When do exemplars help CountGD distinguish a target from look-alike distractors?
6. Can a model produce the correct count while localizing the wrong objects?
7. What is gained by combining text and visual exemplars?
8. How does bounded adaptation affect synthetic-domain detection/counting skill?

No single overall winner SHALL be produced.

---

# 3. Notebook profile

Declare:

**Profile:** `MULTI-CAPABILITY`  
**Mode:** `WORKSHOP`

This is preferable to forcing all three models into an `E2E` comparison because:

- Grounding DINO supports release-grade supervised adaptation;
- CountGD supports release-grade supervised adaptation;
- OWLv2's live DIMER capability is currently `TASK-INFERENCE`;
- its E2E adaptation carrier remains candidate.

Canonical structure:

```text
shared sample
→ validation

Capability A:
  Grounding DINO
  vs OWLv2
  → common detection evaluation

Capability B:
  CountGD text
  vs exemplar
  vs text+exemplar
  → count/localisation evaluation

Capability C:
  convert detector boxes into counts
  → compare against CountGD

→ freeze
→ independent test
→ cross-capability interpretation
→ export
```

Adaptation is a `FULL`-tier extension.

---

# 4. Execution tiers

Recommended:

```python
WORKSHOP_TIER = "STANDARD"  # @param ["STANDARD", "FULL"]
```

## STANDARD

Runs:

- Grounding DINO frozen
- OWLv2 frozen
- CountGD frozen
- common detection evaluation
- CountGD three prompt modes
- detection-as-counting
- prompt/threshold diagnostics
- freeze-before-test
- independent test

This SHOULD be the canonical release path.

## FULL

Additionally runs:

- Grounding DINO bounded adaptation
- CountGD bounded adaptation
- FSC-147 regression check for adapted CountGD
- fresh adapter export/reload

OWLv2 remains frozen until its E2E carrier is release-grade.

This separation is important because Grounding DINO + CountGD adaptation together materially increase T4 runtime.

---

# 5. Model A — Grounding DINO Tiny

**Model:** `IDEA-Research/grounding-dino-tiny`  
**Revision:**

```text
a2bb814dd30d776dcf7e30523b00659f4f141c71
```

License:

**Apache-2.0**

Checkpoint:

```text
model.safetensors
689,359,096 bytes
SHA-256:
1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3
```

Parameters:

approximately:

```text
172,249,090
```

Architecture:

```text
Swin-T visual backbone
+ BERT text encoder
+ multimodal feature enhancer
+ DETR-style encoder/decoder
+ 900 object queries
```

---

# 6. Grounding DINO input semantics

Input:

```text
image
+
1–16 text phrases
```

Native preprocessing:

```text
short edge = 800
long edge <= 1333
```

Prompt text:

- lowercase;
- phrase-based;
- BERT tokenization;
- total grounding text up to 256 tokens.

Its public interactive contract uses:

```text
box threshold = 0.4
text threshold = 0.3
```

but these MUST NOT be used as the canonical AP-evaluation cutoff.

For controlled evaluation the notebook SHALL use a low score floor.

---

# 7. Model B — OWLv2

**Model:**

`google/owlv2-base-patch16-ensemble`

**Revision:**

```text
cfd3195ba4ea9592eec887ded089f4c08eff231d
```

License:

**Apache-2.0**

Checkpoint:

```text
model.safetensors
619,918,824 bytes
SHA-256:
e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7
```

Architecture:

```text
CLIP ViT-B/16 image tower
+ CLIP text tower
+ 60 × 60 image-patch grid
→ 3,600 patch candidates
```

---

# 8. OWLv2 input semantics

Native preprocessing:

```text
pad image to square
→ resize 960 × 960
→ CLIP normalization
```

Each phrase becomes a separate CLIP text query.

Text:

```text
maximum 16 tokens per phrase
```

Every patch predicts:

- a box;
- one score per phrase.

No NMS is performed by the model's standard DIMER path.

This duplicate-box behavior is a workshop topic, not something to hide.

---

# 9. Model C — CountGD

**Model:** `nikigoli/countgd`  
**Pinned Space revision:**

```text
6e82e59569a84ee5c6aafa35d396f2d2bee57be2
```

License:

**MIT**

Source checkpoint:

```text
checkpoint_best_regular.pth
1,250,122,522 bytes
SHA-256:
c1bab864b17db345b4c6e3aaabb5765bc2c0a90d0bc8defb5e664a74a50aa126
```

DIMER-served converted checkpoint:

```text
countgd.safetensors
937,560,480 bytes
SHA-256:
8e44867b951e3a4205d918e022b78bc5fea218fd17c1851b864a01c421d2d443
```

Parameters:

```text
233,362,816
```

---

# 10. CountGD architecture

CountGD extends Grounding DINO with visual exemplar prompting.

Conceptually:

```text
image
+
text tokens
+
optional exemplar-box tokens
→ multimodal feature fusion
→ 900 object queries
→ boxes + prompt similarities
→ threshold
→ count
```

Each exemplar box is pooled from the image feature map and represented as an additional prompt token.

---

# 11. CountGD prompt modes

The workshop MUST compare:

### Text only

```text
"red circle"
```

### Exemplars only

Up to three boxes containing representative target instances.

### Text + exemplars

```text
"red circle"
+
three target exemplar boxes
```

These are distinct inference conditions.

---

# 12. Common runtime

All three capabilities can share approximately:

```text
Python 3.12
torch==2.14.0
torchvision==0.29.0
torchaudio==2.11.0
transformers==4.57.6
huggingface-hub==0.36.2
safetensors==0.8.0
numpy==2.5.3
pillow==11.3.0
scipy==1.18.1
```

The notebook SHOULD prefer one runtime unless clean qualification exposes conflicts.

---

# 13. Shared canonical dataset

Reuse and extend the existing CountGD deterministic synthetic dataset.

Existing split:

| Split | Scenes |
|---|---:|
| Train | **24** |
| Validation | **8** |
| Test | **12** |
| Total | **44** |

Seed ranges:

```text
train      : 1000+
validation : 2000+
test       : 3000+
```

Image size:

```text
512 × 384
```

---

# 14. Scene composition

Each scene contains:

### Target objects

```text
6–40
```

objects sharing:

- one colour;
- one shape.

### Distractors

```text
4–20
```

objects with:

- another colour;
- another shape.

### Clutter

```text
300 random specks
```

---

# 15. Shape vocabulary

Shapes:

```text
circle
square
triangle
```

Colours:

```text
red
blue
green
yellow
```

Potential phrase vocabulary:

```text
12 combinations
```

such as:

```text
red circle
blue triangle
green square
yellow circle
```

---

# 16. Why this dataset works for all three models

The same scene supports:

### Detection

Ground truth for:

```text
target phrase
distractor phrase
```

Example:

```text
target:
  blue circle

distractor:
  green triangle
```

### Counting

Target:

```text
blue circle
```

with:

- exact count;
- exact target boxes;
- exact target centers;
- three exemplar boxes.

This gives all three models exactly the same source pixels.

---

# 17. Generator extension

The existing CountGD generator records:

- target boxes;
- target points;
- target label;
- exemplar boxes;
- distractor label;
- distractor count.

The comparative workshop SHALL additionally record:

```text
distractor_boxes
distractor_points
```

No rendering semantics need to change.

The only extension is retaining the distractor annotations already known during drawing.

---

# 18. Canonical record

```text
{
  "id": "synth-03000",
  "image": PIL.Image,
  "split": "test",

  "target_label": "blue circle",
  "target_boxes": [...],
  "target_points": [...],
  "target_count": 35,

  "exemplars": [...],

  "distractor_label": "green triangle",
  "distractor_boxes": [...],
  "distractor_points": [...],
  "distractor_count": 16
}
```

---

# 19. Why not use BCCD as the common dataset

Grounding DINO's live E2E tutorial uses the real BCCD microscopy corpus.

However OWLv2 explicitly has no validated medical-imaging evidence and its upstream training regime is photographic/web-centric.

Using BCCD as the core comparison would unnecessarily mix:

> open-vocabulary detector differences

with:

> severe domain shift into blood microscopy.

The synthetic shared corpus is therefore cleaner for the **comparative** workshop.

BCCD remains appropriate in the dedicated Grounding DINO E2E notebook.

---

# 20. Why not use FSC-147 for all three

FSC-147 is excellent for counting but provides:

- points;
- target counts;
- three exemplar boxes;

rather than complete boxes for every object.

That prevents a full common detection mAP evaluation for Grounding DINO and OWLv2.

Therefore FSC-147 remains a CountGD-specific regression check in `FULL`.

---

# 21. Dataset manifest

Before model acquisition export:

```text
outputs/data/dataset_manifest.json
```

containing:

```text
generator version
split seed ranges
image size
target/distractor vocabulary
scene IDs
pixel SHA-256
target box annotations
distractor box annotations
counts
exemplar IDs/boxes
```

The notebook MUST assert no image digest occurs across multiple splits.

---

# 22. Capability A — common detection task

For every scene, both Grounding DINO and OWLv2 receive exactly:

```text
image
prompts = [
  target_label,
  distractor_label
]
```

Example:

```text
["blue circle", "green triangle"]
```

Ground truth consists of **both classes**.

This prevents the detector from being rewarded merely for finding the target and ignoring visually distinct distractors.

---

# 23. Canonical prompt wording

Primary comparison uses the exact bare object phrases:

```text
blue circle
green triangle
```

No architecture-specific prompt engineering is permitted in the core comparison.

Grounding DINO and OWLv2 receive semantically identical phrases.

---

# 24. Why not use different “best” prompts per model

OWLv2 documentation often uses:

```text
a photo of a <thing>
```

Grounding DINO commonly uses bare noun phrases.

Allowing separate prompt templates in the main experiment would confound:

```text
architecture
with
prompt engineering
```

Therefore the core uses one wording contract.

Model-specific phrasing belongs in the prompt-sensitivity experiment.

---

# 25. Common detection score rule

For evaluation, every candidate box SHALL receive:

```text
best phrase
best phrase score
```

Exactly one phrase is assigned per candidate.

This mirrors the comparison question:

> Which of the two requested concepts best describes this candidate box?

Grounding DINO's token-level similarities MUST be reduced to one score per supplied phrase.

OWLv2 already produces one image-text logit per phrase.

---

# 26. Detection evaluation floor

Canonical evaluation score floor:

```text
0.05
```

Purpose:

- preserve recall;
- bound output volume;
- avoid using architecture-specific demo thresholds.

This is an **evaluation floor**, not the deployed decision threshold.

---

# 27. Common candidate cap

Maximum detections entering evaluation:

```text
900 per image
```

Why:

- Grounding DINO has 900 object queries;
- OWLv2 has 3,600 patch candidates.

Capping score-ordered candidates at 900 gives both systems the same evaluator budget.

The notebook SHOULD report whether the cap was ever reached.

---

# 28. Primary evaluation uses no added NMS

Main Grounding DINO/OWLv2 evaluation SHOULD use their normalized raw output without notebook-added NMS.

This preserves system behavior:

- OWLv2 may generate duplicate boxes;
- Grounding DINO query outputs may overlap.

Duplicate handling is itself a meaningful difference.

---

# 29. Secondary common-NMS experiment

A separate diagnostic MAY apply:

```text
class-wise NMS IoU = 0.50
```

to both systems.

Report:

```text
raw
vs
common NMS
```

This answers:

> How much of each detector's apparent error/counting behavior comes from duplicate predictions rather than failure to localize the object?

NMS results MUST NOT replace the primary raw-output metrics.

---

# 30. Common detection metrics

Implement one notebook-owned evaluator.

Required:

### AP

IoU thresholds:

```text
0.50:0.05:0.95
```

### AP50

### AP75

### Recall@50

### Per-phrase AP50

Every scene contributes:

- target phrase;
- distractor phrase.

---

# 31. Additional detection diagnostics

Report:

- detections per image;
- false positives/image;
- missed objects/image;
- mean matched IoU;
- duplicate detections per matched ground-truth object;
- target-class recall;
- distractor-class recall.

These help explain AP differences.

---

# 32. Empty detector baseline

Baseline:

```text
return no boxes
```

Expected:

```text
AP = 0
recall = 0
```

This validates the common evaluator.

---

# 33. Grid-prior baseline

Reuse the Grounding DINO tutorial concept:

```text
mean-sized boxes tiled over image
+
majority/target phrase
```

Score through the same evaluator.

This is a deliberately weak geometric prior.

---

# 34. Prompt-sensitivity experiment

Using **validation only**, compare:

### Bare

```text
red circle
```

### Photo template

```text
a photo of a red circle
```

For each detector report:

```text
AP50
recall@50
detections/image
```

Do not select a test prompt template based on test results.

Canonical test wording remains the frozen bare phrase unless the experiment explicitly predefines another selection rule.

---

# 35. Negative-prompt control

Every validation scene SHOULD also receive one absent phrase, for example:

```text
purple star
```

where stars are never drawn.

Report:

```text
false boxes / image
```

at several thresholds.

This demonstrates an important open-vocabulary property:

> A text query does not guarantee that the named object actually exists in the image.

---

# 36. Threshold-sensitivity experiment

Validation-only thresholds:

```text
0.05
0.10
0.20
0.30
0.40
0.50
```

For both detection models show:

- precision;
- recall;
- AP50 if recomputed after thresholding;
- detections/image.

Raw scores MUST NOT be described as calibrated probabilities.

---

# 37. Capability B — CountGD counting task

For each scene CountGD counts only:

```text
target_label
```

Example:

```text
blue circle
```

Ground truth:

```text
target_count
target_points
target_boxes
```

Distractors MUST NOT contribute to the target count.

---

# 38. CountGD text-only condition

Input:

```text
text = target_label
exemplars = []
```

This tests semantic prompting.

Question:

> Can the model distinguish the target phrase from visually different distractors using text alone?

---

# 39. CountGD exemplar-only condition

Input:

```text
text = None
exemplars = three target boxes
```

This tests visual matching.

The notebook SHOULD highlight the known failure mode:

> Exemplars identify appearance, not necessarily semantic category.

Objects with similar shape/texture may be counted even if they are conceptually different.

---

# 40. CountGD text + exemplars

Input:

```text
text = target_label
exemplars = three target boxes
```

This combines:

- semantic information;
- in-image appearance information.

It SHOULD be the primary CountGD condition.

---

# 41. CountGD threshold

Canonical frozen threshold:

```text
0.23
```

This is the upstream authors' value.

It is not calibrated for the workshop domain.

Validation threshold sensitivity SHOULD include:

```text
0.15
0.20
0.23
0.30
0.40
```

Test threshold remains frozen before test.

---

# 42. Count metrics

Required:

### MAE

\[
MAE=\frac{1}{N}\sum|\hat c-c|
\]

### RMSE

\[
RMSE=\sqrt{\frac{1}{N}\sum(\hat c-c)^2}
\]

### NAE

\[
NAE=\frac{1}{N}\sum\frac{|\hat c-c|}{c}
\]

Also:

- exact-count fraction;
- under-count fraction;
- over-count fraction.

---

# 43. Counting localisation metrics

A correct numerical count does not prove the correct objects were counted.

Therefore also calculate:

### Point precision / recall / F1

Predicted box centers matched one-to-one against gold target points.

### Box precision / recall / F1 at IoU 0.5

### Mean matched IoU

This allows cases such as:

```text
gold count = 10
predicted count = 10
but only 7 predicted boxes match target objects
```

to be recognized as errors.

---

# 44. Count baseline A — mean count

Compute the mean target count from **training scenes only**.

For every validation/test image predict:

```text
round(training mean)
```

This provides a simple non-image baseline.

---

# 45. Count baseline B — template matcher

Reuse the existing CountGD tutorial's lightweight exemplar/template matching baseline.

This asks:

> Can simple appearance matching compete with the foundation model on these rendered shapes?

Score:

- MAE;
- RMSE;
- point localization if supported.

---

# 46. Capability C — use detectors as counters

For Grounding DINO and OWLv2:

```text
detect target phrase
→ common classwise NMS
→ count retained target boxes
```

This creates:

```text
Grounding DINO-as-counter
OWLv2-as-counter
```

The result SHOULD be scored with the same:

- MAE;
- RMSE;
- NAE;
- exact fraction.

---

# 47. Why detection-as-counting matters

A dedicated counting model is not automatically necessary.

If:

```text
detector
+
de-duplication
```

already counts accurately, it may be sufficient.

But detection-based counting can fail through:

- duplicate boxes;
- low recall in dense scenes;
- threshold sensitivity;
- NMS suppression of adjacent objects;
- candidate/query ceilings.

The workshop measures rather than assumes this tradeoff.

---

# 48. Validation selection for detection-as-counting

Using validation only, each detector MAY choose a count threshold from a fixed grid:

```text
0.05
0.10
0.15
0.20
0.30
0.40
```

Selection criterion:

> lowest validation count MAE after common classwise NMS.

Tie-break:

1. lower MAE;
2. lower RMSE;
3. higher threshold.

Store the chosen threshold before test.

This is separate from the canonical detection AP floor.

---

# 49. Cross-capability count table

Independent test should include:

| System | Prompt mode | MAE | RMSE | NAE | Exact | Point F1 | Box F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Mean count | — | | | | | — | — |
| Template matcher | exemplar | | | | | | |
| Grounding DINO-as-counter | text | | | | | | |
| OWLv2-as-counter | text | | | | | | |
| CountGD | text | | | | | | |
| CountGD | exemplar | | | | | | |
| CountGD | text+exemplar | | | | | | |

No composite winner.

---

# 50. Density analysis

The synthetic target count spans approximately:

```text
6–40 objects
```

Group test scenes into:

```text
low density
medium density
high density
```

using thresholds determined from the training distribution.

Report MAE by density group.

This tests whether systems fail disproportionately as object count increases.

---

# 51. Object-size analysis

Target radius varies approximately:

```text
7–14 px
```

before model resizing.

Group scenes by mean target-box area.

Compare:

- detector recall;
- CountGD MAE.

This illustrates the effect of small-object scale.

---

# 52. Distractor analysis

Record:

```text
target count
distractor count
target/distractor shape
target/distractor colour
```

Analyze count error against:

```text
distractor-to-target ratio
```

This is particularly useful for exemplar-only CountGD.

---

# 53. Model-specific preprocessing

The core comparison controls source images, not model transforms.

### Grounding DINO

```text
shortest side 800
longest <= 1333
ImageNet normalization
```

### CountGD

approximately the same GroundingDINO-style 800/1333 preprocessing.

### OWLv2

```text
square-pad
→ 960 × 960
→ CLIP normalization
```

The notebook MUST state:

> These are comparisons of complete available pretrained systems, not an architecture-only controlled experiment.

---

# 54. Freeze-before-test

Before test create:

```text
outputs/frozen/frozen_experiment.json
```

containing:

```text
dataset digest
scene IDs
generator version

Grounding DINO revision
OWLv2 revision
CountGD revision/digest

canonical detection prompts
detection score floor
candidate cap
NMS diagnostic policy

CountGD threshold
CountGD prompt modes

detector-as-counter selected thresholds

metric definitions
validation results
```

Nothing may change after test metrics are viewed.

---

# 55. Independent test

Run exactly the 12 scenes:

```text
seeds 3000–3011
```

with the frozen:

- prompt wording;
- score floors;
- counting thresholds;
- NMS policies;
- model revisions.

No test-dependent threshold adjustment.

---

# 56. Primary detection table

| Model | AP | AP50 | AP75 | Recall@50 | Dets/image | Duplicate rate |
|---|---:|---:|---:|---:|---:|---:|
| Empty | 0 | 0 | 0 | 0 | 0 | — |
| Grid prior | | | | | | |
| Grounding DINO | | | | | | |
| OWLv2 | | | | | | |

Also produce:

```text
target AP50
distractor AP50
```

per scene/class grouping.

---

# 57. Raw versus common-NMS table

| Model | Output | AP50 | Recall | Dets/image | Count MAE |
|---|---|---:|---:|---:|---:|
| Grounding DINO | raw | | | | — |
| Grounding DINO | common NMS | | | | |
| OWLv2 | raw | | | | — |
| OWLv2 | common NMS | | | | |

This is diagnostic only.

---

# 58. Prompt-sensitivity table

Validation:

| Model | Prompt template | AP50 | Recall@50 | FP on absent prompt |
|---|---|---:|---:|---:|
| Grounding DINO | `{label}` | | | |
| Grounding DINO | `a photo of a {label}` | | | |
| OWLv2 | `{label}` | | | |
| OWLv2 | `a photo of a {label}` | | | |

This teaches that **the prompt itself is part of the model input distribution**.

---

# 59. Detection visualizations

For deterministic test scenes show:

### Scene A

low target count.

### Scene B

highest target count.

### Scene C

largest Grounding DINO/OWLv2 disagreement.

Panels:

```text
ground truth
Grounding DINO
OWLv2
```

Use consistent phrase colours.

---

# 60. Counting visualizations

For deterministic scenes show:

```text
gold target boxes/points
CountGD text-only
CountGD exemplar-only
CountGD text+exemplar
```

Include:

```text
gold count
predicted count
MAE for scene
```

---

# 61. Interesting failure gallery

Select by deterministic metric rules:

1. exact CountGD text+exemplar count;
2. largest CountGD over-count;
3. largest CountGD under-count;
4. exemplar-only false-distractor case;
5. detector count closest to CountGD;
6. detector/CountGD largest disagreement.

Do not hand-pick visually convenient examples.

---

# 62. FULL tier — Grounding DINO adaptation

Grounding DINO MAY be adapted to the complete synthetic phrase vocabulary:

```text
red circle
red square
red triangle
blue circle
...
yellow triangle
```

Total:

```text
12 phrases
```

Train:

**24 scenes**

Validation:

**8 scenes**

Test:

**12 scenes**

---

# 63. Grounding DINO bounded adaptation

Preserve the current DIMER recipe:

Train:

- cross-modality decoder;
- reference-point head;
- box heads;
- contrastive heads.

Freeze:

- image backbone;
- BERT text encoder;
- feature enhancer;
- query-selection stage.

Approximate trainable parameters:

```text
11,187,460 / 172,249,090
```

---

# 64. Grounding DINO adaptation recipe

Canonical:

```text
epochs = 6
batch size = 4
learning rate = 5e-5
gradient clip = 0.1
```

Objective:

```text
Hungarian matching
+ sigmoid focal alignment
+ L1 box loss
+ GIoU
```

Selection:

> highest validation mAP50

The frozen model MUST remain a reported baseline.

---

# 65. Grounding DINO adaptation interpretation

Adapting to the 12 synthetic colour/shape phrases teaches the decoder this rendering domain.

It does **not** imply improved general open-vocabulary detection.

The notebook MUST state:

> A phrase outside this synthetic adaptation vocabulary remains an open-vocabulary inference request, but its behavior after adaptation has not been validated.

---

# 66. FULL tier — CountGD adaptation

Preserve the live CountGD bounded recipe.

Trainable:

- last 2 of 6 decoder layers;
- decoder norm;
- shared box head.

Trainable parameters:

```text
3,619,584
```

Artifact:

approximately:

```text
14.5 MB
```

---

# 67. CountGD adaptation recipe

Canonical:

```text
epochs = 4
learning rate = 2e-4
seed = 0
```

Training:

**24 synthetic scenes**

Validation:

**8 synthetic scenes**

Selection:

> lowest validation count MAE

The frozen model remains eligible if no epoch improves validation MAE.

---

# 68. CountGD regression check

Because CountGD adaptation occurs on the synthetic rendering domain, `FULL` SHOULD additionally evaluate:

**24 digest-pinned FSC-147 test photographs**

from:

`isentropic/FSC147`

revision:

```text
3e420cb6537e803dd6d4516623ce82a79c0317b8
```

Purpose:

> Did synthetic adaptation improve the workshop domain while degrading ordinary photographic counting?

This is a regression check, not part of the Grounding DINO/OWLv2 comparison.

---

# 69. OWLv2 adaptation boundary

OWLv2 MUST remain frozen in this workshop until its DIMER E2E carrier is promoted from candidate.

The notebook MUST NOT quietly implement an unqualified fine-tuning recipe merely for symmetry.

When the OWLv2 E2E carrier becomes release-grade, it can be added as a future `FULL` adaptation branch.

---

# 70. Adaptation result tables

## Detection

```text
Grounding DINO frozen
Grounding DINO adapted
OWLv2 frozen
```

Compare using the same test detector evaluator.

## Counting

```text
CountGD frozen
CountGD adapted
```

Compare:

- synthetic MAE/RMSE;
- localization F1;
- box F1;
- FSC-147 regression MAE.

Do not combine those into one metric.

---

# 71. Adapter artifacts

### Grounding DINO

Export its existing SafeTensors adapter format containing only the trained decoder/head tensors plus manifest.

### CountGD

Format:

```text
org.valcorza.countgd.adapter.v1
```

with:

```text
adapter.safetensors
manifest.json
```

Every artifact MUST name:

- immutable base identity;
- base revision;
- base weight digest;
- tensor names;
- adaptation recipe;
- selected epoch;
- file size;
- SHA-256.

---

# 72. Fresh reload verification

For both adapted models:

```text
destroy adapted model
→ reconstruct fresh base
→ verify base files
→ verify adapter manifest/digest
→ apply adapter
→ rerun fixed validation scene
```

Verify:

### Grounding DINO

- same labels;
- same boxes within tolerance;
- same scores within tolerance.

### CountGD

- identical count;
- same boxes/points within tolerance;
- same scores within tolerance.

---

# 73. New-scene inference

Generate new scenes using seeds outside all train/validation/test ranges.

Recommended:

```text
4000
4001
4002
```

Run:

- Grounding DINO;
- OWLv2;
- CountGD text;
- CountGD exemplar;
- CountGD combined.

Since generator truth is known, these MAY be scored as **post-test sanity examples**, but they MUST NOT alter the primary test summary.

---

# 74. Computational comparison

Record:

| Model | Weight size | Device | Load | sec/image | Output candidates | Notes |
|---|---:|---|---:|---:|---:|---|
| Grounding DINO Tiny | ~689 MB | T4 | | | ≤900 | text grounding |
| OWLv2 Base P16 | ~620 MB | T4 | | | ≤3600 | 960×960 ViT |
| CountGD | ~938 MB | T4 | | | ≤900 | text + exemplars |

Also report in `FULL`:

- adaptation time;
- trainable parameters;
- adapter bytes;
- peak VRAM.

Models SHOULD be loaded sequentially and unloaded between capabilities.

---

# 75. Why the models should not remain loaded together

Combined checkpoint footprint exceeds:

```text
2.2 GB
```

before activations and framework memory.

Canonical pattern:

```text
load Grounding DINO
→ evaluate/export
→ unload

load OWLv2
→ evaluate/export
→ unload

load CountGD
→ evaluate/adapt/export
→ unload
```

This also makes memory attribution clearer.

---

# 76. BYOD dataset contract

A full measurable BYOD archive SHOULD contain:

```text
dataset.zip
├── scenes.csv
├── boxes.csv
└── images/
```

`scenes.csv`:

```text
image_id,file,target_label,split
scene001,images/001.jpg,red apple,train
```

`boxes.csv`:

```text
image_id,label,x0,y0,x1,y1,is_exemplar
scene001,red apple,10,20,60,80,true
scene001,red apple,80,20,130,80,false
scene001,green apple,200,50,250,110,false
```

---

# 77. Why two BYOD tables

Detection needs:

```text
all object labels and boxes
```

Counting needs:

```text
one target label per scene
+
target instances
+
optional exemplars
```

The two-table contract cleanly supports both without inventing separate datasets.

---

# 78. BYOD split

Preferred:

explicit:

```text
train
validation
test
```

If omitted:

```text
60 / 20 / 20
seed 42
```

at the image level.

The notebook MUST check:

- target labels occur in every required split;
- no image crosses split boundaries;
- exemplar boxes correspond to target objects.

---

# 79. BYOD validation

Reject:

- remote image URLs;
- ZIP traversal;
- symlinks;
- unreadable images;
- duplicate IDs;
- invalid boxes;
- boxes outside images;
- labels longer than declared prompt ceilings;
- more than three exemplar boxes;
- exemplar box smaller than 2 px;
- test-only target categories absent from train when adaptation is requested.

---

# 80. BYOD privacy guidance

The notebook MUST state:

> User-supplied images, text queries, annotations, and exemplar boxes are processed inside the selected notebook runtime and are not submitted to a DIMER worker or API. A hosted notebook remains an external compute environment. Do not upload confidential, personal, biometric, surveillance, security-sensitive, restricted, or proprietary imagery unless authorized.

---

# 81. Person-related prompt boundary

The built-in workshop contains no people.

The notebook SHOULD warn that open-vocabulary prompts can name:

- people;
- clothing;
- age-related descriptions;
- religious dress;
- disability-related objects;
- other personal attributes.

Neither detector's phrase score is evidence that such an attribute is actually present.

Person-related or demographic deployments require separate fairness, consent, privacy, and application-specific review.

---

# 82. Score semantics

None of the three model scores are calibrated probabilities.

### Grounding DINO

Sigmoid text-token similarity.

### OWLv2

Sigmoid image-text query logit.

### CountGD

Best prompt-token similarity for the object query.

Therefore:

```text
0.8 Grounding DINO
!=
0.8 OWLv2
!=
0.8 CountGD
```

in probabilistic meaning.

---

# 83. Counting ceiling

Grounding DINO and CountGD use approximately:

```text
900 queries
```

per pass.

CountGD's carried inference path does not implement the authors' dense-image cropping strategy.

Therefore the workshop MUST state:

> Very dense scenes approaching or exceeding the query budget cannot be counted reliably in one pass.

The built-in 6–40 target objects remain well below this limit.

---

# 84. Prompt semantics limitation

Open vocabulary does not mean perfect natural-language understanding.

Results can change with:

- singular/plural;
- adjective order;
- synonyms;
- longer descriptions;
- spelling;
- prompt templates.

The prompt is part of the experiment and MUST be recorded in provenance.

---

# 85. Interpretation boundary — synthetic imagery

The canonical scenes are:

- flat rendered shapes;
- flat backgrounds;
- artificial clutter.

They are deliberately useful for exact box/count supervision.

They do not establish performance on:

- photographs;
- aerial imagery;
- microscopy;
- documents;
- industrial cameras;
- wildlife;
- people.

---

# 86. Open-vocabulary boundary

Open-vocabulary detection means:

> the inference class names are supplied as text rather than being limited to one fixed trained classifier head.

It does **not** mean:

- every imaginable concept is detectable;
- all phrases are understood;
- attributes/relations are reliable;
- class scores are calibrated;
- absence can be proven.

---

# 87. Detection-versus-counting lesson

The concluding interpretation SHOULD emphasize:

### Detection asks:

> Where are the instances, and which prompted phrase best matches them?

### Counting asks:

> How many instances of this prompted concept are present?

Those tasks overlap but optimize different behavior.

A dedicated counter can outperform box-counting when:

- instances are dense;
- duplicate detections are common;
- exact localization is less important than count;
- visual exemplars provide category information not captured by text.

Conversely, a detector can be sufficient when:

- objects are sparse;
- localization is already required;
- NMS/de-duplication is reliable.

---

# 88. Output structure

```text
outputs/
├── data/
│   ├── dataset_manifest.json
│   ├── train.json
│   ├── validation.json
│   └── test.json
│
├── detection/
│   ├── validation/
│   │   ├── grounding_dino.csv
│   │   ├── owlv2.csv
│   │   ├── prompt_sensitivity.csv
│   │   └── threshold_sweep.csv
│   └── test/
│       ├── aggregate_metrics.csv
│       ├── per_phrase_metrics.csv
│       ├── raw_vs_nms.csv
│       └── predictions/
│
├── counting/
│   ├── validation/
│   └── test/
│       ├── aggregate_metrics.csv
│       ├── per_scene_metrics.csv
│       ├── density_metrics.csv
│       └── predictions/
│
├── cross_capability/
│   ├── detection_as_counting.csv
│   └── comparison.csv
│
├── adaptation/
│   ├── grounding_dino/
│   └── countgd/
│
├── frozen/
│   └── frozen_experiment.json
│
├── artifacts/
├── figures/
├── new_data/
├── provenance/
│   └── experiment_manifest.json
└── workshop_summary.json
```

---

# 89. Provenance

Record:

```text
notebook_spec
profile
mode
execution_tier
workshop_revision

dataset:
  generator version
  seed ranges
  scene digests
  target labels
  distractor labels
  counts
  boxes
  exemplars

grounding_dino:
  model id
  revision
  weight digest
  prompt format
  score floor

owlv2:
  model id
  revision
  weight digest
  prompt format
  score floor

countgd:
  model id
  revision
  converted weight digest
  count threshold
  prompt modes

detection_evaluation:
  candidate cap
  IoU thresholds
  NMS diagnostic policy

count_evaluation:
  count metrics
  point matching rule
  box IoU threshold

selected validation thresholds

adaptation configuration
artifact identities
```

---

# 90. Notebook metadata

```json
{
  "dimer": {
    "notebook_spec": "2.1",
    "notebook_profile": "MULTI-CAPABILITY",
    "notebook_mode": "WORKSHOP",
    "standalone": true,
    "capability": "open-vocabulary-detection-and-counting",
    "carrier": "comparative text-prompted detection and open-world counting workshop",
    "dataset": "44 deterministic synthetic target+distractor scenes with exact boxes, points, counts and exemplars",
    "default_tier": "STANDARD",
    "canonical_runtime": "NVIDIA Tesla T4",
    "worker_required": false,
    "credentials_required": false,
    "clean_runtime_evidence": "pending"
  }
}
```

---

# 91. Release acceptance — STANDARD

| Requirement | Required |
|---|---:|
| Notebook Spec 2.1 | PASS |
| `MULTI-CAPABILITY` / `WORKSHOP` | PASS |
| Fresh T4 `Run all` | PASS |
| No Git clone | PASS |
| No DIMER runtime source fetch | PASS |
| No services / credentials | PASS |
| Shared 24/8/12 sample | PASS |
| Target + distractor boxes | PASS |
| Grounding DINO snapshot verification | PASS |
| OWLv2 snapshot verification | PASS |
| CountGD converted weight verification | PASS |
| Grounding DINO inference | PASS |
| OWLv2 inference | PASS |
| Common detection AP evaluator | PASS |
| Prompt-sensitivity experiment | PASS |
| Negative-prompt control | PASS |
| CountGD text-only | PASS |
| CountGD exemplar-only | PASS |
| CountGD text+exemplar | PASS |
| Count MAE/RMSE/NAE | PASS |
| Point localization | PASS |
| Box localization | PASS |
| Detection-as-counting | PASS |
| Freeze-before-test | PASS |
| Independent 12-scene test | PASS |
| BYOD validation path | PASS |
| Provenance export | PASS |

---

# 92. Release acceptance — FULL

Additionally:

| Requirement | Required |
|---|---:|
| Grounding DINO adaptation | PASS |
| Validation mAP50 selection | PASS |
| Grounding DINO artifact/reload | PASS |
| CountGD adaptation | PASS |
| Validation MAE selection | PASS |
| CountGD artifact/reload | PASS |
| FSC-147 regression check | PASS |
| Adaptation runtime/VRAM record | PASS |

OWLv2 adaptation:

```text
N/A until live E2E release
```

---

# 93. Suggested registry entry

```markdown
| Notebook | Profile | Mode | Capability | Runtime | Sample | BYOD | Run-all | Status |
|---|---|---|---|---|---|---|---|---|
| `DIMER_Open_Vocabulary_Detection_and_Counting_Workshop.ipynb` | `MULTI-CAPABILITY` | `WORKSHOP` | Grounding DINO vs OWLv2 detection + CountGD open-world counting | T4 | 44 seeded synthetic target/distractor scenes with exact boxes, points, counts and exemplars | yes | pending | candidate |
```

---

# 94. Implementation principle

The notebook should be built around one shared scene contract:

> **same image → same target phrase → same distractor phrase → same exact boxes**

Then branch:

```text
Grounding DINO ─┐
                ├→ open-vocabulary detection evaluation
OWLv2 ──────────┘

Grounding DINO ─┐
OWLv2 ──────────┼→ de-duplicate boxes → count → count error
CountGD ────────┘

CountGD:
  text
  exemplar
  text+exemplar
→ dedicated counting/localisation evaluation
```

That gives the workshop a much stronger conceptual story than simply running three unrelated models.

The core lesson becomes:

> **Open-vocabulary detection and open-world counting share prompt-conditioned localization, but they are not interchangeable tasks. A detector can be used as a counter, yet duplicate handling, density, thresholding, exemplars, and the objective itself determine whether that is actually a good idea.**