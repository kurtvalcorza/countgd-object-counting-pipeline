"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package (eight
modules, carried verbatim in dependency order), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

Generator /2 keys in use: ``modules`` lists every module of ``src/countgd_pipeline/`` except
``__init__.py``; ``entry_module`` is ``config.py`` (it holds ``MODEL_ID``/``MODEL_REVISION``/
``MODEL_LICENSE`` and the model key under the package's own spelling ``DEFAULT_MODEL_KEY``, mapped by
``identity_names``); ``extra_weights`` carries the second pinned snapshot, the BERT tokenizer (vocabulary
and configurations, no weights); ``model_host`` names the authors' Hugging Face Space the checkpoint is
fetched from; ``rewrites`` carries the two ``__file__`` uses in ``model.py`` (the weights root and the
repository-checkout convenience in ``resolve_weights_path``) so the standalone notebook resolves both to the
working directory; ``model_load`` lets the pipeline pick CUDA when it is visible (CPU is the fallback).

This template configures an E2E open-world counting workflow: the authors' Space snapshot is
digest-verified, its pickle checkpoint statically audited and converted once into the served safetensors
file, and loaded strictly into the vendored network; synthetic counting scenes with known object boxes and
80 digest-pinned FSC-147 test photographs are validated and split; the demo scene is counted by text, by
exemplar boxes and by both through the inference contract; the frozen model is scored against two
non-neural baselines on held-out scenes (count error, point localisation and box IoU) and on the FSC-147
photographs; a bounded counting fine-tune runs on the synthetic training scenes with validation-MAE epoch
selection; both held-out sets are scored again; and the adapter is exported and reloaded with parity.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "countgd_pipeline",
    "repo_name": "countgd-object-counting-pipeline",
    "stem": "countgd_object_counting",
    "notebook_name": "countgd_object_counting_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "pipeline_class": "CountGDPipeline",
    "weights_key": "countgd",
    "modules": ["config.py", "modeling.py", "model.py", "metrics.py", "samples.py", "synthetic.py", "pipeline.py", "provenance.py"],
    "entry_module": "config.py",
    "identity_names": {"MODEL_KEY": "DEFAULT_MODEL_KEY"},
    "extra_weights": [
        {
            "key": "bert-base-uncased",
            "var": "TOKENIZER_MANIFEST",
            "dir": "TOKENIZER_WEIGHTS_DIR",
            "identity": ["TOKENIZER_MODEL_ID", "TOKENIZER_REVISION"],
            "stage": "stage_missing_tokenizer_files",
            "verify": "verify_tokenizer_snapshot",
        }
    ],
    "model_host": {"name": "the Hugging Face Hub", "reference_url": "https://huggingface.co/spaces/nikigoli/countgd", "revision_label": "Space revision"},
    "rewrites": [
        ['^_WEIGHTS_ROOT = Path\\(__file__\\)\\.resolve\\(\\)\\.parents\\[2\\] / "weights"$', '_WEIGHTS_ROOT = Path.cwd() / "weights"  # standalone rewrite (build_notebook.py): working-directory snapshots, no repository checkout'],
        ["^    repo_root = Path\\(__file__\\)\\.resolve\\(\\)\\.parents\\[2\\]$", "    repo_root = Path.cwd()  # standalone rewrite (build_notebook.py): no repository checkout to resolve"],
    ],
    "model_load": "CountGDPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, tokenizer_dir=TOKENIZER_WEIGHTS_DIR)",
    "runtime_imports": ["torch", "transformers", "numpy"],
    "title": "CountGD Open-World Object Counting Pipeline — DIMER E2E counting tutorial (standalone)",
    "badges": [
        ("GitHub", "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white", "https://github.com/kurtvalcorza/countgd-object-counting-pipeline"),
        ("Open In Colab", "https://colab.research.google.com/assets/colab-badge.svg", "https://colab.research.google.com/github/kurtvalcorza/countgd-object-counting-pipeline/blob/main/tutorials/countgd_object_counting_colab.ipynb"),
        ("Hugging Face Space", "https://img.shields.io/badge/%F0%9F%A4%97%20Space-nikigoli%2Fcountgd-ffcc4d?style=flat", "https://huggingface.co/spaces/nikigoli/countgd"),
        ("Upstream", "https://img.shields.io/badge/Upstream-niki--amini--naieni%2FCountGD-181717?style=flat&logo=github&logoColor=white", "https://github.com/niki-amini-naieni/CountGD"),
        ("arXiv", "https://img.shields.io/badge/arXiv-2407.04619-b31b1b.svg", "https://arxiv.org/abs/2407.04619"),
    ],
    "capability": "open-world object counting from a text prompt, up to three exemplar boxes, or both — a count, one box and one point per counted object — with count, point-localisation and box-IoU evaluation and a bounded counting fine-tune, using the pinned CountGD checkpoint",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "authors' pinned CountGD snapshot (a 1.25 GB pickle checkpoint, statically audited and converted once in this kernel "
        "into the 938 MB `countgd.safetensors` that is actually loaded) and the BERT tokenizer snapshot, draws the seeded "
        "synthetic counting scenes (24 / 8 / 12 training, validation and test scenes with known object boxes) and fetches the 80 "
        "pinned FSC-147 test photographs (about 2.7 MB, each refused on any byte-size or SHA-256 mismatch), validates and splits "
        "both without leakage, counts the demo scene by text, by exemplar boxes and by both through the inference contract with "
        "an input manifest and a rejection probe, scores the frozen model beside a mean-count baseline and a template matcher on "
        "the held-out scenes (count error, point localisation and box IoU) and on 24 FSC-147 photographs, runs a bounded "
        "counting fine-tune of the last two decoder layers and the shared box head on the training scenes with validation-MAE "
        "epoch selection, scores both held-out sets again, re-counts the demo scene, exports the adapter as safetensors with a "
        "manifest and reloads it into a fresh pipeline to verify count and box parity. The default path needs no repository "
        "clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5). "
        "A CUDA runtime is used automatically when present (a Kaggle T4 finishes the model work in minutes); on CPU the path "
        "is about 20 minutes of model time on the build workstation, several times longer on a 2-vCPU hosted runtime."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to upload one zip "
        "(or also set `BYOD_ZIP_PATH` to a zip already in the runtime, which skips the upload dialog) "
        "holding a `labels.csv` (columns `id`, `file`, `label`, `count`; optional `points`, `exemplars` and `boxes` as JSON "
        "lists) beside the image files — at least eight images. Your records replace the synthetic scenes and pass through the "
        "same validation, seeded split, baselines, fine-tune, held-out evaluation, artifact export and reload-parity cells; "
        "the FSC-147 check is skipped. Records with `boxes` train on their object extents and are scored by box IoU; records "
        "with points only train on upstream's point boxes and are scored by point localisation. The schema and the ceilings "
        "are stated in the Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is optional and "
        "never part of the default path."
    ),
    "intro": (
        "CountGD (Amini-Naieni, Han and Zisserman, NeurIPS 2024) counts the objects in an image that a prompt describes: a "
        "**text prompt** naming the category, up to three **exemplar boxes** drawn around single instances, or both. It is "
        "GroundingDINO's open-set detector — a Swin-B image backbone, a BERT text encoder, a feature-enhancer that fuses "
        "them, and a six-layer decoder with 900 queries — extended so that each exemplar box is pooled from the image "
        "features and entered as an extra prompt token beside the text. Every query predicts a box and a similarity to each "
        "prompt token; the queries whose best score exceeds a threshold (0.23, upstream's) are the counted objects, so the "
        "model returns a **count, one box and one point per counted object** — 233,362,816 parameters, fine-tuned on "
        "FSC-147, published by its authors under the **MIT** licence. The score is a sigmoid similarity, uncalibrated: the "
        "count is a threshold decision, not a probability.\n\n"
        "This notebook reads the model three ways. The **count error** (MAE and RMSE — FSC-147's own metrics) says how many; "
        "**point localisation** (predicted points matched one-to-one to gold points within a radius) says whether it counted "
        "the right things; and **box IoU** (predicted boxes matched one-to-one to gold object boxes at IoU ≥ 0.5) says "
        "whether the boxes enclose them. FSC-147 annotates one point per object and three exemplar boxes per image, so it can "
        "score the first two but not the third; the tutorial therefore draws **synthetic counting scenes with known object "
        "boxes** — targets of one colour and shape among distractors of another — which score all three and are the data the "
        "fine-tune trains on, and keeps 24 real FSC-147 test photographs (MIT-licensed mirror; categories CountGD never saw "
        "in training) as the check that the fine-tune did not break counting on real images.\n\n"
        "What this notebook adds to inference is a **bounded counting fine-tune**: the last two decoder layers, the decoder's "
        "final norm and the shared box head — 3,619,584 of 233,362,816 parameters — trained on upstream's own objective "
        "(token focal loss and L1 box loss after Hungarian matching) with the gold object boxes as targets, the epoch chosen "
        "on the validation scenes' count error. The honest question is narrow: does a few minutes of that move the count "
        "error and the boxes on held-out scenes, against a **mean-count baseline** and a **template matcher** built from the "
        "same exemplars, and does it leave the FSC-147 photographs alone? The build record's answer is in Sections 6 and 8. "
        "Nothing here is a quality claim about your images: it is one seeded draw of synthetic scenes and 24 photographs.\n\n"
        "**Snapshot note:** the authors distribute the paper checkpoint as a 1.25 GB PyTorch pickle "
        "(`checkpoint_best_regular.pth` in their Space, byte-identical to the Google Drive file their README links). A pickle "
        "can execute code when it is opened, so the carried package audits it **statically** first — every global its "
        "`data.pkl` would import is listed without executing anything and must be one of five allowed names (four torch "
        "tensor builders and the `argparse.Namespace` of the training arguments) — then opens it once with torch's restricted "
        "unpickler, keeps the `model` state dict and writes `countgd.safetensors`, whose SHA-256 is pinned. Only that file is "
        "loaded. The GroundingDINO network is carried as vendored code with the multi-scale deformable attention in pure "
        "PyTorch (`grid_sample`), so no compiled CUDA extension is needed on CPU or GPU."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable "
        "upstream snapshot and see a pickle checkpoint audited statically and converted into safetensors before anything "
        "loads it; draw seeded synthetic counting scenes with known boxes and fetch a digest-pinned set of FSC-147 "
        "photographs, validate both and split them without leakage; count one scene by text, by exemplar boxes and by both "
        "through the public API and read what each prompt mode counts; read a count, a set of points and a set of boxes "
        "correctly (uncalibrated scores, a threshold decision); measure the frozen model's count error, point localisation "
        "and box IoU beside a mean-count baseline and a template matcher; run a bounded counting fine-tune with explicit "
        "hyperparameters and validation-based epoch selection; evaluate on held-out scenes and on real photographs; and "
        "export a safetensors adapter that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "object detection or segmentation as a product (the boxes are a reading of what was counted, not a detector to "
        "ship), counting by density map, video or tracking, dense scenes beyond one 800-pixel pass (FSC-147 images reach "
        "3,701 objects; the upstream test-time cropping for those is not carried), the upstream SAM-based test-time "
        "normalisation, calibrated confidence, full fine-tuning or training from scratch, evaluation on the full FSC-147 "
        "benchmark (only 24 of its test photographs are scored here), and any claim that coloured shapes stand in for your "
        "images. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab, Kaggle or Jupyter, Python 3.12). CUDA is used automatically when available; the default path also runs on CPU (float32). The build record measured 4.6 s per image to count on the build workstation's CPU and 4.7 s per fine-tuning step, so the whole default path is about 20 minutes of model time there after the downloads (a 2-vCPU hosted runtime will be several times slower); a hosted T4 finishes it in minutes. The pinned `torch==2.14.0` install and the 1.25 GB checkpoint are the large downloads; the conversion writes a further 938 MB, so allow about 3 GB of disk.",
        "- **Knowledge:** basic Python and PIL; what a bounding box and intersection-over-union are; what MAE and RMSE measure; why a score above a threshold is a decision and not a probability; what fine-tuning the last layers of a network changes and what it leaves alone.",
        "- **Data contract:** records are `{id, image, label, count}` with optional `points` (`[x, y]`, one per counted object), `exemplars` (0..3 boxes `[x0, y0, x1, y1]` around single instances) and `boxes` (one full object box per counted object) in image pixels. Images are PIL images or files decodable by Pillow with sides between `MIN_IMAGE_SIDE` (32) and `MAX_IMAGE_SIDE` (4096) px; the label is the category to count, at most 64 plain characters; `count` agrees with the points and the boxes when they are given (box centres stand in for missing points). Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a dataset needs 4..20,000 records. Every image is resized to a shortest side of 800 px (longest at most 1333) like upstream's test transform. Evaluation needs a gold count; adaptation needs points or boxes. BYOD accepts one zip of images plus a `labels.csv` in that shape.",
        "- **Validation is structural, not semantic:** every image is decoded and every annotation checked against the image and against the count, but nothing checks that a label names what the boxes enclose — a mislabelled set is counted without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.",
        "- **External access (data):** besides the model and tokenizer snapshots, the default path fetches 80 JPEG files from the Hugging Face dataset mirror `isentropic/FSC147` at revision `3e420cb6537e…` (about 2.7 MB in total), each pinned by byte size and SHA-256 in the carried `samples.py` and refused on any mismatch. The mirror is published under MIT; FSC-147's authors collected the images from the web and state no per-image licence. The synthetic scenes are drawn in the kernel. Nothing is committed to the repository.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Synthetic scenes, FSC-147 photographs and splits\n\n"
                "`build_synthetic_dataset` draws the seeded synthetic splits — scene `i` of a split is `synthetic_scene(base + i)` "
                "with disjoint seed ranges per split, 24 / 8 / 12 scenes of 512×384 pixels, each holding 6..40 **targets** of one "
                "colour and shape (the label, e.g. `blue circle`) and 4..20 **distractors** of another colour *and* shape, with one "
                "gold box per target and three of the targets' boxes as exemplars. `fetch_corpus` returns the 80 pinned FSC-147 "
                "photographs from the cache under `weights/fsc147-subset/` or the Hub mirror — every cached file re-hashed, every "
                "fetched file refused on any byte-size or SHA-256 mismatch — `read_corpus` decodes them into records with their "
                "gold points and three exemplar boxes, and `build_sample_dataset` draws a seeded stratified split per category (5 / "
                "2 / 3 of 10 → 40 / 16 / 24); only its 24 test photographs are scored in this notebook. `validate_dataset` checks "
                "every record against the contract, `check_split_disjoint` asserts no image (by decoded-pixel digest) is shared "
                "between splits, and the training scenes' labels table is written to `outputs/{stem}_train.csv` in the shape BYOD "
                "expects.\n\n"
                "Look for: 44 synthetic scenes with their gold-count range, 80 photographs over eight categories, one digest per "
                "split, and four refusal probes — a duplicate id, an image over the side ceiling, a count that disagrees with its "
                "boxes, and a box outside its image — each rejected before the model does anything."
            ),
            "code": (
                "import json\n"
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_ZIP_PATH = ''  # @param {{type:\"string\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD and BYOD_ZIP_PATH:\n"
                "    byod_zip = Path(BYOD_ZIP_PATH)  # a location field: no upload dialog (NOTEBOOK_SPEC 2.1 EXE2)\n"
                "    file_name = byod_zip.name\n"
                "elif USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_zip = Path('work') / 'byod.zip'\n"
                "    byod_zip.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_zip.write_bytes(payload)\n"
                "if USE_BYOD:\n"
                "    records = load_byod_dataset(byod_zip)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    fsc_test = []\n"
                "else:\n"
                "    splits = build_synthetic_dataset(SYNTHETIC_SPLIT)\n"
                "    data_source = f'synthetic counting scenes (seeds {{SYNTHETIC_SEED_BASE}}, sizes {{SYNTHETIC_SPLIT}}, {{SYNTHETIC_SIZE[0]}}x{{SYNTHETIC_SIZE[1]}} px)'\n"
                "    corpus_files = fetch_corpus(cache_dir='weights/fsc147-subset')\n"
                "    corpus = read_corpus(corpus_files)\n"
                "    fsc_splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    fsc_manifest = validate_dataset(fsc_splits['test'])\n"
                "    fsc_test = fsc_manifest['records']\n"
                "    print({{'fsc147': {{'photographs': len(corpus), 'bytes': sum(len(v) for v in corpus_files.values()), 'categories': sorted({{r['label'] for r in corpus}}), 'splits': check_split_disjoint(fsc_splits), 'test_gold_count': fsc_manifest['gold_count'], 'test_digest': fsc_manifest['digest'][:16] + '...'}}}})\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'splits': disjoint}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'labels': len(manifest['classes']), 'gold_count': manifest['gold_count'], 'with_boxes': manifest['with_boxes'], 'digest': manifest['digest'][:16] + '...'}}}})\n\n"
                "example = train_records[0]\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:4]],\n"
                "    'image over the side ceiling': [{{**example, 'image': Image.new('RGB', (MAX_IMAGE_SIDE + 1, 64)), 'boxes': None, 'points': None, 'exemplars': None, 'count': 0}}],\n"
                "    'count disagrees with its boxes': [{{**example, 'count': example['count'] + 1, 'points': None}}],\n"
                "    'box outside its image': [{{**example, 'boxes': [[0, 0, 9999, 10]] + example['boxes'][1:], 'points': None}}],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe, min_records=1)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Count one scene three ways through the inference contract\n\n"
                "The demo scene (`synthetic_scene(DEMO_SEED)`) holds 35 blue circles among 16 green triangles. `validate_inputs` "
                "applies exactly the checks the public operations apply — image decoding and sides, the prompt's characters and "
                "length, the exemplars' count, extent and placement — and returns an input manifest, here with one entry per prompt mode; a remote URL is validated too "
                "and its rejection recorded as a finding. `count` then runs the scene three times: with the **text** `blue circle` "
                "alone, with the three **exemplar boxes** alone (the caption is then the neutral word `object` with the exemplar "
                "tokens attached), and with **both**. Each result carries the count, one box and one point per counted object, "
                "their scores and the highest score of the 900 queries. The predicted boxes are drawn onto the scene and saved as "
                "PNG. The per-call `evaluation_report` against the scene's gold count is `sample-sanity` — plumbing evidence, not a "
                "measurement; Section 6 measures.\n\n"
                "Look for what each prompt counts. The build record counted 35, 51 and 35: text and text-plus-exemplars count the "
                "circles, while three exemplar boxes alone count 51 — every drawn shape, circles and triangles alike, because "
                "three boxes around small flat blobs describe *a small flat blob* at least as well as they describe *a blue "
                "circle*. That is the exemplar mode's known weakness on look-alike distractors, and the reason the fine-tune "
                "below trains with both prompts on scenes full of distractors."
            ),
            "code": (
                "demo = synthetic_scene(DEMO_SEED)\n"
                "print({{'ceilings': {{'MAX_EXEMPLARS': MAX_EXEMPLARS, 'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_BATCH': MAX_BATCH, 'MAX_COUNT': MAX_COUNT, 'SHORT_SIDE': SHORT_SIDE, 'MAX_SIDE': MAX_SIDE, 'threshold': CONFIDENCE_THRESHOLD, 'device': str(pipe.device)}}}})\n"
                "PROMPTS = {{'text': {{'text': demo['label']}}, 'exemplars': {{'exemplars': demo['exemplars']}}, 'both': {{'text': demo['label'], 'exemplars': demo['exemplars']}}}}\n"
                "input_manifest = validate_inputs(demo['image'], text=demo['label'], exemplars=demo['exemplars'], names=[demo['id']])\n"
                "input_manifest['prompt_modes'] = {{mode: validate_inputs(demo['image'], **kwargs)['inputs'][0] for mode, kwargs in PROMPTS.items()}}\n"
                "try:\n"
                "    validate_inputs('https://example.invalid/not-allowed.png', text='cells')\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'remote-url-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'scene': demo['id'], 'label': demo['label'], 'gold': demo['count'], 'distractors': demo['distractors'], 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n\n\n"
                "def draw_counts(image, entry, colour=(255, 0, 255)):\n"
                "    canvas = image.copy()\n"
                "    pen = ImageDraw.Draw(canvas)\n"
                "    for box in entry['boxes']:\n"
                "        pen.rectangle(box, outline=colour, width=2)\n"
                "    for x0, y0, x1, y1 in entry['exemplars']:\n"
                "        pen.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=3)\n"
                "    return canvas\n\n\n"
                "t0 = time.perf_counter()\n"
                "demo_frozen = {{mode: pipe.count(demo['image'], **kwargs)['results'][0] for mode, kwargs in PROMPTS.items()}}\n"
                "again = pipe.count(demo['image'], **PROMPTS['both'])['results'][0]\n"
                "for mode, entry in demo_frozen.items():\n"
                "    draw_counts(demo['image'], entry).save(f'outputs/{stem}_demo_frozen_{{mode}}.png')\n"
                "    print(mode, '->', {{'count': entry['count'], 'gold': demo['count'], 'max_score': round(entry['max_score'], 3), 'box_f1': round(box_metrics([match_boxes(entry['boxes'], demo['boxes'])])['f1'], 3)}})\n"
                "w, h = demo['image'].size\n"
                "checks = {{\n"
                "    'one_box_and_point_per_count': all(len(e['boxes']) == len(e['points']) == e['count'] for e in demo_frozen.values()),\n"
                "    'scores_above_threshold': all(min(e['scores'], default=1.0) > CONFIDENCE_THRESHOLD for e in demo_frozen.values()),\n"
                "    'points_inside_the_image': all(0 <= x <= w and 0 <= y <= h for e in demo_frozen.values() for x, y in e['points']),\n"
                "    'same_input_same_output': again['count'] == demo_frozen['both']['count'] and again['boxes'] == demo_frozen['both']['boxes'],\n"
                "    'exemplars_echoed_in_input_pixels': demo_frozen['exemplars']['exemplars'] == [[float(v) for v in b] for b in demo['exemplars']],\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'inference output failed a sanity check: {{checks}}')\n"
                "demo_report = evaluation_report({{'results': list(demo_frozen.values())}}, [demo['count']] * 3, sample_kind='synthetic')\n"
                "print({{'checks': checks, 'seconds': round(time.perf_counter() - t0, 2), 'report': {{m['id']: m['value'] for m in demo_report['metrics']}}, 'verdict': demo_report['verdict']}})"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model on held-out scenes and photographs\n\n"
                "`pipe.evaluate` counts every record with its own label and its own exemplars (both prompts, as the fine-tune "
                "will train) and reports the **count error** (MAE, RMSE, the normalised absolute error, the fraction counted "
                "exactly), the **point localisation** (precision, recall and F1 of the predicted points matched one-to-one to the "
                "gold points within half the mean exemplar side) and, where gold object boxes exist, the **box IoU** reading "
                "(precision, recall and F1 of the predicted boxes matched one-to-one to the gold boxes at IoU ≥ 0.5, the mean IoU "
                "of the matched pairs, and the mean over gold boxes of the best IoU any prediction reaches). Two non-neural "
                "baselines are scored by the same code on the same records: the **mean-count baseline** answers every image with "
                "the rounded mean gold count of the training records (the counting analogue of a majority floor), and the "
                "**template matcher** correlates the mean of the image's own exemplar crops over the image and counts the peaks "
                "above 0.6 — a classical, learning-free use of the same three boxes the model gets.\n\n"
                "Two sets, read separately. On the **12 held-out synthetic scenes** the build record measured the frozen model "
                "at MAE 7.42 (RMSE 9.97), box F1 0.453 and point F1 0.862, "
                "against MAE 10.00 for the mean count and 14.50 for the template matcher — the distractors are "
                "where its count error comes from. On the **24 FSC-147 test photographs** (categories it never saw in training) it "
                "measured MAE 4.54 against 16.21 and 34.00. Read the per-image rows: a few dense "
                "images carry most of an RMSE."
            ),
            "code": (
                "CK = ('mae', 'rmse', 'nae', 'exact_fraction')\n\n\n"
                "def brief(result):\n"
                "    row = {{k: round(result[k], 3) for k in CK}}\n"
                "    row['point_f1'] = None if result['localisation']['f1'] is None else round(result['localisation']['f1'], 3)\n"
                "    boxes = result.get('boxes') or {{}}\n"
                "    if boxes.get('f1') is not None:\n"
                "        row.update({{'box_f1': round(boxes['f1'], 3), 'box_mean_matched_iou': round(boxes['mean_matched_iou'] or 0.0, 3), 'box_mean_best_iou': round(boxes['mean_best_iou'], 3)}})\n"
                "    return row\n\n\n"
                "t0 = time.perf_counter()\n"
                "frozen_syn = pipe.evaluate(test_records)\n"
                "frozen_fsc = pipe.evaluate(fsc_test) if fsc_test else None\n"
                "frozen_seconds = round(time.perf_counter() - t0, 1)\n"
                "baselines = {{'synthetic': {{'mean_count': mean_count_baseline(train_records, test_records), 'template_matching': template_matching_baseline(test_records)}}}}\n"
                "if fsc_test:\n"
                "    baselines['fsc147'] = {{'mean_count': mean_count_baseline(fsc_splits['train'], fsc_test), 'template_matching': template_matching_baseline(fsc_test)}}\n"
                "print({{'frozen_synthetic_test': brief(frozen_syn), 'n': frozen_syn['n'], 'verdict': frozen_syn['verdict'], 'seconds': frozen_seconds}})\n"
                "if frozen_fsc:\n"
                "    print({{'frozen_fsc147_test': brief(frozen_fsc), 'n': frozen_fsc['n'], 'verdict': frozen_fsc['verdict']}})\n"
                "for dataset, rows in baselines.items():\n"
                "    for name, base in rows.items():\n"
                "        print({{dataset + '/' + name: {{k: round(base[k], 3) for k in CK}}, 'note': base['baseline']}})\n"
                "print({{'definitions': {{**frozen_syn['definitions'], **frozen_syn['boxes']['definitions']}}}})\n"
                "for row in frozen_syn['per_image'][:6]:\n"
                "    print({{k: row[k] for k in ('id', 'label', 'gold', 'predicted')}}, {{'boxes': row['boxes']}})\n"
                "assert frozen_syn['boxes']['n'] == len(test_records) and frozen_syn['localisation']['n'] == len(test_records)"
            ),
        },
        {
            "md": (
                "## 7. Bounded counting fine-tune\n\n"
                "`pipe.adapt` continues CountGD's own training on the training scenes: the last `TRAINABLE_LAYERS` decoder "
                "layers, the decoder's final LayerNorm and the shared box head train — two layers by default, 3,619,584 of "
                "233,362,816 parameters; the Swin-B backbone, BERT, the feature enhancer, the encoder and the first four decoder "
                "layers stay frozen — on upstream's objective: the token-level sigmoid focal loss and the L1 box loss after "
                "Hungarian matching, over the final and every intermediate decoder output, with the caption `<label> .`, the "
                "scene's three exemplars and the **gold object boxes** as targets (FSC-147 records, which have points only, would "
                "train on upstream's 2×2-pixel point boxes instead). One scene per step, AdamW at a fixed learning rate with weight "
                "decay 1e-4, gradient clipping at 0.1 (upstream's), seeded order, no scheduler, no augmentation. Epoch 0 records "
                "the frozen model's validation count error; every epoch is scored on the eight validation scenes and the epoch "
                "with the lowest validation MAE is kept — so the selection can return the frozen model itself (epoch 0) when "
                "nothing beats it, and never returns a worse one. The validation rows also print point and box F1: they are "
                "read, not selected on.\n\n"
                "The build record's recipe sweep (`docs/release-verification.md`) found `1e-5` too small to move anything in "
                "a few epochs, `1e-4` for three epochs on twelve scenes moved the test MAE only from 7.25 to 5.62, and `2e-4` for four epochs on "
                "the 24 scenes moved the validation MAE from 9.38 to 0.75 at epoch 3 before it rose again at epoch 4 (2.75) — "
                "so the selector kept epoch 3. The default below is that configuration; expect the validation MAE to fall and "
                "then turn, and the kept epoch to be the lowest one, not the last."
            ),
            "code": (
                "EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 2e-4  # @param {{type:\"number\"}}\n"
                "TRAINABLE_LAYERS = 2  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 3)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: None if entry['val'][k] is None else round(entry['val'][k], 3) for k in ('mae', 'rmse', 'localisation_f1', 'box_f1')}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, trainable_layers=TRAINABLE_LAYERS, seed=0, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'objective': adapt_result['objective'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seconds': adapt_seconds}})\n"
                "val_history = {{h['epoch']: h['val']['mae'] for h in adapt_result['history']}}\n"
                "assert val_history[adapt_result['best_epoch']] <= val_history[0]  # the selector never returns an epoch worse than the frozen model"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation after the fine-tune\n\n"
                "The test scenes and the FSC-147 photographs were never used for training or epoch selection, and no image appears "
                "in two splits. The adapted model is scored exactly as the frozen model was in Section 6 — same records, same "
                "prompts, same threshold — and the systems are put side by side: on the synthetic test the mean count, the "
                "template matcher, the frozen model and the adapted one on count error, point F1 and box IoU; on FSC-147 the same "
                "on count error and point F1. The evaluation report is written as JSON. Read it in this order: the **validation "
                "MAE** that selected the epoch, then the **synthetic test** (the build record measured MAE 7.42 → "
                "0.92 and box F1 0.453 → 0.550), then **FSC-147** (MAE "
                "4.54 → 3.50) — the check that training on coloured shapes did not break counting on "
                "photographs. Twelve scenes and 24 photographs from one seeded draw give **no dispersion estimate**, a count error "
                "moves in steps of one object per image, and the synthetic scenes are easy in ways real images are not: this is "
                "sample-sanity evidence that the adaptation contract works, not a benchmark, and not a claim about your images "
                "until you measure them."
            ),
            "code": (
                "adapted_syn = pipe.evaluate(test_records)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "adapted_fsc = pipe.evaluate(fsc_test) if fsc_test else None\n\n\n"
                "def side_by_side(dataset, frozen, adapted):\n"
                "    table = {{name: {{k: round(base[k], 3) for k in CK}} for name, base in baselines[dataset].items()}}\n"
                "    table.update({{'frozen': brief(frozen), 'adapted': brief(adapted)}})\n"
                "    return table\n\n\n"
                "comparison = {{\n"
                "    'synthetic_test': side_by_side('synthetic', frozen_syn, adapted_syn),\n"
                "    'synthetic_delta_vs_frozen': {{k: round(v - brief(frozen_syn)[k], 3) for k, v in brief(adapted_syn).items() if v is not None and brief(frozen_syn).get(k) is not None}},\n"
                "    'validation_mae': {{'frozen': round(val_history[0], 3), 'selected_epoch': adapt_result['best_epoch'], 'selected': round(val_history[adapt_result['best_epoch']], 3), 'rescored': round(adapted_val['mae'], 3)}},\n"
                "}}\n"
                "if fsc_test:\n"
                "    comparison['fsc147_test'] = side_by_side('fsc147', frozen_fsc, adapted_fsc)\n"
                "    comparison['fsc147_delta_vs_frozen'] = {{k: round(v - brief(frozen_fsc)[k], 3) for k, v in brief(adapted_fsc).items() if v is not None and brief(frozen_fsc).get(k) is not None}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "strip = lambda result: None if result is None else {{k: v for k, v in result.items() if k != 'definitions'}}  # noqa: E731\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': DEFAULT_MODEL_KEY, 'weight_sha256': MODEL_SHA256}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'synthetic': {{'frozen_test': strip(frozen_syn), 'adapted_test': strip(adapted_syn), 'adapted_validation': strip(adapted_val), 'baselines': baselines['synthetic']}},\n"
                "    'fsc147': {{'frozen_test': strip(frozen_fsc), 'adapted_test': strip(adapted_fsc), 'baselines': baselines.get('fsc147')}},\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False, default=str)\n"
                "assert abs(adapted_val['mae'] - val_history[adapt_result['best_epoch']]) < 1e-6  # the kept epoch re-scores to the number that selected it\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Re-count the demo scene, export the adapter and reload it\n\n"
                "The demo scene from Section 5 is a *training-distribution* scene the fine-tune never saw (its seed lies outside "
                "every split's range); it is counted again by the adapted model with the same three prompts and drawn beside the "
                "frozen result. The exemplar-only count is the number to watch: the fine-tune trained with both prompts on scenes "
                "full of distractors, and in the build record the exemplar-only count moved from 51 to 35 — the circles only — "
                "while the text and text-plus-exemplar counts stayed at 35. A different count here is a finding to record, not a "
                "failure.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the last two decoder layers, the decoder norm and the shared "
                "box head, about 14.5 MB — as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the base "
                "model id and revision, the digest of the base `countgd.safetensors`, the objective, the tensor names, the file "
                "size and SHA-256, the training configuration and the epoch history (OUT8). `CountGDPipeline.from_artifact` "
                "re-verifies the base snapshot, checks the artifact manifest, its digest and its exact tensor set **before** "
                "deserialising, and overlays the tensors onto a freshly loaded base — a new object from files, not the in-memory "
                "model (VER2). The cell asserts identical counts and boxes on four test scenes (VER4)."
            ),
            "code": (
                "import shutil\n\n"
                "demo_adapted = {{mode: pipe.count(demo['image'], **kwargs)['results'][0] for mode, kwargs in PROMPTS.items()}}\n"
                "demo_rows = []\n"
                "for mode in PROMPTS:\n"
                "    before, after = demo_frozen[mode], demo_adapted[mode]\n"
                "    draw_counts(demo['image'], after).save(f'outputs/{stem}_demo_adapted_{{mode}}.png')\n"
                "    row = {{'prompt': mode, 'gold': demo['count'], 'frozen': before['count'], 'adapted': after['count'], 'frozen_box_f1': round(box_metrics([match_boxes(before['boxes'], demo['boxes'])])['f1'], 3), 'adapted_box_f1': round(box_metrics([match_boxes(after['boxes'], demo['boxes'])])['f1'], 3)}}\n"
                "    demo_rows.append(row)\n"
                "    print(row)\n"
                "demo_adapted_report = evaluation_report({{'results': list(demo_adapted.values())}}, [demo['count']] * 3, sample_kind='synthetic')\n"
                "with open('outputs/{stem}_demo.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump({{'scene': demo['id'], 'label': demo['label'], 'gold': demo['count'], 'distractors': demo['distractors'], 'rows': demo_rows, 'frozen_report': demo_report, 'adapted_report': demo_adapted_report}}, handle, indent=2)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = CountGDPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, tokenizer_dir=TOKENIZER_WEIGHTS_DIR, device=pipe.device)\n"
                "parity_records = test_records[:4]\n"
                "before = [pipe.count(r['image'], text=r['label'], exemplars=r['exemplars'])['results'][0] for r in parity_records]\n"
                "after = [reloaded.count(r['image'], text=r['label'], exemplars=r['exemplars'])['results'][0] for r in parity_records]\n"
                "parity = {{\n"
                "    'identical_counts': int(sum(a['count'] == b['count'] for a, b in zip(before, after, strict=True))),\n"
                "    'max_abs_box_difference': float(max((abs(x - y) for a, b in zip(before, after, strict=True) for p, q in zip(a['boxes'], b['boxes'], strict=True) for x, y in zip(p, q, strict=True)), default=0.0)),\n"
                "    'max_abs_score_difference': float(max((abs(x - y) for a, b in zip(before, after, strict=True) for x, y in zip(a['scores'], b['scores'], strict=True)), default=0.0)),\n"
                "    'of': len(parity_records),\n"
                "}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_counts'] == parity['of'] and parity['max_abs_box_difference'] < 0.05 and parity['max_abs_score_difference'] < 1e-4\n\n"
                "write_provenance('outputs/provenance.json', pipeline=pipe)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'fetched_this_run': fetched, 'source_file': SOURCE_CKPT_NAME, 'source_sha256': SOURCE_CKPT_SHA256, 'pickle_audit_sha256': PICKLE_AUDIT_SHA256, 'weight_file': MODEL_FILENAME, 'weight_format': 'safetensors, converted once from the audited pickle, digest-verified', 'weight_sha256': MODEL_SHA256}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'license': CORPUS_LICENSE, 'base_url': CORPUS_BASE_URL, 'bytes': CORPUS_BYTES, 'pinned_photographs': len(SAMPLE_RECORDS), 'scored_photographs': len(fsc_test)}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'demo': demo_rows}},\n"
                "    'comparison': comparison,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'numpy': numpy.__version__, 'device': str(pipe.device), 'dtype': 'float32', 'checkpoint_source': pipe.checkpoint_source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False, default=str)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen model counts what a prompt names on scenes and photographs it never trained on — MAE 7.42 "
        "on the held-out synthetic scenes and 4.54 on 24 FSC-147 photographs in the build record, beside "
        "10.00 / 16.21 for the mean count and 14.50 / 34.00 for the template matcher — "
        "and its boxes enclose the objects it counts (box F1 0.453 at IoU ≥ 0.5 on the scenes). What the three "
        "prompt modes on the demo scene show is where it errs: exemplar boxes alone describe *what an object looks like* and "
        "count look-alike distractors; the text is what separates a blue circle from a green triangle. A bounded fine-tune of "
        "3.6 M of its 233 M parameters on 24 scenes with distractors, chosen on the validation count error, moves the "
        "held-out synthetic MAE from 7.42 to 0.92 and the FSC-147 MAE from 4.54 "
        "to 3.50, and exports a 14.5 MB adapter that reloads to identical counts. That is the claim: the "
        "adaptation contract works end to end, the selector keeps the frozen model when nothing beats it, and the numbers are "
        "read on count, points and boxes against non-neural baselines and the frozen model rather than in isolation.\n\n"
        "The synthetic scenes are a teaching instrument, not a domain: flat colours, one scale per scene, no occlusion, "
        "no texture. They were chosen because they carry object boxes FSC-147 does not, and because their distractors make "
        "the prompt modes' behaviour visible. Where a fine-tune earns its place is a real domain with a counting failure you "
        "can name — cells, cars, seeds, colonies — with boxes or points for a few dozen images; that is what BYOD is for. "
        "The test sets are 12 scenes and 24 photographs from one seeded draw, the validation set that picks the epoch is 8 "
        "scenes, a count error moves in steps of one object, and the scores are uncalibrated similarities read against a "
        "fixed threshold.\n\n"
        "Three things to carry to real data. **Baselines first:** the mean count and the template matcher on *your* images "
        "are the numbers to read before any adapted one. **Watch the other set:** a fine-tune that helps its own domain can "
        "hurt another; keep a held-out set from the original distribution, as FSC-147 is kept here. **Dense images need "
        "cropping:** a single 800-pixel pass caps what 900 queries can count; upstream tiles images with many small objects, "
        "which this repository does not carry.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, can "
        "acquire and digest-verify the pinned snapshots, audit and convert a pickle checkpoint into a digest-pinned "
        "safetensors file, fetch and digest-verify a real photograph set, validate the demonstrated dataset contract without "
        "leakage, execute the inference contract and a bounded counting fine-tune, evaluate against non-neural baselines and "
        "the frozen model on held-out scenes and photographs, and emit the shown machine-readable artifacts — without the "
        "repository being reachable. It does **not** establish benchmark superiority, performance on the FSC-147 benchmark, "
        "counting quality on any other population or camera, or production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** count the demo scene with `threshold=0.35` and "
        "watch the exemplar-only count fall; set `TRAINABLE_LAYERS = 0` to train the box head alone; fine-tune on "
        "`fsc_splits['train']` (points only, upstream's 2×2-pixel boxes) and read what happens to the synthetic box IoU; or "
        "bring your own images through BYOD and read the baselines before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/countgd-object-counting-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/countgd-object-counting-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance, the pickle audit and the conversion: https://github.com/kurtvalcorza/countgd-object-counting-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream Space (pinned checkpoint host): https://huggingface.co/spaces/{MODEL_ID}\n"
        "- Upstream code: https://github.com/niki-amini-naieni/CountGD (MIT; the GroundingDINO parts carry IDEA's Apache-2.0 header)\n"
        "- CountGD: Multi-Modal Open-World Counting (Amini-Naieni, Han and Zisserman, NeurIPS 2024): https://arxiv.org/abs/2407.04619\n"
        "- Grounding DINO (Liu et al., 2023): https://arxiv.org/abs/2303.05499\n"
        "- FSC-147 / Learning To Count Everything (Ranjan, Sharma, Nguyen and Hoai, CVPR 2021): https://github.com/cvlab-stonybrook/LearningToCountEverything — Hub mirror https://huggingface.co/datasets/isentropic/FSC147\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.2 (fleet specs in the ml-worker repository)"
    ),
}
