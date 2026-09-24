# Weight provenance, the pickle audit, the conversion, and the pinned data

This repository pins two model snapshots, each described by its own `dimer-base-manifest.json`, and one set of
FSC-147 photographs. The CountGD checkpoint is a pickle, which this pipeline audits and converts but never
serves.

## CountGD checkpoint

- Upstream code: `niki-amini-naieni/CountGD` at commit `b6f362b3f5cd20db4a171faa410dfed8f2f466d8` (MIT; the
  GroundingDINO parts carry IDEA's Apache-2.0 header). Its README links the paper checkpoint on Google Drive as
  `checkpoint_fsc147_best.pth` (1.2 GB).
- Pinned host: the authors' Hugging Face Space `nikigoli/countgd` at revision
  `6e82e59569a84ee5c6aafa35d396f2d2bee57be2`, file `checkpoint_best_regular.pth`.
- Identity: 1,250,122,522 bytes, SHA-256 `c1bab864b17db345b4c6e3aaabb5765bc2c0a90d0bc8defb5e664a74a50aa126`. On
  2026-09-24 both the Space file and the Google Drive file were downloaded and hashed: the two are byte-identical,
  and the digest equals the Space's LFS object id. The Space is pinned because a Hub revision is immutable and
  fetchable without an interactive confirmation page.
- Contents: a torch zip archive with one pickle member, `checkpoint_best_regular/data.pkl`, and 2,544 tensor
  storages. The unpickled payload is a dict with `model` (1,146 tensors, float32 and int64), `optimizer`,
  `lr_scheduler`, `epoch` (19) and `args` (the training arguments). Only `model` is read.
- Licence: MIT, per the upstream repository's `LICENSE` and README. The Space card declares no licence of its
  own.
- Local layout: `weights/countgd/` holds the manifest (the Space `README.md` and the checkpoint, with byte size
  and SHA-256 each; 1,250,123,048 bytes in total) and, once converted, `countgd.safetensors`. `verify_snapshot()`
  checks the manifest entries and asserts the checkpoint's digest against the package constants, which win over
  an edited manifest.

## What the pickle would execute, and how it is audited

A pickle is executable serialization. `audit_pickle()` in `src/countgd_pipeline/model.py` reads every `.pkl`
member of the torch zip archive with `pickletools.genops`, collects every `GLOBAL` / `STACK_GLOBAL` it would
import, and refuses any name outside the allow-list. Nothing is executed.

| File | Globals found | Allow-list | Audit SHA-256 |
|---|---|---|---|
| `checkpoint_best_regular.pth` | `argparse.Namespace`, `collections.OrderedDict`, `torch.FloatStorage`, `torch.LongStorage`, `torch._utils._rebuild_tensor_v2` | exactly those five | `4606eaf365d27d2fddd901bdc069218dabbd418f9856ed2d0e3707612ad4c527` |

The audit digest is the SHA-256 of the sorted global names joined by newlines (the same construction as the
Prithvi burn-scar pipeline's audit, whose four names produce `5b9f0ba0…`). It is pinned in
`PICKLE_AUDIT_SHA256`, and `convert_checkpoint()` refuses a file whose audit digest differs. `argparse.Namespace`
is the one class beyond torch's default restricted-unpickler set: it is the training-arguments object, and
constructing it sets attributes and nothing else. Tests craft a torch archive carrying `os.system` and a bare
pickle, and assert that the audit refuses each before anything is constructed.

## The conversion

`convert_checkpoint()` runs size check → SHA-256 check against the package constants → static audit and
audit-digest check, and only then:

1. `torch.load(map_location="cpu", weights_only=True)` inside `torch.serialization.safe_globals([argparse.Namespace])`
   — torch's restricted unpickler with that one class added — must return a dict whose `model` entry holds
   exactly 1,146 tensors;
2. the 38 `feature_map_encoder.*` tensors are dropped: they belong to an exemplar encoder that upstream builds
   but whose call is commented out of the forward pass, so the vendored network does not build it;
3. tensors that share storage are grouped — `dec_pred_bbox_embed_share=True` ties the six decoder box heads and
   the encoder's — and each group is stored once under its lexicographically first name, the other 66 names
   recorded as aliases;
4. the 1,042 stored tensors are written to `countgd.safetensors` with the aliases as **one** JSON metadata entry,
   `countgd_aliases`, and the result is checked against its pinned size and digest.

| File | Bytes | Tensors | SHA-256 | In Git |
|---|---|---|---|---|
| `countgd.safetensors` | 937,560,480 | 1,042 stored + 66 aliases = 1,108 state-dict entries | `8e44867b951e3a4205d918e022b78bc5fea218fd17c1851b864a01c421d2d443` | no (regenerated) |

**Why one metadata key.** The safetensors writer serializes a multi-key metadata map in hash-map order, which
changes from process to process: writing the same 40-key map three times gave three different file digests,
while a one-key map gave the same digest every time. With the 66 aliases in a single sorted JSON value, the
conversion is deterministic; it produced the same digest in two separate processes on the build workstation and
again inside the tutorial notebook, which converts the file it downloads.

**Cross-check against the authors' own export.** The authors also publish a safetensors export on the Hub,
`nikigoli/CountGD` at revision `6b989ad11db9413e44863ea5c57735cae6cacff8` (`model.safetensors`, 937,560,192
bytes). It holds the same 1,042 tensors and 66 aliases: every tensor is bit-equal (`torch.equal`, same dtype
and shape) to the pickle's `model` entry, and the tensor data section of the two files is byte-identical
(SHA-256 `0e2995b7e8af396a…`). The files differ only in the header, where the export stores the aliases as 66
separate metadata keys in one hash order. The pipeline pins the pickle, the source the upstream README names,
and not the export.

`load_state_dict()` refuses a safetensors file whose metadata is anything other than the one `countgd_aliases`
entry, restores the aliases, and `load_components()` loads the 1,108 entries into the vendored network with
`strict=True`: 0 missing, 0 unexpected, 0 shape mismatches.

## The vendored network

`src/countgd_pipeline/modeling.py` carries the named top-level definitions of the upstream model files at
commit `b6f362b3`, in dependency order, with every edit marked `# vendored:`. The edits that matter for the
weights: the compiled `MultiScaleDeformableAttention` CUDA op is not carried — the upstream pure-PyTorch
`grid_sample` path runs on CPU and CUDA alike, so no custom build is needed — and the never-called
`feature_map_encoder` is not built. The built model has 233,362,816 parameters.

## The tokenizer snapshot

`weights/bert-base-uncased/` pins `google-bert/bert-base-uncased` at revision
`86b5e0934494bd15c9632b12f734a8a67f723594` (Apache-2.0): `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `vocab.txt`, `LICENSE` and `README.md`, each with byte size and SHA-256 in its
manifest. BERT is constructed from `config.json` at random initialisation and then overwritten by the 199
`bert.*` tensors of the CountGD checkpoint; no BERT weight file is fetched.

## The pinned photographs

`samples.py` pins 80 photographs of the FSC-147 test split (Ranjan, Sharma, Nguyen and Hoai, CVPR 2021; eight
categories, ten images each, gold counts 8..104) served by the Hub mirror `isentropic/FSC147` at revision
`3e420cb6537e803dd6d4516623ce82a79c0317b8` (`images_384_VarV2/`), each by byte size and SHA-256 (2,732,222 bytes
in total), with FSC-147's point annotation and three exemplar boxes per image. Files are fetched at run time into
`weights/fsc147-subset/` (git-ignored) and refused on any mismatch. The FSC-147 splits are category-disjoint, so
CountGD never saw these categories in training. The mirror is published under MIT, as is the upstream
`LearningToCountEverything` repository; FSC-147's authors collected the images from the web and state no
per-image licence, which is why the repository redistributes none of the images.

## Files deliberately not staged

The Space also carries `checkpoints/groundingdino_swinb_cogcoor.pth` (the GroundingDINO initialisation the
authors trained from), a BERT `model.safetensors`, the Gradio app and prebuilt `MultiScaleDeformableAttention`
wheels. None of them is listed in the manifest or fetched: the CountGD checkpoint already carries every tensor
the model needs, and no compiled op is used.

## DIMER hosting

- MIT permits use, modification, redistribution and commercial use subject to preservation of the licence
  notice. DIMER may host the converted safetensors under those terms; it is derived from, and recorded beside,
  the unmodified upstream checkpoint.
- Upload set: `countgd.safetensors` (`python scripts/fetch_weights.py --zip countgd-dimer.zip`). **The `.pth`
  file must not be uploaded**: a profile that carries it would reintroduce the executable serialization the
  conversion removes.
- Loader trust boundary: no `trust_remote_code`, no Hub-hosted code, no pickle on the serving path. The model
  class is the vendored code in this repository and the served state dict is safetensors.
- One review item is open: whether the one-time restricted unpickle (in the build, and in the tutorial's
  runtime) meets the DIMER deserialization-trust bar, or whether DIMER hosts only maintainer-converted files.
  The served file is the same either way.
