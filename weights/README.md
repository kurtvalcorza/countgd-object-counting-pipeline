# Base Model Weights Cache

This directory holds the pinned snapshots of the CountGD pipeline and their cryptographic manifests, plus the
git-ignored files the pipeline and the tutorial fill at run time.

```
weights/
├── countgd/
│   ├── dimer-base-manifest.json      (committed: the authors' Space at the pinned revision, 2 files)
│   ├── README.md                     (committed: the Space card at the pinned revision)
│   ├── checkpoint_best_regular.pth   (NOT in Git: 1,250,122,522-byte pickle, fetched by scripts/fetch_weights.py)
│   └── countgd.safetensors           (NOT in Git: 937,560,480 bytes, converted once from the pickle)
├── bert-base-uncased/                (committed: vocabulary, tokenizer and architecture configuration; no weights)
│   ├── dimer-base-manifest.json
│   ├── config.json, tokenizer.json, tokenizer_config.json, vocab.txt, LICENSE, README.md
└── fsc147-subset/                    (NOT in Git: 80 FSC-147 test photographs fetched by fetch_corpus())
```

## Snapshots

- [**`countgd`**](countgd/): the authors' Hugging Face Space `nikigoli/countgd` at revision
  `6e82e59569a84ee5c6aafa35d396f2d2bee57be2`. Its manifest lists the Space card and
  `checkpoint_best_regular.pth`, the paper checkpoint as a PyTorch pickle (SHA-256
  `c1bab864b17db345b4c6e3aaabb5765bc2c0a90d0bc8defb5e664a74a50aa126`, byte-identical to the Google Drive file the
  upstream README links). The package audits that pickle statically, opens it once with torch's restricted
  unpickler and writes `countgd.safetensors` (SHA-256
  `8e44867b951e3a4205d918e022b78bc5fea218fd17c1851b864a01c421d2d443`), the only file it loads. See
  [`docs/WEIGHTS.md`](../docs/WEIGHTS.md).
- [**`bert-base-uncased`**](bert-base-uncased/): `google-bert/bert-base-uncased` at revision
  `86b5e0934494bd15c9632b12f734a8a67f723594` — the vocabulary, the tokenizer configuration and the architecture
  configuration CountGD's text encoder is built from. BERT's weights are not taken from here: they are inside the
  CountGD checkpoint (199 `bert.*` tensors).

## Git tracking

1. **Binary weights** (`*.pth`, `*.safetensors`) are excluded by `.gitignore` and obtained with
   `python scripts/fetch_weights.py`, which stages the manifest-listed files from the Space at the pinned
   revision, verifies them, and converts the pickle.
2. **Manifests, cards and tokenizer files** are committed, so the pipeline can build the tokenizer and verify
   the checkpoint offline once it is present.
3. **`fsc147-subset/`** is filled by `fetch_corpus()` from the Hub mirror `isentropic/FSC147` (80 files, about
   2.7 MB, each pinned by byte size and SHA-256 in `samples.py`); it is never committed.
4. `.gitattributes` carries `weights/** -text`, so a Windows checkout cannot rewrite a snapshot file's line
   endings and break its recorded digest.

## DIMER upload

`python scripts/fetch_weights.py --zip countgd-dimer.zip` writes an archive holding `countgd/countgd.safetensors`
only. The pickle is never packaged: an upload that carried it would put executable serialization back on the
serving path the conversion removes.
