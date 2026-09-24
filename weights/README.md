# Base Model Weights Cache

This directory holds offline base-model weights, configurations, and cryptographic manifests for the ViT-MAE pre-training pipeline, plus the git-ignored photograph cache the tutorial fills at run time.

The model is organized in its own isolated subfolder corresponding to its canonical identifier:

```
weights/
├── vit-mae-base/
│   ├── config.json
│   ├── preprocessor_config.json
│   ├── dimer-base-manifest.json
│   ├── README.md (upstream model card)
│   └── model.safetensors (Excluded from Git; acquired via scripts/fetch_weights.py or DIMER upload)
└── inat-birds/ (Excluded from Git; 360 CC0 iNaturalist photographs fetched by fetch_corpus())
```

## Available Base Model Snapshots

- [**`vit-mae-base`**](vit-mae-base/): Dedicated snapshot for Meta AI's ViT-MAE Base masked autoencoder (`facebook/vit-mae-base`, `ViTMAEForPreTraining`, encoder + decoder, 111,907,840 parameters).
  - [**Upstream model card**](vit-mae-base/README.md): the Hub card as hosted at the pinned revision (architecture, intended use, Apache-2.0 license terms).
  - [**Manifest**](vit-mae-base/dimer-base-manifest.json): Cryptographic record of byte counts and SHA-256 hashes for all four snapshot files.

## DIMER Architecture & Git Tracking Strategy

In the DIMER workbench ecosystem:
1. **Large Binary Weights (`model.safetensors`):** The ~448 MB (447,670,680 bytes) weight payload is excluded from Git via `.gitignore` (`weights/**/*.safetensors`) and uploaded directly to DIMER as a model asset or downloaded using `scripts/fetch_weights.py`.
2. **Configuration:** The accompanying configuration files (`config.json`, `preprocessor_config.json`), the upstream card and the cryptographic manifest are version-controlled in the repository so the pipeline and offline Docker containers can initialize the image processor without network dependencies.
3. **Photograph cache (`inat-birds/`):** filled by `fetch_corpus()` from the iNaturalist open-data bucket (about 39 MB, CC0 1.0, each file pinned by byte size and SHA-256 in `samples.py`); never committed.

## Management & Verification Tooling

Manage, download, and cryptographically verify model snapshots using [`scripts/fetch_weights.py`](../scripts/fetch_weights.py):

```bash
# Verify the existing snapshot in weights/vit-mae-base:
python scripts/fetch_weights.py --verify-only

# Download and verify default model into its dedicated subfolder:
python scripts/fetch_weights.py --dest weights/vit-mae-base

# Verify an explicit destination directory:
python scripts/fetch_weights.py --verify-only --dest weights/vit-mae-base
```
