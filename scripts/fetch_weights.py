#!/usr/bin/env python
"""Fetch, verify and convert the pinned CountGD checkpoint (a thin command line over the package).

The snapshot is the authors' Hugging Face Space `nikigoli/countgd` at an immutable revision; its manifest
(`weights/countgd/dimer-base-manifest.json`) lists the Space README and the 1.25 GB pickle checkpoint. The
pickle is statically audited and converted once into `countgd.safetensors`, the only file the package serves
and the only file a DIMER upload carries.

Usage:
    python scripts/fetch_weights.py                  # stage what is missing, verify, convert
    python scripts/fetch_weights.py --verify-only    # verify the source and the converted file, fetch nothing
    python scripts/fetch_weights.py --dry-run        # list what would be fetched
    python scripts/fetch_weights.py --zip dimer.zip  # also write the DIMER upload archive (converted file only)
"""
# ruff: noqa: E501  -- messages and assertions are kept on one line

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from countgd_pipeline.config import (  # noqa: E402
    DEFAULT_MODEL_KEY,
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_REVISION,
    SOURCE_CKPT_NAME,
)
from countgd_pipeline.model import (  # noqa: E402
    DEFAULT_WEIGHTS_DIR,
    MANIFEST_NAME,
    ensure_converted,
    stage_missing_files,
    verify_converted,
    verify_snapshot,
)


def missing_entries(dest: Path) -> list[dict]:
    manifest = json.loads((dest / MANIFEST_NAME).read_text(encoding="utf-8"))
    return [entry for entry in manifest["files"] if not (dest / entry["path"]).is_file()]


def package_dimer_zip(dest: Path, zip_path: Path) -> Path:
    """The DIMER upload archive: the converted safetensors under `countgd/`. The pickle is never packaged."""
    verify_converted(dest)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(dest / MODEL_FILENAME, arcname=f"{DEFAULT_MODEL_KEY}/{MODEL_FILENAME}")
    return zip_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch, verify and convert the pinned CountGD checkpoint.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_WEIGHTS_DIR, help=f"snapshot directory (default: {DEFAULT_WEIGHTS_DIR})")
    parser.add_argument("--verify-only", action="store_true", help="verify the source snapshot and the converted file; fetch nothing")
    parser.add_argument("--dry-run", action="store_true", help="list the manifest entries that would be fetched")
    parser.add_argument("--zip", type=Path, default=None, help="write the DIMER upload archive (converted file only)")
    args = parser.parse_args(argv)
    dest: Path = args.dest
    if not (dest / MANIFEST_NAME).is_file():
        print(f"error: no {MANIFEST_NAME} in {dest}; copy the committed one from weights/{DEFAULT_MODEL_KEY}/")
        return 1
    print(f"snapshot: {MODEL_ID} (Space) @ {MODEL_REVISION[:12]} -> {dest}")
    try:
        if args.dry_run:
            for entry in missing_entries(dest):
                print(f"  would fetch {entry['path']} ({entry['bytes']:,} bytes)")
            return 0
        if args.verify_only:
            verify_snapshot(dest)
            converted = verify_converted(dest)
            print(f"OK: {SOURCE_CKPT_NAME} and {MODEL_FILENAME} ({converted['sha256'][:12]}...) match their pins")
            return 0
        fetched = stage_missing_files(dest, allow_download=True)
        print(f"fetched: {fetched or 'nothing (all present)'}")
        verify_snapshot(dest)
        report = ensure_converted(dest)
        state = "converted this run" if report["converted_this_run"] else "already present"
        print(f"OK: {MODEL_FILENAME} {state}, {report['bytes']:,} bytes, sha256 {report['sha256']}")
        if args.zip:
            print(f"DIMER archive: {package_dimer_zip(dest, args.zip)}")
        return 0
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
