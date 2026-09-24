"""Opt-in smoke of the real pinned checkpoint: download the 1.25 GB source from the authors' Space, audit and
convert it, and count the tutorial's demo scene (RUN_INTEGRATION=1; network and ~2.2 GB of disk)."""
# ruff: noqa: E501  -- messages and assertions are kept on one line
from __future__ import annotations

import os

import pytest

from countgd_pipeline import DEMO_SEED, MODEL_SHA256, CountGDPipeline, synthetic_scene

pytestmark = pytest.mark.integration


def test_real_checkpoint_smoke(tmp_path):
    if os.environ.get("RUN_INTEGRATION") != "1":
        pytest.skip("set RUN_INTEGRATION=1 to download, convert and run the pinned checkpoint")
    pipe = CountGDPipeline.from_pretrained(device="cpu", cache_dir=tmp_path)
    assert pipe.weight_sha256 == MODEL_SHA256
    scene = synthetic_scene(DEMO_SEED)
    entry = pipe.count(scene["image"], text=scene["label"], exemplars=scene["exemplars"])["results"][0]
    assert abs(entry["count"] - scene["count"]) <= max(2, scene["count"] // 5)
