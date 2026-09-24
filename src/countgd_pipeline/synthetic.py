"""Synthetic counting scenes with known object boxes: the tutorial's first sample and the box-IoU benchmark.

FSC-147 annotates one point per object and three exemplar boxes per image, so it can score a count and the
placement of the predicted points but not the extent of the predicted boxes. These scenes supply what it lacks:
every counted object is drawn inside a known box. A scene is a flat background with clutter specks, 6 to 40
**targets** of one colour and shape (the category the prompt names, e.g. ``"red circle"``) and 4 to 20
**distractors** of a different colour *and* shape, which a counter must leave out. Everything is drawn with
integer rasterisation from a seeded `random.Random`, so a seed always yields the same pixels; the splits use
disjoint seed ranges, so no scene appears in two splits.
"""
# ruff: noqa: E501

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from PIL import Image, ImageDraw

SYNTHETIC_SIZE = (512, 384)  # width, height; the counter resizes to a shortest side of 800 px like any image
SYNTHETIC_SHAPES = ("circle", "square", "triangle")
SYNTHETIC_COLOURS: dict[str, tuple[int, int, int]] = {
    "red": (200, 40, 40),
    "blue": (40, 70, 200),
    "green": (40, 150, 60),
    "yellow": (220, 190, 30),
}
SYNTHETIC_TARGETS = (6, 40)  # inclusive range of target objects per scene
SYNTHETIC_DISTRACTORS = (4, 20)  # inclusive range of distractor objects per scene
SYNTHETIC_RADIUS = (7, 14)  # half-side of an object's box in pixels (each object varies by +-2 around the scene's radius)
SYNTHETIC_SPECKS = 300
SYNTHETIC_SEED_BASE = {"train": 1_000, "validation": 2_000, "test": 3_000}
SYNTHETIC_SPLIT = {"train": 24, "validation": 8, "test": 12}
DEMO_SEED = 1  # the tutorial's first scene: 35 blue circles among 16 green triangles (text and text+exemplars count 35 on the CPU; exemplars alone count every shape, 51)


def _draw(draw: ImageDraw.ImageDraw, shape: str, box: list[int], fill: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    if shape == "circle":
        draw.ellipse(box, fill=fill)
    elif shape == "square":
        draw.rectangle(box, fill=fill)
    else:
        draw.polygon([((x0 + x1) // 2, y0), (x1, y1), (x0, y1)], fill=fill)


def synthetic_scene(seed: int, *, n_targets: int | None = None) -> dict[str, Any]:
    """One scene as a counting record: ``{id, image, label, count, boxes, points, exemplars, distractors}``.

    `boxes` are the targets' full extents `[x0, y0, x1, y1]` (inclusive pixel bounds of the drawn shape plus one,
    so a box's width is the shape's width), `points` their centres, `exemplars` three of the target boxes chosen
    by the same seed. `n_targets` overrides the drawn target count (clipped to what fits)."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative int")
    rng = random.Random(seed)
    width, height = SYNTHETIC_SIZE
    image = Image.new("RGB", SYNTHETIC_SIZE, tuple(rng.randint(150, 225) for _ in range(3)))
    draw = ImageDraw.Draw(image)
    for _ in range(SYNTHETIC_SPECKS):
        draw.point((rng.randrange(width), rng.randrange(height)), fill=tuple(rng.randint(90, 250) for _ in range(3)))
    target_shape = rng.choice(SYNTHETIC_SHAPES)
    target_colour = rng.choice(sorted(SYNTHETIC_COLOURS))
    distractor_shape = rng.choice([s for s in SYNTHETIC_SHAPES if s != target_shape])
    distractor_colour = rng.choice([c for c in sorted(SYNTHETIC_COLOURS) if c != target_colour])
    wanted_targets = rng.randint(*SYNTHETIC_TARGETS) if n_targets is None else int(n_targets)
    wanted_distractors = rng.randint(*SYNTHETIC_DISTRACTORS)
    radius = rng.randint(*SYNTHETIC_RADIUS)
    placed: list[tuple[int, int, int]] = []
    targets: list[list[float]] = []
    distractors = 0
    for kind, wanted in (("target", wanted_targets), ("distractor", wanted_distractors)):
        tries = 0
        while wanted > 0 and tries < 5_000:
            tries += 1
            size = radius + rng.randint(-2, 2)
            cx, cy = rng.randint(size + 2, width - size - 3), rng.randint(size + 2, height - size - 3)
            if any((cx - px) ** 2 + (cy - py) ** 2 <= (size + ps + 3) ** 2 for px, py, ps in placed):
                continue
            placed.append((cx, cy, size))
            box = [cx - size, cy - size, cx + size, cy + size]
            if kind == "target":
                _draw(draw, target_shape, box, SYNTHETIC_COLOURS[target_colour])
                targets.append([float(box[0]), float(box[1]), float(box[2] + 1), float(box[3] + 1)])
            else:
                _draw(draw, distractor_shape, box, SYNTHETIC_COLOURS[distractor_colour])
                distractors += 1
            wanted -= 1
    exemplars = [list(b) for b in rng.sample(targets, k=min(3, len(targets)))]
    return {
        "id": f"synth-{seed:05d}",
        "image": image,
        "label": f"{target_colour} {target_shape}",
        "count": len(targets),
        "boxes": targets,
        "points": [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in targets],
        "exemplars": exemplars,
        "distractors": {"count": distractors, "label": f"{distractor_colour} {distractor_shape}"},
    }


def build_synthetic_dataset(sizes: Mapping[str, int] | None = None) -> dict[str, list[dict[str, Any]]]:
    """The seeded synthetic splits: scene `i` of split `s` is ``synthetic_scene(SYNTHETIC_SEED_BASE[s] + i)``."""
    sizes = dict(sizes or SYNTHETIC_SPLIT)
    out: dict[str, list[dict[str, Any]]] = {}
    for name, n in sizes.items():
        if name not in SYNTHETIC_SEED_BASE:
            raise ValueError(f"unknown split {name!r}; expected one of {sorted(SYNTHETIC_SEED_BASE)}")
        if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n < 1_000:
            raise ValueError(f"{name}: size must be an int in 0..999")
        out[name] = [synthetic_scene(SYNTHETIC_SEED_BASE[name] + i) for i in range(n)]
    return out
