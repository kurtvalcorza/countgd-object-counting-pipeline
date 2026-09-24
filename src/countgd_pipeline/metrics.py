"""Corpus-level measures for open-world counting, in numpy / torch: the count errors FSC-147 is scored on
(MAE, RMSE, and the normalised absolute error), a point-based localisation reading of the predicted boxes,
and two non-neural baselines scored by the same code — the mean training count and a normalised
cross-correlation template matcher built from the exemplar boxes."""
# ruff: noqa: E501  -- fleet metrics module written at the 110-column fleet width; this repo lints at 100

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment

METRIC_DEFINITIONS = {
    "mae": "mean over images of |predicted count - gold count|; lower is better (FSC-147's headline metric)",
    "rmse": "square root of the mean squared count error; lower is better, dominated by the largest errors",
    "nae": "mean over images of |predicted - gold| / gold; a scale-free count error, lower is better",
    "under_count_fraction": "fraction of images the system counts fewer objects than the gold count",
    "exact_fraction": "fraction of images counted exactly",
}
LOCALISATION_DEFINITIONS = {
    "precision": "fraction of predicted points matched one-to-one to a gold point within the match radius",
    "recall": "fraction of gold points matched one-to-one to a predicted point within the match radius",
    "f1": "harmonic mean of localisation precision and recall (micro-averaged over the scored images)",
    "match_radius": "per image: half the mean exemplar side when exemplars are given, else 2 % of the longer image side, never below 4 px",
}
MIN_MATCH_RADIUS = 4.0
DEFAULT_TEMPLATE_THRESHOLD = 0.6  # normalised cross-correlation peak the template matcher counts


# ------------------------------------------------------------------------------------ counting


def counting_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """MAE / RMSE / NAE over `{id, gold, predicted, label}` rows, with a per-class breakdown."""
    if not rows:
        raise ValueError("no rows to score")
    gold = np.asarray([float(r["gold"]) for r in rows])
    pred = np.asarray([float(r["predicted"]) for r in rows])
    if np.any(gold < 0) or np.any(pred < 0):
        raise ValueError("counts must be non-negative")
    err = pred - gold
    per_class: dict[str, dict[str, Any]] = {}
    for label in sorted({str(r.get("label", "")) for r in rows}):
        idx = [i for i, r in enumerate(rows) if str(r.get("label", "")) == label]
        e = err[idx]
        per_class[label] = {
            "n": len(idx),
            "mae": float(np.mean(np.abs(e))),
            "rmse": float(math.sqrt(np.mean(e**2))),
            "mean_gold": float(np.mean(gold[idx])),
            "mean_predicted": float(np.mean(pred[idx])),
        }
    safe_gold = np.where(gold > 0, gold, 1.0)
    return {
        "n": int(len(rows)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(np.mean(err**2))),
        "nae": float(np.mean(np.abs(err) / safe_gold)),
        "under_count_fraction": float(np.mean(err < 0)),
        "exact_fraction": float(np.mean(err == 0)),
        "total_gold": int(gold.sum()),
        "total_predicted": int(pred.sum()),
        "per_class": per_class,
        "definitions": dict(METRIC_DEFINITIONS),
    }


# ------------------------------------------------------------------------------------ localisation


def match_radius(record: Mapping[str, Any]) -> float:
    """Half the mean exemplar side, else 2 % of the longer image side; never below MIN_MATCH_RADIUS px."""
    exemplars = record.get("exemplars") or []
    if exemplars:
        sides = [((b[2] - b[0]) + (b[3] - b[1])) / 2.0 for b in exemplars]
        radius = 0.5 * float(np.mean(sides))
    else:
        width, height = record["image"].size
        radius = 0.02 * max(width, height)
    return max(MIN_MATCH_RADIUS, radius)


def match_points(predicted: Sequence[Sequence[float]], gold: Sequence[Sequence[float]], radius: float) -> dict[str, int]:
    """One-to-one Hungarian matching of predicted to gold points within `radius` (pixels): TP / FP / FN."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    if not predicted or not gold:
        return {"tp": 0, "fp": len(predicted), "fn": len(gold)}
    p = np.asarray(predicted, dtype=np.float64).reshape(-1, 2)
    g = np.asarray(gold, dtype=np.float64).reshape(-1, 2)
    dist = np.sqrt(((p[:, None, :] - g[None, :, :]) ** 2).sum(-1))
    cost = np.where(dist <= radius, dist, 1e6)
    rows, cols = linear_sum_assignment(cost)
    tp = int(np.sum(dist[rows, cols] <= radius))
    return {"tp": tp, "fp": int(len(p) - tp), "fn": int(len(g) - tp)}


def localisation_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Micro-averaged precision / recall / F1 over `{tp, fp, fn}` rows (images without gold points are skipped)."""
    scored = [r for r in rows if r is not None]
    if not scored:
        return {"n": 0, "precision": None, "recall": None, "f1": None, "definitions": dict(LOCALISATION_DEFINITIONS)}
    tp = sum(int(r["tp"]) for r in scored)
    fp = sum(int(r["fp"]) for r in scored)
    fn = sum(int(r["fn"]) for r in scored)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": len(scored), "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "definitions": dict(LOCALISATION_DEFINITIONS)}


# ------------------------------------------------------------------------------------ baselines


def mean_count_baseline(train: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every image gets the rounded mean gold count of the training set (the counting analogue of a majority floor)."""
    if not train:
        raise ValueError("the mean-count baseline needs training records")
    mean = float(np.mean([float(r["count"]) for r in train]))
    predicted = int(round(mean))
    rows = [{"id": r["id"], "label": r["label"], "gold": r["count"], "predicted": predicted} for r in records]
    result = counting_metrics(rows)
    result["rows"] = rows
    result["baseline"] = f"mean training count ({mean:.1f} -> {predicted} for every image)"
    return result


def _grey(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image.convert("L"), dtype=np.float32) / 255.0)


def _ncc_map(image: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    """Normalised cross-correlation of a zero-mean, unit-norm template over the image (same-size output)."""
    th, tw = template.shape
    t = template - template.mean()
    t_norm = torch.sqrt((t**2).sum()) + 1e-6
    pad = (tw // 2, tw - tw // 2 - 1, th // 2, th - th // 2 - 1)
    img = F.pad(image[None, None], pad, mode="reflect")
    ones = torch.ones(1, 1, th, tw)
    local_sum = F.conv2d(img, ones)
    local_sq = F.conv2d(img**2, ones)
    n = th * tw
    local_var = (local_sq - local_sum**2 / n).clamp_min(0.0)
    local_norm = torch.sqrt(local_var) + 1e-6
    corr = F.conv2d(img, t[None, None])
    return (corr / (t_norm * local_norm))[0, 0]


def template_match(record: Mapping[str, Any], *, threshold: float = DEFAULT_TEMPLATE_THRESHOLD) -> dict[str, Any]:
    """Count one image by normalised cross-correlation with the mean of its exemplar crops (grayscale, resized
    to the mean exemplar size): local maxima above `threshold`, one per exemplar-sized neighbourhood. A
    classical, learning-free use of the same exemplars the model gets."""
    exemplars = record.get("exemplars") or []
    if not exemplars:
        return {"count": None, "points": [], "note": "no exemplars"}
    image = record["image"].convert("RGB")
    grey = _grey(image)
    tw = max(3, int(round(float(np.mean([b[2] - b[0] for b in exemplars])))))
    th = max(3, int(round(float(np.mean([b[3] - b[1] for b in exemplars])))))
    tw, th = min(tw, grey.shape[1]), min(th, grey.shape[0])
    crops = []
    for x0, y0, x1, y1 in exemplars:
        crop = image.crop((int(round(x0)), int(round(y0)), max(int(round(x1)), int(round(x0)) + 1), max(int(round(y1)), int(round(y0)) + 1))).resize((tw, th), Image.BILINEAR)
        crops.append(_grey(crop))
    template = torch.stack(crops).mean(0)
    ncc = _ncc_map(grey, template)
    k = max(3, (min(th, tw) // 2) * 2 + 1)
    peaks = F.max_pool2d(ncc[None, None], kernel_size=k, stride=1, padding=k // 2)[0, 0]
    keep = (ncc >= threshold) & (ncc == peaks)
    ys, xs = torch.nonzero(keep, as_tuple=True)
    points = [[float(x), float(y)] for x, y in zip(xs.tolist(), ys.tolist(), strict=True)]
    return {"count": len(points), "points": points, "template_size": [tw, th], "threshold": threshold, "peak_max": float(ncc.max())}


def template_matching_baseline(records: Sequence[Mapping[str, Any]], *, threshold: float = DEFAULT_TEMPLATE_THRESHOLD) -> dict[str, Any]:
    """The template matcher over a set: count errors and, where gold points exist, localisation."""
    rows, loc_rows = [], []
    for record in records:
        result = template_match(record, threshold=threshold)
        predicted = result["count"] if result["count"] is not None else 0
        rows.append({"id": record["id"], "label": record["label"], "gold": record["count"], "predicted": predicted})
        if "points" in record:
            loc_rows.append(match_points(result["points"], record["points"], match_radius(record)))
    out = counting_metrics(rows)
    out["rows"] = rows
    out["localisation"] = localisation_metrics(loc_rows)
    out["baseline"] = f"normalised cross-correlation with the mean exemplar crop, peaks >= {threshold} (no learning)"
    return out


__all__ = [
    "DEFAULT_TEMPLATE_THRESHOLD",
    "LOCALISATION_DEFINITIONS",
    "METRIC_DEFINITIONS",
    "MIN_MATCH_RADIUS",
    "counting_metrics",
    "localisation_metrics",
    "match_points",
    "match_radius",
    "mean_count_baseline",
    "template_match",
    "template_matching_baseline",
]
