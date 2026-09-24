"""Corpus-level measures for open-world counting, in numpy / torch: the count errors FSC-147 is scored on
(MAE, RMSE, and the normalised absolute error), a point-based localisation reading of the predicted boxes,
a box-extent reading (IoU matching) where gold object boxes exist, and two non-neural baselines scored by the same code — the mean training count and a normalised
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
MIN_WINDOW_STD = 0.01  # grey levels in [0, 1]: an image window or template flatter than this has no defined correlation


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


# ------------------------------------------------------------------------------------ box extents

BOX_DEFINITIONS = {
    "iou_threshold": "a predicted box is a true positive when its one-to-one Hungarian partner (maximising IoU) is a gold box with IoU >= this",
    "precision": "fraction of predicted boxes matched to a gold box at IoU >= the threshold",
    "recall": "fraction of gold boxes matched to a predicted box at IoU >= the threshold",
    "f1": "harmonic mean of box precision and recall (micro-averaged over the scored images)",
    "mean_matched_iou": "mean IoU of the true-positive pairs (how tight the boxes are that do match)",
    "mean_best_iou": "mean over gold boxes of the highest IoU any predicted box reaches (0 for a gold box nothing overlaps)",
}
BOX_IOU_THRESHOLD = 0.5


def box_iou_matrix(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> np.ndarray:
    """Pairwise IoU of `[x0, y0, x1, y1]` boxes, shape (len(a), len(b))."""
    p = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    g = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    ix = np.clip(np.minimum(p[:, None, 2], g[None, :, 2]) - np.maximum(p[:, None, 0], g[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(p[:, None, 3], g[None, :, 3]) - np.maximum(p[:, None, 1], g[None, :, 1]), 0, None)
    inter = ix * iy
    area_p = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    area_g = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    union = area_p[:, None] + area_g[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def match_boxes(predicted: Sequence[Sequence[float]], gold: Sequence[Sequence[float]], threshold: float = BOX_IOU_THRESHOLD) -> dict[str, Any]:
    """One-to-one Hungarian matching of predicted to gold boxes by IoU: TP / FP / FN at `threshold`, the IoUs of
    the true positives and each gold box's best IoU."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if not predicted or not gold:
        return {"tp": 0, "fp": len(predicted), "fn": len(gold), "matched_iou": [], "best_iou": [0.0] * len(gold)}
    iou = box_iou_matrix(predicted, gold)
    rows, cols = linear_sum_assignment(-iou)
    hits = iou[rows, cols] >= threshold
    tp = int(hits.sum())
    return {
        "tp": tp,
        "fp": int(len(predicted) - tp),
        "fn": int(len(gold) - tp),
        "matched_iou": [float(v) for v in iou[rows, cols][hits]],
        "best_iou": [float(v) for v in iou.max(axis=0)],
    }


def box_metrics(rows: Sequence[Mapping[str, Any] | None], threshold: float = BOX_IOU_THRESHOLD) -> dict[str, Any]:
    """Micro-averaged box precision / recall / F1 at the IoU threshold, the mean IoU of the matched pairs and the
    mean best IoU per gold box, over `match_boxes` rows (images without gold boxes are skipped)."""
    scored = [r for r in rows if r is not None]
    base = {"iou_threshold": threshold, "definitions": dict(BOX_DEFINITIONS)}
    if not scored:
        return {"n": 0, "precision": None, "recall": None, "f1": None, "mean_matched_iou": None, "mean_best_iou": None, **base}
    tp = sum(int(r["tp"]) for r in scored)
    fp = sum(int(r["fp"]) for r in scored)
    fn = sum(int(r["fn"]) for r in scored)
    matched = [v for r in scored for v in r["matched_iou"]]
    best = [v for r in scored for v in r["best_iou"]]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(scored),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": float(np.mean(matched)) if matched else None,
        "mean_best_iou": float(np.mean(best)) if best else None,
        **base,
    }


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
    """Normalised cross-correlation of a zero-mean template over the image (same-size output, in [-1, 1]).

    The correlation is undefined where the image window (or the template) is flat; those positions get 0
    rather than a ratio of two near-zero numbers, which float error would turn into spurious peaks."""
    th, tw = template.shape
    n = th * tw
    t = template - template.mean()
    t_var = (t**2).sum()
    if t_var <= (MIN_WINDOW_STD**2) * n:
        return torch.zeros_like(image)
    pad = (tw // 2, tw - tw // 2 - 1, th // 2, th - th // 2 - 1)
    img = F.pad(image[None, None], pad, mode="reflect")
    ones = torch.ones(1, 1, th, tw)
    local_sum = F.conv2d(img, ones)
    local_sq = F.conv2d(img**2, ones)
    local_var = (local_sq - local_sum**2 / n).clamp_min(0.0)
    corr = F.conv2d(img, t[None, None])
    valid = local_var > (MIN_WINDOW_STD**2) * n
    ncc = torch.where(valid, corr / (torch.sqrt(t_var) * torch.sqrt(local_var.clamp_min(1e-12))), torch.zeros_like(corr))
    return ncc.clamp(-1.0, 1.0)[0, 0]


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
    "BOX_DEFINITIONS",
    "BOX_IOU_THRESHOLD",
    "DEFAULT_TEMPLATE_THRESHOLD",
    "LOCALISATION_DEFINITIONS",
    "METRIC_DEFINITIONS",
    "MIN_MATCH_RADIUS",
    "box_iou_matrix",
    "box_metrics",
    "counting_metrics",
    "localisation_metrics",
    "match_boxes",
    "match_points",
    "match_radius",
    "mean_count_baseline",
    "template_match",
    "template_matching_baseline",
]
