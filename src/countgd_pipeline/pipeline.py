"""The CountGD pipeline: open-world counting from a text prompt, visual exemplar boxes or both, its evaluation
on a counting set, a bounded continuation of the detection-style training on the last decoder layers, and
the adapter artifact that reloads onto a freshly verified base.

What the model does (Amini-Naieni, Han and Zisserman, NeurIPS 2024): GroundingDINO's Swin-B image encoder and
BERT text encoder fused by a feature enhancer, with the exemplar boxes' pooled image features inserted as extra
text-side tokens (`*`), and a DETR-style decoder of 900 queries each predicting a box and a per-token similarity
logit. Upstream counts the queries whose best token score exceeds 0.23; the predicted "boxes" are tiny — the
model was trained on 2 x 2 px boxes centred on FSC-147's points — so a prediction is read as a **point**, the
count is the number of points, and no box overlap is measured.
"""
# ruff: noqa: E501  -- fleet pipeline module written at the 110-column fleet width; this repo lints at 100

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .config import (
    CONFIDENCE_THRESHOLD,
    DECODER_LAYERS,
    DEFAULT_MODEL_KEY,
    IMAGE_MEAN,
    IMAGE_STD,
    MAX_EXEMPLARS,
    MAX_SIDE,
    MAX_TEXT_CHARS,
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    MODEL_SHA256,
    SHORT_SIDE,
    TARGET_BOX_PX,
)
from .metrics import (
    box_metrics,
    counting_metrics,
    localisation_metrics,
    match_boxes,
    match_points,
    match_radius,
)
from .model import (
    load_components,
    stage_missing_files,
    stage_missing_tokenizer_files,
    verify_snapshot,
    verify_tokenizer_snapshot,
)
from .samples import MAX_IMAGE_SIDE, MIN_IMAGE_SIDE, _check_boxes

ImageInput = str | Path | bytes | Image.Image

MAX_BATCH = 16  # images per count() call (each is one forward pass; the batch is a convenience, not a tensor batch)
DEFAULT_TRAINABLE_LAYERS = 2  # the last two of the six decoder layers train beside the shared box head
EXEMPLAR_CARRIER_WORD = "object"  # the caption word the exemplar tokens are attached to when no text is given
PARAMETER_COUNT = 233_362_816
MAX_EVAL_RECORDS = 5_000
MIN_SCORED_RECORDS = 50  # below this a scored set is labelled a small sample
ARTIFACT_FORMAT = "org.valcorza.countgd.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"
INPUT_SCHEMA = {
    "image": "local path, bytes or PIL image; sides in [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE]; resized to a shortest side of 800 px (longest at most 1333) like upstream's test transform",
    "text": "the category to count, at most 64 plain characters (upstream appends ' .'); optional when exemplars are given",
    "exemplars": "0..3 boxes [x0, y0, x1, y1] in the input image's pixels around single instances; optional when text is given",
    "threshold": "the score a query must exceed to count, in (0, 1); upstream's 0.23 by default",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------ input checks


def _coerce_image(value: ImageInput) -> Image.Image:
    if isinstance(value, Image.Image):
        image = value
    elif isinstance(value, bytes):
        import io

        image = Image.open(io.BytesIO(value))
    elif isinstance(value, str | Path):
        text = str(value)
        if text.lower().startswith(("http://", "https://")):
            raise ValueError("remote URLs are not accepted; pass a local path, bytes or a PIL image")
        path = Path(text)
        if not path.is_file():
            raise FileNotFoundError(f"image file not found: {path}")
        image = Image.open(path)
    else:
        raise ValueError("image must be a local path, bytes or a PIL image")
    image.load()
    return image.convert("RGB")


def _check_size(image: Image.Image, what: str) -> None:
    w, h = image.size
    if min(w, h) < MIN_IMAGE_SIDE or max(w, h) > MAX_IMAGE_SIDE:
        raise ValueError(f"{what}: sides must be in [{MIN_IMAGE_SIDE}, {MAX_IMAGE_SIDE}] px, got {w} x {h}")


def _check_text(text: Any) -> str | None:
    if text is None:
        return None
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    cleaned = " ".join(text.strip().lower().rstrip(".").split())
    if len(cleaned) > MAX_TEXT_CHARS:
        raise ValueError(f"text must be at most {MAX_TEXT_CHARS} characters")
    if not all(ch.isalnum() or ch in " _-'" for ch in cleaned):
        raise ValueError("text must be plain words (letters, digits, spaces, hyphens, apostrophes)")
    return cleaned


def _check_threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 < float(value) < 1.0:
        raise ValueError("threshold must be a number in (0, 1)")
    return float(value)


def _check_prompt(image: Image.Image, text: Any, exemplars: Any, what: str) -> tuple[str | None, list[list[float]]]:
    cleaned = _check_text(text)
    boxes = _check_boxes(exemplars, image.size[0], image.size[1], what) if exemplars else []
    if cleaned is None and not boxes:
        raise ValueError(f"{what}: give a text prompt, exemplar boxes, or both")
    return cleaned, boxes


def validate_inputs(
    images: Sequence[ImageInput] | ImageInput,
    *,
    text: str | None = None,
    exemplars: Sequence[Sequence[Sequence[float]]] | Sequence[Sequence[float]] | None = None,
    threshold: float = CONFIDENCE_THRESHOLD,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: the input manifest (schema, per-image observations, verdict), raising exactly what
    `count` would raise. `exemplars` is one box list for a single image or one list per image."""
    single = isinstance(images, str | Path | bytes | Image.Image)
    items = [images] if single else list(images)
    if not items:
        raise ValueError("images must contain at least one image")
    if len(items) > MAX_BATCH:
        raise ValueError(f"at most {MAX_BATCH} images per call")
    ratio = _check_threshold(threshold)
    per_image = _split_exemplars(exemplars, len(items))
    if names is not None and len(names) != len(items):
        raise ValueError("names must have one entry per image")
    observations = []
    for i, item in enumerate(items):
        image = _coerce_image(item)
        _check_size(image, f"image {i}")
        cleaned, boxes = _check_prompt(image, text, per_image[i], f"image {i}")
        observations.append({"id": names[i] if names else f"image-{i}", "mode": image.mode, "size": list(image.size), "text": cleaned, "exemplars": len(boxes)})
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": observations,
        "threshold": ratio,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def _split_exemplars(exemplars: Any, n: int) -> list[Any]:
    if exemplars is None:
        return [None] * n
    seq = list(exemplars)
    if seq and all(isinstance(b, Sequence) and len(b) == 4 and not isinstance(b[0], Sequence) for b in seq):
        if n != 1:
            raise ValueError("pass one exemplar list per image when counting several images")
        return [seq]
    if len(seq) != n:
        raise ValueError("exemplars must hold one box list per image")
    return seq


# ------------------------------------------------------------------ the pipeline


class CountGDPipeline:
    def __init__(
        self,
        model: Any,
        criterion: Any,
        tokenizer: Any,
        *,
        device: str | torch.device = "cpu",
        checkpoint_path: Path | str | None = None,
        checkpoint_source: str | None = None,
        manifest_verified: bool = False,
        weight_sha256: str | None = None,
        weight_size_bytes: int | None = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.checkpoint_source = checkpoint_source
        self.manifest_verified = manifest_verified
        self.weight_sha256 = weight_sha256
        self.weight_size_bytes = weight_size_bytes
        self.adapter: dict[str, Any] | None = None
        if hasattr(self.model, "parameters"):  # injected fakes in the offline tests carry none
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str | torch.device | None = None,
        cache_dir: str | Path | None = None,
        weights_path: str | Path | None = None,
        weights_dir: str | Path | None = None,
        tokenizer_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> CountGDPipeline:
        """Load the one supported checkpoint (fleet snapshot directories → stage absent entries when allowed →
        verify against the manifests and the pinned digests → convert the audited pickle once when the served file
        is absent → load the safetensors file strictly)."""
        manifest_verified = False
        if weights_dir is not None:
            if weights_path is not None:
                raise ValueError("pass either weights_dir or weights_path, not both")
            stage_missing_files(weights_dir, allow_download=allow_download)
            verify_snapshot(weights_dir)
            weights_path, manifest_verified = weights_dir, True
        if tokenizer_dir is not None:
            stage_missing_tokenizer_files(tokenizer_dir, allow_download=allow_download)
            verify_tokenizer_snapshot(tokenizer_dir)
        model, criterion, tokenizer, target_device, _, metadata = load_components(device=device, cache_dir=cache_dir, weights_path=weights_path, tokenizer_path=tokenizer_dir, manifest_verified=manifest_verified, return_metadata=True)
        return cls(
            model,
            criterion,
            tokenizer,
            device=target_device,
            checkpoint_path=metadata.get("checkpoint_path"),
            checkpoint_source=metadata.get("checkpoint_source"),
            manifest_verified=metadata.get("manifest_verified", False),
            weight_sha256=metadata.get("weight_sha256"),
            weight_size_bytes=metadata.get("weight_size_bytes"),
        )

    # ------------------------------------------------------------------ tensors

    @staticmethod
    def _scale(image: Image.Image) -> tuple[float, tuple[int, int]]:
        w, h = image.size
        scale = SHORT_SIDE / min(w, h)
        if max(w, h) * scale > MAX_SIDE:
            scale = MAX_SIDE / max(w, h)
        return scale, (max(1, round(w * scale)), max(1, round(h * scale)))

    def _prepare(self, image: Image.Image) -> tuple[torch.Tensor, float]:
        scale, (nw, nh) = self._scale(image)
        resized = image if (nw, nh) == image.size else image.resize((nw, nh), Image.BILINEAR)
        pixels = TF.normalize(TF.to_tensor(resized), list(IMAGE_MEAN), list(IMAGE_STD))
        return pixels.to(self.device), scale

    def _forward(self, pixels: torch.Tensor, exemplars: Sequence[Sequence[float]], scale: float, text: str | None, targets: Sequence[Mapping[str, Any]] | None = None) -> Any:
        boxes = torch.tensor([[b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale] for b in exemplars], dtype=torch.float32, device=self.device).reshape(-1, 4)
        # Exemplar-only: upstream's evaluation removes the class word from a caption that still names other
        # classes and inserts the exemplar tokens where it stood; with a single-class caption that leaves no word
        # to anchor them, so the neutral carrier word below stands in and the exemplar tokens follow it.
        caption = f"{text} ." if text else f"{EXEMPLAR_CARRIER_WORD} ."
        if targets is None:
            return self.model(pixels.unsqueeze(0), [boxes], [torch.tensor([0], device=self.device)], captions=[caption])
        return self.model(pixels.unsqueeze(0), [boxes], [torch.tensor([0], device=self.device)], targets=[{**targets[0], "caption": caption}])

    @staticmethod
    def _read(out: Any, threshold: float, size: tuple[int, int]) -> dict[str, Any]:
        logits = out["pred_logits"][0].sigmoid()
        boxes = out["pred_boxes"][0]
        scores = logits.max(dim=-1).values
        keep = scores > threshold
        w, h = size
        kept = boxes[keep].detach().cpu().numpy().astype(np.float64)
        kept_scores = scores[keep].detach().cpu().numpy().astype(np.float64)
        xyxy = np.stack([(kept[:, 0] - kept[:, 2] / 2) * w, (kept[:, 1] - kept[:, 3] / 2) * h, (kept[:, 0] + kept[:, 2] / 2) * w, (kept[:, 1] + kept[:, 3] / 2) * h], axis=1) if len(kept) else np.zeros((0, 4))
        points = np.stack([kept[:, 0] * w, kept[:, 1] * h], axis=1) if len(kept) else np.zeros((0, 2))
        order = np.argsort(-kept_scores)
        return {
            "count": int(keep.sum()),
            "points": points[order].round(2).tolist(),
            "boxes": xyxy[order].round(2).tolist(),
            "scores": kept_scores[order].round(4).tolist(),
            "max_score": float(scores.max()),
            "queries": int(scores.numel()),
        }

    # ------------------------------------------------------------------ inference contract

    def count(
        self,
        images: Sequence[ImageInput] | ImageInput,
        *,
        text: str | None = None,
        exemplars: Sequence[Sequence[Sequence[float]]] | Sequence[Sequence[float]] | None = None,
        threshold: float = CONFIDENCE_THRESHOLD,
    ) -> dict[str, Any]:
        """Count the objects a text prompt and/or 0..3 exemplar boxes describe: one result per image with the
        count, the predicted points (box centres in the input image's pixels, best first), the predicted boxes and
        the scores. Text alone, exemplars alone, or both, as upstream allows."""
        manifest = validate_inputs(images, text=text, exemplars=exemplars, threshold=threshold)
        single = isinstance(images, str | Path | bytes | Image.Image)
        items = [images] if single else list(images)
        per_image = _split_exemplars(exemplars, len(items))
        cleaned = _check_text(text)
        results = []
        for i, item in enumerate(items):
            image = _coerce_image(item)
            boxes = _check_boxes(per_image[i], image.size[0], image.size[1], f"image {i}") if per_image[i] else []
            pixels, scale = self._prepare(image)
            with torch.no_grad():
                out = self._forward(pixels, boxes, scale, cleaned)
            entry = self._read(out, manifest["threshold"], image.size)
            entry.update({"input_size": list(image.size), "model_input_size": [int(pixels.shape[-1]), int(pixels.shape[-2])], "text": cleaned, "exemplars": boxes})
            results.append(entry)
        return {
            "results": results,
            "threshold": manifest["threshold"],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_license": MODEL_LICENSE,
        }

    # ------------------------------------------------------------------ evaluation

    def evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        threshold: float = CONFIDENCE_THRESHOLD,
        use_text: bool = True,
        use_exemplars: bool = True,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Count every record of a validated set with its own label and exemplars and score the counts (MAE,
        RMSE, NAE, per class), the localisation of the predicted points where gold points exist and the extent of
        the predicted boxes (IoU matching) where gold object boxes exist."""
        from .samples import validate_dataset

        if not use_text and not use_exemplars:
            raise ValueError("evaluate with text, exemplars, or both")
        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        ratio = _check_threshold(threshold)
        rows, loc_rows, box_rows, per_image = [], [], [], []
        started = time.perf_counter()
        for i, record in enumerate(checked):
            exemplars = record["exemplars"] if use_exemplars else []
            text = record["label"] if use_text else None
            if not exemplars and text is None:
                raise ValueError(f"{record['id']}: no exemplars to count with and text is off")
            pixels, scale = self._prepare(record["image"])
            with torch.no_grad():
                out = self._forward(pixels, exemplars, scale, text)
            entry = self._read(out, ratio, record["image"].size)
            rows.append({"id": record["id"], "label": record["label"], "gold": record["count"], "predicted": entry["count"]})
            loc = None
            if "points" in record:
                loc = match_points(entry["points"], record["points"], match_radius(record))
                loc_rows.append(loc)
            box = None
            if "boxes" in record:
                box = match_boxes(entry["boxes"], record["boxes"])
                box_rows.append(box)
            per_image.append({"id": record["id"], "label": record["label"], "gold": record["count"], "predicted": entry["count"], "max_score": entry["max_score"], "localisation": loc, "boxes": {k: box[k] for k in ("tp", "fp", "fn")} if box else None})
            if progress is not None:
                progress(i + 1, len(checked))
        result = counting_metrics(rows)
        result["localisation"] = localisation_metrics(loc_rows)
        result["boxes"] = box_metrics(box_rows)
        result["per_image"] = per_image
        result["threshold"] = ratio
        result["prompt"] = {"text": use_text, "exemplars": use_exemplars}
        result["seconds"] = round(time.perf_counter() - started, 2)
        result["verdict"] = "measured" if len(checked) >= MIN_SCORED_RECORDS else "small-sample"
        result["adapted"] = self.adapter is not None
        return result

    # ------------------------------------------------------------------ adaptation contract

    def _trainable_names(self, trainable_layers: int) -> list[str]:
        if isinstance(trainable_layers, bool) or not isinstance(trainable_layers, int) or not 0 <= trainable_layers <= DECODER_LAYERS:
            raise ValueError(f"trainable_layers must be an int in 0..{DECODER_LAYERS}")
        prefixes = tuple(f"transformer.decoder.layers.{i}." for i in range(DECODER_LAYERS - trainable_layers, DECODER_LAYERS))
        names = [n for n, _ in self.model.named_parameters() if n.startswith(prefixes) or n.startswith("bbox_embed.0.") or n.startswith("transformer.decoder.norm.")]
        return names

    def _targets(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Training boxes in normalised cx, cy, w, h: the gold object boxes when the record has them, else upstream's
        2 x 2 px boxes centred on the gold points (FSC-147 annotates points only)."""
        w, h = record["image"].size
        if record.get("boxes"):
            b = torch.tensor(record["boxes"], dtype=torch.float32)
            boxes = torch.stack([(b[:, 0] + b[:, 2]) / 2 / w, (b[:, 1] + b[:, 3]) / 2 / h, (b[:, 2] - b[:, 0]) / w, (b[:, 3] - b[:, 1]) / h], dim=1)
            return {"boxes": boxes.to(self.device), "labels": torch.zeros(len(b), dtype=torch.long, device=self.device)}
        points = record.get("points")
        if not points:
            raise ValueError(f"{record['id']}: adaptation needs gold points (count alone cannot place the training boxes)")
        pts = torch.tensor(points, dtype=torch.float32)
        boxes = torch.stack([pts[:, 0] / w, pts[:, 1] / h, torch.full((len(pts),), TARGET_BOX_PX / w), torch.full((len(pts),), TARGET_BOX_PX / h)], dim=1)
        return {"boxes": boxes.to(self.device), "labels": torch.zeros(len(pts), dtype=torch.long, device=self.device)}

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 3,
        lr: float = 1e-5,
        trainable_layers: int = DEFAULT_TRAINABLE_LAYERS,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded continuation of CountGD's training on a validated counting set with gold points.

        Trains the last `trainable_layers` decoder layers, the decoder's final norm and the shared box head on
        upstream's own objective — the token-level sigmoid focal loss and the L1 box loss of `SetCriterion`
        after Hungarian matching, over the final and every intermediate decoder output — with the caption
        `<label> .`, the record's exemplars and, as target boxes, the record's gold object boxes when it has them
        (the synthetic scenes) or upstream's 2 x 2 px boxes centred on the gold points (FSC-147), one image per
        step; AdamW at a fixed learning rate (weight decay 1e-4), gradient clipping at 0.1 (upstream's), seeded
        order, no scheduler, no augmentation. Epoch 0 records the frozen model's validation MAE; every epoch is
        scored on the validation set and the epoch with the lowest validation MAE is kept — the frozen model
        itself when nothing beats it. Transactional: any failure restores the base tensors."""
        from .samples import validate_dataset

        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 20:
            raise ValueError("epochs must be an int in 1..20")
        if not (0.0 < lr <= 1e-3):
            raise ValueError("lr must be in (0, 1e-3]")
        names = self._trainable_names(trainable_layers)
        train_checked = validate_dataset(train, min_records=1)["records"]
        val_checked = validate_dataset(val, min_records=1, max_records=MAX_EVAL_RECORDS)["records"] if val else []
        for record in train_checked:
            if not record.get("points"):
                raise ValueError(f"{record['id']}: adaptation needs gold points")
        model = self.model
        wanted = set(names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in wanted)
        params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in params)
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        started = time.perf_counter()
        weight_dict = self.criterion.weight_dict

        def score_val() -> dict[str, Any] | None:
            if not val_checked:
                return None
            model.eval()
            result = self.evaluate(val_checked)
            return {k: result[k] for k in ("mae", "rmse", "nae", "n")} | {"localisation_f1": result["localisation"]["f1"], "box_f1": result["boxes"]["f1"]}

        def key(entry: dict[str, Any]) -> float:
            return -entry["val"]["mae"] if entry["val"] else -math.inf

        history: list[dict[str, Any]] = []
        entry: dict[str, Any] = {"epoch": 0, "train_loss": None, "val": score_val(), "note": "frozen model"}
        history.append(entry)
        if progress is not None:
            progress(entry)
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
        initial_state = {k: v.clone() for k, v in best_state.items()}
        best_epoch = 0
        rng = random.Random(seed)
        torch.manual_seed(seed)
        try:
            for epoch in range(1, epochs + 1):
                model.train()
                order = list(range(len(train_checked)))
                rng.shuffle(order)
                losses = []
                for index in order:
                    record = train_checked[index]
                    pixels, scale = self._prepare(record["image"])
                    target = self._targets(record)
                    optimiser.zero_grad(set_to_none=True)
                    out = self._forward(pixels, record["exemplars"], scale, record["label"], targets=[target])
                    loss_dict = self.criterion(out, [target], [[record["label"]]], [f"{record['label']} ."])
                    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 0.1)
                    optimiser.step()
                    losses.append(float(loss.detach()))
                model.eval()
                entry = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val": score_val()}
                history.append(entry)
                if progress is not None:
                    progress(entry)
                if val_checked:
                    if key(entry) > key(history[best_epoch]):
                        best_epoch = epoch
                        best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
                else:
                    best_epoch = epoch
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
        except BaseException:
            self._overlay(initial_state)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            raise
        self._overlay(best_state)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "objective": "CountGD's own: token sigmoid focal loss + L1 box loss after Hungarian matching, final and intermediate decoder outputs",
            "trainable_layers": trainable_layers,
            "n_trainable": int(n_trainable),
            "n_total": int(sum(p.numel() for p in model.parameters())),
            "epochs": epochs,
            "lr": lr,
            "batch_size": 1,
            "seed": seed,
            "n_train": len(train_checked),
            "n_val": len(val_checked),
            "best_epoch": best_epoch,
            "selection": "lowest validation MAE (validation split)" if val_checked else "final epoch (no validation split)",
            "seconds": round(time.perf_counter() - started, 2),
            "history": history,
            "trainable_names": names,
        }
        return dict(self.adapter)

    def _overlay(self, tensors: Mapping[str, torch.Tensor]) -> None:
        """Copy tensors into the model's parameters by name (shared modules — the tied box heads — follow)."""
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, value in tensors.items():
                if name not in params:
                    raise ValueError(f"tensor {name} is not a parameter of the base")
                if tuple(value.shape) != tuple(params[name].shape):
                    raise ValueError(f"tensor {name} has shape {tuple(value.shape)}, base has {tuple(params[name].shape)}")
                params[name].copy_(value.to(params[name].device, params[name].dtype))

    # ------------------------------------------------------------------ artifact

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the adapted tensors as safetensors plus a base manifest."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        params = dict(self.model.named_parameters())
        tensors = {k: params[k].detach().cpu().contiguous() for k in self.adapter["trainable_names"]}
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": DEFAULT_MODEL_KEY, "weight_file": MODEL_FILENAME, "weight_sha256": MODEL_SHA256},
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [{"path": ARTIFACT_WEIGHTS_NAME, "bytes": weights_path.stat().st_size, "sha256": _sha256_file(weights_path)}],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest, digest and exact tensor set **before** deserialising, then overwrite
        exactly the parameters it carries."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        weights_path = _check_artifact_manifest(root, manifest)
        entry = manifest["files"][0]
        if not weights_path.is_file():
            raise FileNotFoundError(f"artifact weights missing: {weights_path}")
        if _sha256_file(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        expected = sorted(self._trainable_names(manifest["adapter"]["trainable_layers"]))
        if sorted(manifest["tensors"]) != expected:
            raise ValueError("artifact tensor list does not match its recorded configuration")
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != expected:
            raise ValueError("artifact tensor names differ from its manifest")
        self._overlay(tensors)
        self.model.eval()
        self.adapter = {**manifest["adapter"], "trainable_names": list(manifest["tensors"]), "history": manifest.get("history", [])}
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | torch.device | None = None,
        weights_dir: str | Path | None = None,
        tokenizer_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> CountGDPipeline:
        """A fresh pipeline from the pinned base with an adapter overlaid."""
        pipe = cls.from_pretrained(device=device, weights_dir=weights_dir, tokenizer_dir=tokenizer_dir, allow_download=allow_download)
        pipe.load_artifact(artifact_dir)
        return pipe


def _check_artifact_manifest(root: Path, manifest: Mapping[str, Any]) -> Path:
    """Refuse an artifact whose manifest is not exactly the one this package writes. Nothing is deserialised
    here; the digest check that follows detects corruption or drift relative to the adjacent manifest."""
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
    if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(f"artifact format_version {manifest.get('format_version')!r} is not the supported {ARTIFACT_FORMAT_VERSION!r}")
    base = manifest.get("base_model", {})
    if (base.get("id"), base.get("revision"), base.get("weight_sha256")) != (MODEL_ID, MODEL_REVISION, MODEL_SHA256):
        raise ValueError("artifact was adapted from a different base model, revision or weight file")
    if base.get("weight_file", MODEL_FILENAME) != MODEL_FILENAME:
        raise ValueError("artifact was adapted from a different base weight file")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 1:
        raise ValueError("artifact manifest must list exactly one file")
    entry = files[0]
    if not isinstance(entry, Mapping) or entry.get("path") != ARTIFACT_WEIGHTS_NAME:
        raise ValueError(f"artifact manifest must name exactly {ARTIFACT_WEIGHTS_NAME!r}")
    weights_path = (root / entry["path"]).resolve()
    if weights_path.parent != root.resolve():
        raise ValueError("artifact weight path must resolve inside the artifact directory")
    adapter = manifest.get("adapter")
    layers = adapter.get("trainable_layers") if isinstance(adapter, Mapping) else None
    if isinstance(layers, bool) or not isinstance(layers, int) or not 0 <= layers <= DECODER_LAYERS:
        raise ValueError("artifact manifest does not record an in-range integer trainable_layers")
    if not isinstance(manifest.get("tensors"), list):
        raise ValueError("artifact manifest must list its tensors")
    return weights_path


def load_pipeline(**kwargs: Any) -> CountGDPipeline:
    return CountGDPipeline.from_pretrained(**kwargs)


def evaluation_report(result: Mapping[str, Any], expected: Sequence[int] | None = None, *, sample_kind: str = "synthetic") -> dict[str, Any]:
    """The per-call reading of a `count` result: the counts, and against `expected` counts the absolute
    errors, with a `sample-sanity` verdict on drawings (plumbing evidence) or `measured` on a scored set."""
    rows = result.get("results", [])
    if not rows:
        raise ValueError("no results to report")
    counts = [int(r["count"]) for r in rows]
    metrics = [{"id": "count", "value": counts, "definition": "objects counted per image (queries above the threshold)"}]
    if expected is not None:
        if len(expected) != len(rows):
            raise ValueError("expected must have one count per result")
        errors = [abs(c - int(e)) for c, e in zip(counts, expected, strict=True)]
        metrics.append({"id": "mae", "value": float(np.mean(errors)), "definition": METRIC_DEFINITIONS_MAE})
        metrics.append({"id": "exact", "value": int(sum(1 for e in errors if e == 0)), "definition": "images counted exactly"})
    return {
        "metrics": metrics,
        "n": len(rows),
        "sample_kind": sample_kind,
        "verdict": "sample-sanity" if sample_kind.startswith("synthetic") or len(rows) < MIN_SCORED_RECORDS else "measured",
        "note": "a count above a fixed score threshold; nothing here is calibrated and a drawing is plumbing evidence, not a measurement",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


METRIC_DEFINITIONS_MAE = "mean over images of |predicted count - expected count|; lower is better"

__all__ = [
    "ARTIFACT_FORMAT",
    "DEFAULT_TRAINABLE_LAYERS",
    "INPUT_SCHEMA",
    "MAX_BATCH",
    "MAX_EXEMPLARS",
    "MAX_IMAGE_SIDE",
    "MIN_IMAGE_SIDE",
    "PARAMETER_COUNT",
    "CountGDPipeline",
    "evaluation_report",
    "load_pipeline",
    "validate_inputs",
]
