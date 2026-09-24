"""Offline tests of the inference contract on a tiny stand-in for the CountGD network: the three prompt modes,
the threshold decision, points from boxes, the input manifest, rejections and batch ceilings. Nothing here
loads the checkpoint."""
# ruff: noqa: E501

from __future__ import annotations

import pytest
import torch
from PIL import Image, ImageDraw

from countgd_pipeline import (
    CONFIDENCE_THRESHOLD,
    DECODER_LAYERS,
    HIDDEN_DIM,
    MAX_BATCH,
    MODEL_ID,
    MODEL_REVISION,
    NUM_QUERIES,
    CountGDPipeline,
    evaluation_report,
    validate_inputs,
)
from countgd_pipeline import pipeline as pl


class FakeTokenizer:
    def __call__(self, captions, **kwargs):
        return {"input_ids": torch.tensor([[101] + [2000 + i for i in range(len(c.split()))] + [102] for c in captions])}


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)


class FakeModel(torch.nn.Module):
    """The surface `CountGDPipeline` uses: parameters under the decoder / box-head names the adaptation
    contract trains, and a forward that 'detects' whatever `self.detections` holds — normalised (cx, cy)
    points the fake places confident queries at — with lower scores when the prompt is exemplar-only."""

    def __init__(self, detections=None) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.backbone = torch.nn.Linear(3, 3)
        self.bert = torch.nn.Linear(3, 3)
        self.transformer = torch.nn.Module()
        self.transformer.encoder = torch.nn.Linear(4, 4)
        self.transformer.decoder = torch.nn.Module()
        self.transformer.decoder.layers = torch.nn.ModuleList([_Layer() for _ in range(DECODER_LAYERS)])
        self.transformer.decoder.norm = torch.nn.LayerNorm(HIDDEN_DIM)
        self.bbox_embed = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
        self.detections = detections if detections is not None else [(0.25, 0.25), (0.75, 0.75)]
        self.calls: list[dict] = []
        self.score = 0.9

    def forward(self, samples, exemplars, labels, targets=None, captions=None, **kw):
        caption = captions[0] if captions else targets[0]["caption"]
        self.calls.append({"caption": caption, "exemplars": int(exemplars[0].shape[0]), "size": tuple(samples.shape[-2:])})
        logits = torch.full((1, NUM_QUERIES, 8), -6.0)
        boxes = torch.full((1, NUM_QUERIES, 4), 0.5)
        boxes[..., 2:] = 0.002
        for i, (cx, cy) in enumerate(self.detections):
            logits[0, i, 1] = torch.logit(torch.tensor(self.score))
            boxes[0, i, 0], boxes[0, i, 1] = cx, cy
        # a trainable dependency so a loss can back-propagate into the box head
        boxes = boxes + 0.0 * self.bbox_embed[0].weight.sum() + 0.0 * self.transformer.decoder.layers[-1].linear.weight.sum()
        return {"pred_logits": logits, "pred_boxes": boxes}


class FakeCriterion(torch.nn.Module):
    weight_dict = {"loss_ce": 5.0, "loss_bbox": 1.0}

    def forward(self, outputs, targets, cat_list, captions):
        boxes = outputs["pred_boxes"][0, : len(targets[0]["boxes"])]
        return {"loss_bbox": (boxes[:, :2] - targets[0]["boxes"][:, :2]).abs().mean(), "loss_ce": boxes.sum() * 0.0 + 0.1}


def _image(size=(160, 120), dots=()):
    image = Image.new("RGB", size, (240, 240, 240))
    draw = ImageDraw.Draw(image)
    for x, y in dots:
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(30, 30, 200))
    return image


def _pipeline(detections=None) -> CountGDPipeline:
    return CountGDPipeline(FakeModel(detections), FakeCriterion(), FakeTokenizer(), device="cpu")


def test_count_reads_confident_queries_as_points_in_input_pixels():
    pipe = _pipeline([(0.25, 0.5), (0.75, 0.5)])
    result = pipe.count(_image(), text="Blue Dot.")
    entry = result["results"][0]
    assert entry["count"] == 2 and entry["text"] == "blue dot" and entry["exemplars"] == []
    assert entry["points"] == [[40.0, 60.0], [120.0, 60.0]] and entry["scores"] == [0.9, 0.9]
    assert entry["boxes"][0][0] < 40.0 < entry["boxes"][0][2] and entry["input_size"] == [160, 120]
    assert entry["model_input_size"][1] == 800  # the shortest side is resized to 800 px
    assert result["model_id"] == MODEL_ID and result["model_revision"] == MODEL_REVISION and result["threshold"] == CONFIDENCE_THRESHOLD
    assert pipe.model.calls[-1]["caption"] == "blue dot ."


def test_threshold_decides_the_count_and_is_validated():
    pipe = _pipeline([(0.5, 0.5)])
    pipe.model.score = 0.3
    assert pipe.count(_image(), text="dot")["results"][0]["count"] == 1
    assert pipe.count(_image(), text="dot", threshold=0.5)["results"][0]["count"] == 0
    for bad in (0.0, 1.0, -0.1, "0.5", True):
        with pytest.raises(ValueError, match="threshold"):
            pipe.count(_image(), text="dot", threshold=bad)


def test_exemplars_are_scaled_to_the_model_input_and_carry_a_caption_when_no_text_is_given():
    pipe = _pipeline()
    image = _image((200, 100))
    result = pipe.count(image, exemplars=[[10, 10, 30, 30], [50, 50, 70, 70]])
    entry = result["results"][0]
    assert entry["exemplars"] == [[10.0, 10.0, 30.0, 30.0], [50.0, 50.0, 70.0, 70.0]] and entry["text"] is None
    assert pipe.model.calls[-1]["exemplars"] == 2 and pipe.model.calls[-1]["caption"] == f"{pl.EXEMPLAR_CARRIER_WORD} ."
    assert pipe.model.calls[-1]["size"] == (666, 1333)  # 200 x 100: the long side caps at 1333 before the short side reaches 800
    both = pipe.count([image, image], text="dot", exemplars=[[[10, 10, 30, 30]], [[10, 10, 30, 30]]])
    assert len(both["results"]) == 2 and pipe.model.calls[-1]["caption"] == "dot ."
    with pytest.raises(ValueError, match="one exemplar list per image"):
        pipe.count([image, image], text="dot", exemplars=[[10, 10, 30, 30]])
    with pytest.raises(ValueError, match="at most 3"):
        pipe.count(image, exemplars=[[1, 1, 5, 5]] * 4)
    with pytest.raises(ValueError, match="outside the image"):
        pipe.count(image, exemplars=[[150, 50, 250, 90]])
    with pytest.raises(ValueError, match="sides must be at least"):
        pipe.count(image, exemplars=[[10, 10, 11, 11]])


def test_prompt_text_and_images_are_validated():
    pipe = _pipeline()
    with pytest.raises(ValueError, match="text prompt, exemplar boxes, or both"):
        pipe.count(_image())
    with pytest.raises(ValueError, match="plain words"):
        pipe.count(_image(), text="dot; DROP TABLE")
    with pytest.raises(ValueError, match="at most 64"):
        pipe.count(_image(), text="x" * 65)
    with pytest.raises(ValueError, match="remote URLs"):
        pipe.count("https://example.invalid/x.png", text="dot")
    with pytest.raises(ValueError, match="at least one image"):
        pipe.count([], text="dot")
    with pytest.raises(ValueError, match=f"at most {MAX_BATCH}"):
        pipe.count([_image()] * (MAX_BATCH + 1), text="dot")
    with pytest.raises(FileNotFoundError):
        pipe.count("nowhere/nothing.png", text="dot")
    with pytest.raises(ValueError, match="sides must be"):
        pipe.count(Image.new("RGB", (8, 8)), text="dot")


def test_validate_inputs_mirrors_the_checks_and_names_inputs():
    manifest = validate_inputs([_image(), _image()], text="Dot", exemplars=[[[1, 1, 9, 9]], None], names=["a", "b"])
    assert [i["id"] for i in manifest["inputs"]] == ["a", "b"] and manifest["verdict"] == "accepted"
    assert manifest["inputs"][0]["exemplars"] == 1 and manifest["inputs"][1]["exemplars"] == 0 and manifest["inputs"][0]["text"] == "dot"
    assert manifest["model_id"] == MODEL_ID and manifest["threshold"] == CONFIDENCE_THRESHOLD
    with pytest.raises(ValueError, match="one entry per image"):
        validate_inputs(_image(), text="dot", names=["a", "b"])
    with pytest.raises(ValueError, match="text prompt, exemplar boxes, or both"):
        validate_inputs([_image(), _image()], exemplars=[[[1, 1, 9, 9]], None])


def test_evaluation_report_reads_counts_against_expected():
    pipe = _pipeline([(0.5, 0.5)])
    result = pipe.count([_image(), _image()], text="dot")
    report = evaluation_report(result, [1, 3], sample_kind="synthetic")
    assert report["verdict"] == "sample-sanity" and report["metrics"][0]["value"] == [1, 1]
    assert report["metrics"][1]["id"] == "mae" and report["metrics"][1]["value"] == 1.0 and report["metrics"][2]["value"] == 1
    with pytest.raises(ValueError, match="one count per result"):
        evaluation_report(result, [1])
    with pytest.raises(ValueError, match="no results"):
        evaluation_report({"results": []})
